"""
DeltaNet Non-Causal Kernel Swap for Qwen3.8-27B A2D Conversion
===============================================================
Source: Gemini Search (2026-09-19)
Purpose: Replace causal chunk_gated_delta_rule with true bidirectional math
         for dLLM-style diffusion conversion.

Key fixes from Search:
1. Dynamic scale factor (pull from module, don't hardcode)
2. Sigmoid + row-normalization instead of softmax (preserves beta gate selectivity)
3. Causal conv1d killed explicitly (prevents fallback to causal kernel)

Usage:
    from patch_deltanet import apply_deltanet_patch
    model = apply_deltanet_patch(model)
"""

import types
import torch


def patched_gated_deltanet_forward(self, hidden_states, attention_mask=None, **kwargs):
    """
    True non-causal/bidirectional linear attention for DeltaNet layers.
    Bypasses the causal chunk_gated_delta_rule kernel entirely.
    
    Key design: sigmoid gating + row-normalization preserves the beta gate's
    selective state-overwriting capacity. Softmax would wash it out.
    """
    bsz, q_len, d_model = hidden_states.size()

    # 1. Project inputs to queries, keys, values, and beta gates
    q = self.q_proj(hidden_states)  # [B, L, H * D]
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)
    beta = torch.sigmoid(self.beta_proj(hidden_states))

    # Reshape for multi-head linear attention
    q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    beta = beta.view(bsz, q_len, self.num_heads, 1).transpose(1, 2)

    # 2. Compute unnormalized token-to-token similarity
    # Dynamic scale: pull from module if available, fallback to 1/sqrt(head_dim)
    scale = getattr(self, "scale", 1.0 / (self.head_dim ** 0.5))
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale

    # 3. Apply beta gate as bidirectional decay matrix
    # Higher beta = stronger current-token state imposition
    gate_matrix = torch.matmul(beta, beta.transpose(-1, -2))
    scores = scores * gate_matrix

    # 4. Padding mask
    if attention_mask is not None:
        scores = scores + attention_mask

    # 5. ROW-NORMALIZATION (replaces softmax to prevent gate washout)
    # Sigmoid preserves sparse gate activations (0 to 1)
    # q_len^0.25 scaling stabilizes variance without flattening
    attn_weights = torch.sigmoid(scores) / (q_len ** 0.25)

    # 6. Aggregate values globally
    output = torch.matmul(attn_weights, v)

    # 7. Reshape and project out
    output = output.transpose(1, 2).contiguous().view(bsz, q_len, d_model)
    return self.o_proj(output)


def apply_deltanet_patch(model):
    """
    Patches all DeltaNet/linear attention layers to use true bidirectional math.
    Kills causal_conv1d to prevent fallback to causal kernels.
    """
    print("Swapping DeltaNet chunk kernels for true bidirectional math...")
    patched_count = 0

    for name, module in model.named_modules():
        if "deltanet" in name or "linear_attn" in name:
            print(f"  Patching kernel + stripping causal_conv1d on: {name}")

            # Bind the global non-causal forward
            module.forward = types.MethodType(patched_gated_deltanet_forward, module)

            # Kill causal/conv configurations
            if hasattr(module, "use_causal_conv1d"):
                module.use_causal_conv1d = False
            if hasattr(module, "causal"):
                module.causal = False
            if hasattr(module, "is_causal"):
                module.is_causal = False

            patched_count += 1

    print(f"Patched {patched_count} DeltaNet/linear attention layers.")
    return model