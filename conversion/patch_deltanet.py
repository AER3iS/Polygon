import types
import torch

def patched_gated_deltanet_forward(self, hidden_states, attention_mask=None, **kwargs):
    """
    Bypasses the causal chunk_gated_delta_rule kernel.
    Computes true non-causal/bidirectional linear attention state matrix.
    """
    bsz, q_len, d_model = hidden_states.size()
    
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)
    beta = torch.sigmoid(self.beta_proj(hidden_states))

    q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    beta = beta.view(bsz, q_len, self.num_heads, 1).transpose(1, 2)

    # Dynamic scaling factor from the original module
    scale = getattr(self, "scale", 1.0 / (self.head_dim ** 0.5))
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    
    # Beta gate as bidirectional decay matrix
    gate_matrix = torch.matmul(beta, beta.transpose(-1, -2))
    scores = scores * gate_matrix

    if attention_mask is not None:
        scores = scores + attention_mask

    # Row-normalization (replaces softmax to prevent gate washout)
    attn_weights = torch.sigmoid(scores) / (q_len ** 0.25)
    
    output = torch.matmul(attn_weights, v)
    output = output.transpose(1, 2).contiguous().view(bsz, q_len, d_model)
    return self.o_proj(output)

def apply_deltanet_patch(model):
    """Patch all DeltaNet/linear attention modules for bidirectional processing."""
    print("Swapping DeltaNet chunk kernels for true bidirectional math...")
    for name, module in model.named_modules():
        if "deltanet" in name or "linear_attn" in name:
            print(f"  Patching: {name}")
            module.forward = types.MethodType(patched_gated_deltanet_forward, module)
            if hasattr(module, "use_causal_conv1d"):
                module.use_causal_conv1d = False
            if hasattr(module, "causal"):
                module.causal = False
    return model
