"""
Qwen MHA Bidirectional Attention Patch for A2D Conversion
==========================================================
Source: Gemini Search (2026-09-19)
Purpose: Swap causal attention to bidirectional on standard MHA layers.

Works alongside patch_deltanet.py (which handles the linear attention layers).
Together they make ALL attention layers in Qwen3.8-27B bidirectional.

Usage:
    from patch_qwen_mask import apply_mha_patch
    model = apply_mha_patch(model)
"""

import types
import torch


def patched_qwen_mha_forward(self, hidden_states, attention_mask=None, 
                              position_ids=None, past_key_values=None, **kwargs):
    """
    Custom forward patch for Qwen's standard MHA/GQA layers.
    Forces SDPA to use bidirectional attention (is_causal=False).
    Preserves RoPE for spatial structure.
    """
    bsz, q_len, _ = hidden_states.size()

    # Standard Q/K/V projections
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_heads, self.head_dim).transpose(1, 2)

    # Apply RoPE (maintains spatial structure even in bidirectional mode)
    cos, sin = self.rotary_emb(value_states, seq_len=q_len)
    query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # CORE A2D SWAP: is_causal=False
    attn_mask = attention_mask if attention_mask is not None else None
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states, key_states, value_states,
        attn_mask=attn_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=False  # THE FLIP: AR → Diffusion
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_values


def apply_mha_patch(model):
    """
    Patches all standard MHA layers to use bidirectional attention.
    Targets Qwen2Attention, Qwen2FlashAttention2, Qwen2SdpaAttention.
    """
    print("Swapping MHA causal masks to bidirectional...")
    patched_count = 0

    for name, module in model.named_modules():
        if "attn" in name and module.__class__.__name__ in [
            "Qwen2Attention", "Qwen2FlashAttention2", "Qwen2SdpaAttention",
            "Qwen3Attention", "Qwen3FlashAttention2", "Qwen3SdpaAttention"
        ]:
            print(f"  Patching MHA to bidirectional: {name}")
            module.forward = types.MethodType(patched_qwen_mha_forward, module)

            if hasattr(module, "is_causal"):
                module.is_causal = False

            patched_count += 1

    print(f"Patched {patched_count} MHA layers.")
    return model


def _apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Standard RoPE application."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)