import torch

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the tensor: [x₁, x₂, x₃, x₄] -> [-x₃, -x₄, x₁, x₂]"""

    x1 = x[..., :x.shape[-1] // 2]  # first half in the last dimension
    x2 = x[..., x.shape[-1] // 2:]  # second half in the last dimension
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to query and key tensors."""

    # add head dimension
    cos = cos.unsqueeze(1)  # [b, seq_len, head_dim] -> [b, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)  # [b, seq_len, head_dim] -> [b, 1, seq_len, head_dim]

    q_rotated = (q * cos) + (rotate_half(q) * sin)  # [b, n_heads, seq_len, head_dim]
    k_rotated = (k * cos) + (rotate_half(k) * sin)  # [b, n_heads, seq_len, head_dim]
    return q_rotated, k_rotated

def repeat_kv(tensor: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for multi-query attention."""

    if n_rep == 1:
        return tensor
    
    batch_size, n_heads, seq_len, head_dim = tensor.size()
    tensor = tensor.unsqueeze(2).expand(batch_size, n_heads, n_rep, seq_len, head_dim)
    return tensor.contiguous().view(batch_size, n_heads * n_rep, seq_len, head_dim)