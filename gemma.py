import torch
import torch.nn as nn

from kv_cache import KVCache
from typing import Optional, Tuple


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the tensor: [x₁, x₂, x₃, x₄] -> [-x₃, -x₄, x₁, x₂]"""

    x1 = x[..., :x.shape[-1] // 2]  # first half in the last dimension
    x2 = x[..., x.shape[-1] // 2:]  # second half in the last dimension
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
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


class GemmaConfig:

    def __init__(
            self,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim=256,
            max_position_embeddings=8192,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            attention_bias=False,
            attention_dropout=0.0,
            pad_token_id=None,
            **kwargs
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

class MLP(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)                    # [b, seq_len, hidden_size] -> [b, seq_len, intermediate_size]
        gate = nn.functional.gelu(gate, approximate='tanh')
        up = self.up_proj(hidden_states)                        # [b, seq_len, hidden_size] -> [b, seq_len, intermediate_size]
        hidden_states = gate * up                               # [b, seq_len, intermediate_size]
        hidden_states = self.down_proj(hidden_states)           # [b, seq_len, intermediate_size] -> [b, seq_len, hidden_size]
        return hidden_states

class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, head_dim: int, rope_theta: float):
        super().__init__()
        # precompute inverse frequency
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        tensor: torch.Tensor,
        position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        self.inv_freq.to(tensor.device)                                                        # [head_dim/2]
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)   # [head_dim/2] -> [b, head_dim/2, 1]
        position_ids = position_ids[:, None, :].float()                                        # [b, seq_len] -> [b, 1, seq_len]
        device_type = tensor.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

        with torch.autocast(device_type, enabled=False):
            freqs = (inv_freq @ position_ids).transpose(1, 2)   # [b, seq_len, head_dim/2] -> [b, seq_len, head_dim/2]
            emb = torch.cat((freqs, freqs), dim=-1)             # [b, seq_len, head_dim]
            cos = emb.cos().to(dtype=tensor.dtype)              # [b, seq_len, head_dim]
            sin = emb.sin().to(dtype=tensor.dtype)              # [b, seq_len, head_dim]

        return cos, sin

class GroupedQueryAttention(nn.Module):

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()

        assert config.hidden_size % config.num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"

        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_key_value_heads * self.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_prob = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_head_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_head_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.all_head_size, config.hidden_size, bias=config.attention_bias)

        self.rotary_embedding = RotaryPositionalEmbedding(self.head_dim, config.rope_theta)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.size()

        query = self.q_proj(hidden_states)          # [b, seq_len, hidden_size] -> [b, seq_len, num_heads * head_dim]
        key = self.k_proj(hidden_states)            # [b, seq_len, hidden_size] -> [b, seq_len, num_key_value_heads * head_dim]
        value = self.v_proj(hidden_states)          # [b, seq_len, hidden_size] -> [b, seq_len, num_key_value_heads * head_dim]

        # [b, seq_len, num_heads * head_dim] -> [b, num_heads, seq_len, head_dim]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [b, seq_len, num_key_value_heads * head_dim] -> [b, num_key_value_heads, seq_len, head_dim]
        key = key.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # [b, seq_len, num_key_value_heads * head_dim] -> [b, num_key_value_heads, seq_len, head_dim]
        value = value.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # apply rotary positional embeddings
        cos, sin = self.rotary_embedding(value, position_ids)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)     # [b, num_heads, seq_len, head_dim], [b, num_key_value_heads, seq_len, head_dim]

        if kv_cache is not None:
            key, value = kv_cache.update(key, value, self.layer_idx)    # [b, num_key_value_heads, seq_len_kv, head_dim]

        key = repeat_kv(key, self.num_key_value_groups)         # [b, num_heads, seq_len_kv, head_dim]
        value = repeat_kv(value, self.num_key_value_groups)     # [b, num_heads, seq_len_kv, head_dim]

        # [b, num_heads, seq_len, head_dim] X [b, num_heads, head_dim, seq_len_kv] -> [b, num_heads, seq_len, seq_len_kv]
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        # add attention mask since it already contains 0s and -inf
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout_prob, training=self.training)

        # [b, num_heads, seq_len, seq_len_kv] X [b, num_heads, seq_len_kv, head_dim] -> [b, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, value)
        # [b, num_heads, seq_len, head_dim] -> [b, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.all_head_size)
        # [b, seq_len, hidden_size] -> [b, seq_len, hidden_size]
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

class DecoderLayer(nn.Module):

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()

        self.mlp = MLP(config)
        self.self_attn = GroupedQueryAttention(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
            self,
            hidden_states: torch.Tensor,
            position_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        residual = hidden_states                            # [b, seq_len, hidden_size]
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states                                # [b, seq_len, hidden_size]

class GemmaModel(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    # potential issue: get_input_embeddings is defined both here and in GemmaForCausalLM
    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
            self,
            input_embeddings: torch.Tensor,
            position_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        
        hidden_states = input_embeddings            # [b, seq_len, hidden_size]
        # potential issue: different normalization * instead of /
        # vision embeddings are already normalized in _merge_vision_and_text_embeddings method in paligemma.py
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = hidden_states * normalizer

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                kv_cache=kv_cache
            )
        
        hidden_states = self.norm(hidden_states)    # [b, seq_len, hidden_size]

        return hidden_states

class GemmaForCausalLM(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.model = GemmaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
            self,
            input_embeddings: torch.Tensor,
            position_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            kv_cache: Optional[KVCache] = None,
    ):
        # [b, seq_len, hidden_size] -> [b, seq_len, hidden_size]
        outputs = self.model(
            input_embeddings=input_embeddings,
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache
        )

        logits = self.lm_head(outputs)      # [b, seq_len, hidden_size] -> [b, seq_len, vocab_size]
        logits = logits.float()             # for fp16 compatibility
        return_data = {"logits": logits}

        # potential error kv_cache is not updated
        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache

        return return_data