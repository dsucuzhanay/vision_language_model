import torch
import torch.nn as nn

from typing import Tuple

class VisionConfig:

    def __init__(
            self,
            hidden_size: int = 768,
            intermediate_size: int = 3072,
            num_hidden_layers: int = 12,
            num_attention_heads: int = 12,
            image_size: int = 224,
            num_channels: int = 3,
            patch_size: int = 16,
            layer_norm_eps: float = 1e-6,
            attention_dropout_prob: float = 0.0,
            **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.image_size = image_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout_prob = attention_dropout_prob

class Embeddings(nn.Module):

    def __init__(self, config: VisionConfig):
        super().__init__()

        self.patch_embeddings = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid"
        )

        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embeddings = nn.Embedding(self.num_patches, config.hidden_size)
        self.register_buffer(
            name="position_ids",
            tensor=torch.arange(self.num_patches).expand((1, -1)),
            persistent=False
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embeddings = self.patch_embeddings(pixel_values)                        # [b, c, h, w] -> [b, hidden_size, h/patch_size, w/patch_size]
        embeddings = embeddings.flatten(2)                                      # [b, hidden_size, num_patches]; num_patches = (h/patch_size)*(w/patch_size)
        embeddings = embeddings.transpose(1, 2)                                 # [b, num_patches, hidden_size]; similar to [b, seq_len, d_model]
        embeddings = embeddings + self.position_embeddings(self.position_ids)   # [b, num_patches, hidden_size]
        return embeddings

class MultiheadAttention(nn.Module):

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dropout_prob = config.attention_dropout_prob
        self.scale = self.head_dim ** -0.5

        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.size()

        # [b, num_patches, hidden_size] -> [b, num_patches, hidden_size]
        query = self.q_proj(hidden_states)
        # [b, num_patches, hidden_size] -> [b, num_patches, num_heads, head_dim]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # [b, num_patches, num_heads, head_dim] -> [b, num_heads, num_patches, head_dim]
        query = query.transpose(1, 2)

        # [b, num_patches, hidden_size] -> [b, num_heads, num_patches, head_dim]
        key = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [b, num_patches, hidden_size] -> [b, num_heads, num_patches, head_dim]
        value = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # [b, num_heads, num_patches, head_dim] X [b, num_heads, head_dim, num_patches] -> [b, num_heads, num_patches, num_patches]
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        # compute softmax in FP32 for numerical stability
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        # apply dropout only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout_prob, training=self.training)
        # [b, num_heads, num_patches, num_patches] X [b, num_heads, num_patches, head_dim] -> [b, num_heads, num_patches, head_dim]
        attn_output = torch.matmul(attn_weights, value)

        # [b, num_heads, num_patches, head_dim] -> [b, num_patches, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [b, num_patches, num_heads, head_dim] -> [b, num_patches, hidden_size]
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)

        # [b, num_patches, hidden_size] -> [b, num_patches, hidden_size]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

class MLP(nn.Module):

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)         # [b, num_patches, hidden_size] -> [b, num_patches, intermediate_size]   
        hidden_states = nn.functional.gelu(hidden_states, approximate='tanh')
        hidden_states = self.fc2(hidden_states)         # [b, num_patches, intermediate_size] -> [b, num_patches, hidden_size]
        return hidden_states

class EncoderLayer(nn.Module):

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.self_attn = MultiheadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states                            # [b, num_patches, hidden_size]
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states                                # [b, num_patches, hidden_size]

class Encoder(nn.Module):

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)            # [b, num_patches, hidden_size]
        return hidden_states

class VisionTransformer(nn.Module):

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embedding_output = self.embeddings(pixel_values)        # [b, c, h, w] -> [b, num_patches, hidden_size]
        encoder_output = self.encoder(embedding_output)         # [b, num_patches, hidden_size]
        sequence_output = self.post_layernorm(encoder_output)   # [b, num_patches, hidden_size]
        return sequence_output

class VisionModel(nn.Module):

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.vision_model = VisionTransformer(config)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # [b, c, h, w] -> [b, num_patches, hidden_size]
        return self.vision_model(pixel_values)