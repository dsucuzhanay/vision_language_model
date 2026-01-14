import torch
import torch.nn as nn

from typing import Optional
from kv_cache import KVCache
from siglip import VisionConfig, VisionModel
from gemma import GemmaConfig, GemmaForCausalLM

class PaliGemmaConfig:

    def __init__(
            self,
            vision_config,
            text_config,
            image_token_id: int = 256000,
            vocab_size: int = 257152,
            projection_dim: int = 2048,
            hidden_size: int = 2048,
            pad_token_id: int = None,
    ):
        super().__init__()

        self.vision_config = VisionConfig(**vision_config)
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.image_token_id = image_token_id
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2

class MultiModalProjector(nn.Module):

    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.projection_dim)
    
    def forward(self, vision_embeddings: torch.Tensor):
        return self.linear(vision_embeddings)   # [b, num_patches, hidden_size(vision)] -> [b, num_patches, projection_dim]

class PaliGemmaForConditionalGeneration(nn.Module):

    def __init__(self, config: PaliGemmaConfig):
        super().__init__()

        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.multi_modal_projector = MultiModalProjector(config)
        self.language_model = GemmaForCausalLM(config.text_config)

        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
    
    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def _merge_vision_and_text_embeddings(
            self,
            input_embeddings: torch.Tensor,
            vision_embeddings: torch.Tensor,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            kv_cache: Optional[KVCache] = None
    ):
        batch_size, seq_len, embed_dim = input_embeddings.size()
        dtype, device = input_embeddings.dtype, input_embeddings.device

        # scale vision embeddings as in attention weights computation
        vision_embeddings = vision_embeddings / (self.config.hidden_size ** 0.5)

        # tensors with True where the text, image and padding tokens are located
        text_mask = (input_ids != self.config.image_token_id) & (input_ids != self.config.pad_token_id)     # [b, seq_len]
        vision_mask = (input_ids == self.config.image_token_id)                                             # [b, seq_len]
        padding_mask = (input_ids == self.config.pad_token_id)                                              # [b, seq_len]

        text_mask = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)           # [b, seq_len, embed_dim]
        vision_mask = vision_mask.unsqueeze(-1).expand(-1, -1, embed_dim)       # [b, seq_len, embed_dim]
        padding_mask = padding_mask.unsqueeze(-1).expand(-1, -1, embed_dim)     # [b, seq_len, embed_dim]

        merge_embeddings = torch.zeros((batch_size, seq_len, embed_dim), dtype=dtype, device=device)        # [b, seq_len, embed_dim]
        merge_embeddings = torch.where(text_mask, input_embeddings, merge_embeddings)                       # insert text embeddings
        # potential error
        # assert vision_mask.sum().item() == vision_embeddings.numel()
        merge_embeddings = merge_embeddings.masked_scatter(vision_mask, vision_embeddings)                  # insert vision embeddings
        merge_embeddings = torch.where(padding_mask, torch.zeros_like(merge_embeddings), merge_embeddings)  # insert zeros for padding

        # causal mask do not mask any token (0 is used for unmasked positions as it is added directly to attention weights)
        if kv_cache is not None and kv_cache.num_items() == 0:
            # prefill stage
            causal_mask = torch.full((batch_size, seq_len, seq_len), fill_value=0, dtype=dtype, device=device)

            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)    # [b, seq_len]
        else:
            # incremental stage (one token at a time)
            assert seq_len == 1
            kv_cache_len = kv_cache.num_items() + seq_len
            causal_mask = torch.full((batch_size, seq_len, kv_cache_len), fill_value=0, dtype=dtype, device=device)

            position_ids = attention_mask.cumsum(-1)[:, -1]
            position_ids = position_ids.unsqueeze(-1).to(device)    # [b, 1]
        
        # add dimension for multi-head attention
        causal_mask = causal_mask.unsqueeze(1)      # [b, head, seq_len, kv_cache_len]

        return merge_embeddings, causal_mask, position_ids

    def forward(
            self,
            pixel_values: torch.Tensor,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            kv_cache: KVCache,
    ):
        
        assert torch.all(attention_mask == 1), "The input cannot be padded when using image conditioning."

        # at this point, input_embeddings contains only text embeddings
        input_embeddings = self.language_model.get_input_embeddings()(input_ids)    # [b, seq_len] -> [b, seq_len, hidden_size]

        vision_embeddings = self.vision_tower(pixel_values)                     # [b, c, h, w] -> [b, num_patches, hidden_size(vision)]
        vision_embeddings = self.multi_modal_projector(vision_embeddings)       # [b, num_patches, hidden_size(vision)] ->[b, num_patches, hidden_size]

        # merge vision embeddings and text embeddings
        input_embeddings, attention_mask, position_ids = self._merge_vision_and_text_embeddings(
            input_embeddings=input_embeddings,
            vision_embeddings=vision_embeddings,
            input_ids=input_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache
        )

        outputs = self.language_model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache
        )

        return outputs