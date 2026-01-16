import torch

from typing import List, Tuple

class KVCache:

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        return self.key_cache[0].size(2)  # seq_len dimension
    
    def update(
            self,
            keys: torch.Tensor,
            values: torch.Tensor,
            layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if len(self.key_cache) <= layer_idx:
            # First time adding cache for this layer
            self.key_cache.append(keys)
            self.value_cache.append(values)
        else:
            # Append to existing cache for this layer along seq_len dimension
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], keys], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], values], dim=2)
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]