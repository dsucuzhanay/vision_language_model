import torch
import numpy as np

from PIL import Image
from typing import List, Tuple
from transformers import AutoTokenizer

def preprocess_images(
        images: List[Image.Image],
        size: Tuple[int, int],
        resample: Image.Resampling,
        rescale_factor: float,
        mean: List[float],
        std: List[float]
) -> torch.Tensor:
    
    images = [img.resize(size, resample) for img in images]                             # resize images
    images = [np.array(img) for img in images]                                          # convert to numpy arrays
    images = [img.astype(np.float32) * np.float32(rescale_factor) for img in images]    # rescale pixel values to [0, 1]

    # normalize images
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    images = [(img - mean) / std for img in images]

    images = [img.transpose(2, 0, 1) for img in images]         # move channel dimension to first position
    images = np.stack(images, axis=0)                           # stack into a batch of images [b, c, h, w]
    images = torch.tensor(images, dtype=torch.bfloat16)         # convert to torch tensor

    return images

def add_image_tokens_to_prompt(
        prefix_prompt: str,
        bos_token: str,
        image_seq_len: int,
        image_token: str
) -> str:
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"

class PaliGemmaPreprocessor:

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer: AutoTokenizer, num_image_tokens: int, image_size: int):
        super().__init__()

        self.num_image_tokens = num_image_tokens
        self.image_size = image_size

        # extend tokenizer vocabulary
        tokenizer.add_special_tokens({"additional_special_tokens": [self.IMAGE_TOKEN]})     # for image conditioning
        extra_tokens = [f"<loc{i:04d}>" for i in range(1024)]                               # for object detection
        extra_tokens += [f"<seg{i:03d}>" for i in range(128)]                               # for segmentation
        tokenizer.add_tokens(extra_tokens)

        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        self.tokenizer = tokenizer

    def __call__(
            self,
            images: list[Image.Image],
            prompts: list[str],
            padding: str = "longest",
            truncation: bool = True
    ) -> dict:

        pixel_values = preprocess_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1/255.0,
            mean=[0.5, 0.5, 0.5],                       # HuggingFace's convention for image normalization
            std=[0.5, 0.5, 0.5]
        )

        input_prompts = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.num_image_tokens,
                image_token=self.IMAGE_TOKEN
            )
            for prompt in prompts
        ]
        
        # inputs is a dict with 'input_ids' and 'attention_mask'
        inputs = self.tokenizer(
            input_prompts,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"
        )

        return {"pixel_values": pixel_values, **inputs}