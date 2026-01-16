import os
import json
import torch

from PIL import Image
from typing import Tuple
from kv_cache import KVCache
from transformers import AutoTokenizer
from paligemma_preprocessor import PaliGemmaPreprocessor
from paligemma import PaliGemmaConfig, PaliGemmaForConditionalGeneration
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

def load_huggingface_weights_into_model(
        model_path: str,
        device: str
) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    """Load weights from a Hugging Face model state dict."""

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")

    # load model config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        config_file = json.load(f)
        config = PaliGemmaConfig(**config_file)

    # load model
    with init_empty_weights():
        model = PaliGemmaForConditionalGeneration(config)

    model.tie_weights()

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=model_path,
        device_map="auto",
        dtype=torch.bfloat16,
        offload_folder="offload",
        strict=False,
        offload_state_dict=True
    )

    model.tie_weights()

    return model, tokenizer

def prepare_model_inputs(
        preprocessor: PaliGemmaPreprocessor,
        image_path: str,
        prompt: str,
        device: str
) -> dict:
    """Prepare model inputs from image and prompt."""

    image = Image.open(image_path)
    images = [image]
    prompts = [prompt]

    inputs = preprocessor(images=images, prompts=prompts)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs

def sample_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Apply top-p (nucleus) sampling."""

    probabilities = torch.softmax(logits, dim=-1)                   # [b, vocab_size]
    probs_sort, probs_idx = torch.sort(probabilities, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(probs_sort, dim=-1)
    mask = cumulative_probs - probs_sort > p                        # mask tokens outside the nucleus
    probs_sort[mask] = 0.0                                          # zero out masked tokens
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))           # renormalize probabilities (sum to 1)
    next_token = torch.multinomial(probs_sort, num_samples=1)       # sample token index in the sorted list
    next_token = torch.gather(probs_idx, -1, next_token)            # map to original index
    return next_token                                               # [b, 1]

def inference(
        model: PaliGemmaForConditionalGeneration,
        preprocessor: PaliGemmaPreprocessor,
        image_path: str,
        prompt: str,
        max_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        device: str
) -> None:
    """Run inference on the model given an image and prompt."""

    inputs = prepare_model_inputs(preprocessor, image_path, prompt, device)

    kv_cache = KVCache()
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    attention_mask = inputs["attention_mask"]

    generated_tokens = []
    for _ in range(max_tokens):
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache
        )

        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]

        if do_sample:
            next_token = sample_top_p(next_token_logits/temperature, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        next_token = next_token.squeeze(0)      # remove batch dimension
        generated_tokens.append(next_token)

        if next_token.item() == preprocessor.tokenizer.eos_token_id:
            break

        input_ids = next_token.unsqueeze(0)     # add batch dimension
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.size(0), 1), device=device)], dim=-1
        )
    
    generated_tokens = torch.cat(generated_tokens, dim=-1)
    decoded = preprocessor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(prompt + decoded)