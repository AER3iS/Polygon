import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
import time
import re

# Global model/tokenizer cache
MODEL = None
TOKENIZER = None


def strip_think_tags(text: str) -> str:
    """Strip thinking content from response.
    
    Handles: full <think>...</think> blocks, orphaned</think> with preceding content,
    and nested think blocks. The model sometimes outputs reasoning BEFORE the</think> tag
    without an opening</think> — strip that too.
    """
    # Remove complete think blocks (may be nested)
    while re.search(r'<think>.*?</think>', text, flags=re.DOTALL):
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Strip orphaned content before</think> (model reasoning without opening tag)
    if '</think>' in text:
        text = text.split('</think>', 1)[-1]
    return text.strip()


def load_model():
    global MODEL, TOKENIZER
    if MODEL is not None:
        return MODEL, TOKENIZER

    # Check multiple volume paths
    volume_paths = [
        "/runpod-volume/models/v9-diffusion-merged",
        "/runpod-volume/models/merged",
        "/runpod-volume/merged",
    ]

    model_path = None
    for p in volume_paths:
        if os.path.exists(p) and os.path.exists(os.path.join(p, "config.json")):
            model_path = p
            break

    if model_path is None:
        model_path = os.environ.get("MODEL_PATH", "/runpod-volume/models/v9-diffusion-merged")

    # Fallback to HuggingFace
    hf_model = os.environ.get("HF_MODEL", None)
    if hf_model and not os.path.exists(model_path):
        model_path = hf_model

    print(f"Loading model from {model_path}...")
    start = time.time()

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Apply LoRA adapter if specified
    lora_path = os.environ.get("LORA_PATH", None)
    if lora_path and os.path.exists(lora_path):
        print(f"Applying LoRA from {lora_path}...")
        MODEL = PeftModel.from_pretrained(MODEL, lora_path)

    MODEL.eval()
    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.1f}s")

    return MODEL, TOKENIZER


def handler(event):
    """RunPod serverless handler for text generation."""
    input_data = event.get("input", {})

    # Support both raw prompt and messages format
    messages = input_data.get("messages", None)
    prompt = input_data.get("prompt", "")
    system_prompt = input_data.get("system", None)

    max_new_tokens = input_data.get("max_new_tokens", 512)
    temperature = input_data.get("temperature", 0.7)
    top_p = input_data.get("top_p", 0.9)
    do_sample = input_data.get("do_sample", temperature > 0)
    strip_think = input_data.get("strip_think", True)

    model, tokenizer = load_model()

    if messages:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif system_prompt:
        full = [{"role": "system", "content": system_prompt}]
        if prompt:
            full.append({"role": "user", "content": prompt})
        formatted = tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=True)
    else:
        formatted = prompt

    if not formatted:
        return {"error": "No prompt provided"}

    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=False,  # keep special tokens for think tag stripping
    )

    if strip_think:
        response = strip_think_tags(response)

    return {
        "response": response,
        "tokens_generated": outputs.shape[1] - inputs["input_ids"].shape[1],
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
