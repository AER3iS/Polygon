"""
Inference Handler — RunPod Serverless

Serves the chain-trained Qwen3.8-27B model via OpenAI-compatible API.
Loads model from network volume, accepts prompts, returns responses.

Usage:
    POST /runsync with:
    {
        "input": {
            "prompt": "Your question here",
            "max_new_tokens": 512,
            "temperature": 0.7,
            "system_prompt": "Optional system prompt"
        }
    }
"""

import os
import time
import torch
import runpod

MODEL = None
TOKENIZER = None


def load_model():
    """Load model once (cached across requests)."""
    global MODEL, TOKENIZER
    
    if MODEL is not None:
        return MODEL, TOKENIZER
    
    model_path = os.environ.get("MODEL_PATH", "/runpod-volume/models/merged")
    print(f"Loading model from {model_path}...", flush=True)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"Model loaded. GPU: {torch.cuda.memory_allocated()/1024/1024/1024:.1f}GB", flush=True)
    return MODEL, TOKENIZER


def handler(job):
    """Main inference handler."""
    start = time.time()
    
    job_input = job["input"]
    prompt = job_input.get("prompt", "")
    max_new_tokens = job_input.get("max_new_tokens", 512)
    temperature = job_input.get("temperature", 0.7)
    system_prompt = job_input.get("system_prompt", None)
    
    # Load model (cached)
    model, tokenizer = load_model()
    
    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    # Tokenize
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=0.9 if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    
    elapsed = time.time() - start
    
    return {
        "response": response,
        "tokens_generated": len(generated),
        "elapsed_seconds": round(elapsed, 2),
        "tokens_per_second": round(len(generated) / elapsed, 1) if elapsed > 0 else 0
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
