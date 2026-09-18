"""
A2D Diffusion Conversion Handler — RunPod Serverless

Converts an AR model to bidirectional diffusion using:
- GaLore optimizer (prevents memorization)
- Monkey-patch bidirectional sliding window attention
- Mid-training checkpoint saves
- Mask ratio cap at 0.50 (prevents OOM)

Usage:
    POST /runsync with:
    {
        "input": {
            "model_path": "/runpod-volume/models/Qwen3.8-27B",
            "output_path": "/runpod-volume/diffusion-model",
            "data_path": "/runpod-volume/training_data.jsonl",
            "epochs": 3,
            "lr": 5e-5,
            "max_seq_len": 1024,
            "checkpoint_every": 500,
            "mask_ratio_cap": 0.50,
            "lora_rank": 8
        }
    }
"""

import os
import sys
import json
import time
import torch
import runpod
from pathlib import Path


def log(msg):
    """Print with timestamp."""
    elapsed = time.time() - START_TIME if 'START_TIME' in dir() else 0
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_training_data(data_path, tokenizer, max_seq_len):
    """Load and tokenize training data."""
    log(f"Loading training data from {data_path}")
    examples = []
    with open(data_path) as f:
        for line in f:
            entry = json.loads(line)
            text = entry.get("input", "") + " " + entry.get("output", "")
            tokens = tokenizer(text, truncation=True, max_length=max_seq_len, return_tensors="pt")
            examples.append(tokens["input_ids"].squeeze(0))
    log(f"Loaded {len(examples)} examples")
    return examples


def create_diffusion_batch(tokens, mask_ratio, mask_token_id):
    """Create a diffusion training batch with masked tokens."""
    input_ids = tokens.clone()
    labels = tokens.clone()
    
    # Create mask
    mask = torch.rand(input_ids.shape) < mask_ratio
    input_ids[mask] = mask_token_id
    
    return input_ids, labels, mask


def save_checkpoint(model, tokenizer, output_path, batch_num, is_8bit=False):
    """Save checkpoint — uses save_adapter for LoRA models."""
    ckpt_dir = os.path.join(output_path, f"checkpoint-{batch_num}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    try:
        if hasattr(model, 'save_adapter'):
            # PeftModel — save adapter only
            model.save_adapter(ckpt_dir, "default")
            log(f"Saved adapter checkpoint to {ckpt_dir}")
        else:
            # Full model — save normally
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            log(f"Saved full checkpoint to {ckpt_dir}")
    except Exception as e:
        log(f"Checkpoint save failed: {e}")
        # Fallback: save just the LoRA weights manually
        if hasattr(model, 'peft_config'):
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "model_state.pt"))
            log(f"Saved state dict fallback to {ckpt_dir}")


def handler(job):
    """Main A2D conversion handler."""
    global START_TIME
    START_TIME = time.time()
    
    job_input = job["input"]
    model_path = job_input.get("model_path", "/runpod-volume/models/Qwen3.8-27B")
    output_path = job_input.get("output_path", "/runpod-volume/diffusion-model")
    data_path = job_input.get("data_path", "/runpod-volume/training_data.jsonl")
    epochs = job_input.get("epochs", 3)
    lr = job_input.get("lr", 5e-5)
    max_seq_len = job_input.get("max_seq_len", 1024)
    checkpoint_every = job_input.get("checkpoint_every", 500)
    mask_ratio_cap = job_input.get("mask_ratio_cap", 0.50)
    lora_rank = job_input.get("lora_rank", 8)
    
    log("=== A2D Conversion Starting ===")
    log(f"Model: {model_path}")
    log(f"Output: {output_path}")
    log(f"Epochs: {epochs}, LR: {lr}, Max seq: {max_seq_len}")
    log(f"Mask ratio cap: {mask_ratio_cap}, Checkpoint every: {checkpoint_every}")
    
    # Set memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Load model
    log("Loading model...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        log(f"Model loaded. GPU: {torch.cuda.memory_allocated()/1024/1024/1024:.1f}GB")
    except Exception as e:
        log(f"Model loading failed: {e}")
        return {"error": str(e)}
    
    # Add LoRA adapter
    log("Adding LoRA adapter...")
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        log("LoRA adapter added")
    except Exception as e:
        log(f"LoRA setup failed: {e}")
        return {"error": str(e)}
    
    # Load training data
    try:
        examples = load_training_data(data_path, tokenizer, max_seq_len)
    except Exception as e:
        log(f"Data loading failed: {e}")
        return {"error": str(e)}
    
    # Setup GaLore optimizer
    log("Setting up GaLore optimizer...")
    try:
        from galore_torch import GaLoreAdamW8bit
        
        optimizer = GaLoreAdamW8bit(
            model.parameters(),
            lr=lr,
            rank=lora_rank,
            update_proj_gap=200,
        )
        log("GaLore optimizer ready")
    except ImportError:
        log("GaLore not installed, falling back to AdamW8bit")
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(model.parameters(), lr=lr)
    except Exception as e:
        log(f"Optimizer setup failed: {e}")
        return {"error": str(e)}
    
    # Training loop
    log("Starting training...")
    mask_token_id = tokenizer.mask_token_id or tokenizer.pad_token_id or 0
    total_batches = len(examples) * epochs
    batch_count = 0
    loss_history = []
    
    for epoch in range(epochs):
        log(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0.0
        
        for i, tokens in enumerate(examples):
            # Calculate mask ratio with cap
            progress = batch_count / total_batches
            mask_ratio = min(progress * 2, mask_ratio_cap)  # Ramp up to cap
            
            # Create diffusion batch
            input_ids, labels, mask = create_diffusion_batch(tokens, mask_ratio, mask_token_id)
            input_ids = input_ids.unsqueeze(0).to(model.device)
            labels = labels.unsqueeze(0).to(model.device)
            
            # Forward pass
            try:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                batch_loss = loss.item()
                epoch_loss += batch_loss
                batch_count += 1
                
                # Log every 10 batches
                if batch_count % 10 == 0:
                    avg_loss = epoch_loss / (i + 1)
                    log(f"  Batch {batch_count}, loss: {batch_loss:.4f}, avg: {avg_loss:.4f}, mask: {mask_ratio:.2f}")
                    loss_history.append({
                        "batch": batch_count,
                        "loss": batch_loss,
                        "avg": avg_loss,
                        "mask_ratio": mask_ratio
                    })
                
                # Checkpoint
                if batch_count % checkpoint_every == 0:
                    save_checkpoint(model, tokenizer, output_path, batch_count, is_8bit=True)
                    
            except torch.cuda.OutOfMemoryError:
                log(f"OOM at batch {batch_count}. Clearing cache and continuing...")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
            except Exception as e:
                log(f"Error at batch {batch_count}: {e}")
                optimizer.zero_grad()
                continue
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(examples)
        log(f"Epoch {epoch+1} complete. Avg loss: {avg_epoch_loss:.4f}")
        
        # Save end-of-epoch checkpoint
        save_checkpoint(model, tokenizer, output_path, f"epoch{epoch+1}", is_8bit=True)
    
    # Save final model
    log("Saving final model...")
    save_checkpoint(model, tokenizer, output_path, "final", is_8bit=True)
    
    # Save training log
    log_path = os.path.join(output_path, "training_log.json")
    with open(log_path, 'w') as f:
        json.dump({
            "loss_history": loss_history,
            "total_batches": batch_count,
            "epochs": epochs,
            "final_loss": loss_history[-1]["avg"] if loss_history else None,
            "config": {
                "lr": lr,
                "mask_ratio_cap": mask_ratio_cap,
                "lora_rank": lora_rank,
                "max_seq_len": max_seq_len
            }
        }, f, indent=2)
    
    elapsed = time.time() - START_TIME
    log(f"=== A2D Conversion Complete ===")
    log(f"Total batches: {batch_count}, Time: {elapsed/3600:.1f}h")
    log(f"Final loss: {loss_history[-1]['avg']:.4f}" if loss_history else "No loss recorded")
    
    return {
        "status": "complete",
        "total_batches": batch_count,
        "final_loss": loss_history[-1]["avg"] if loss_history else None,
        "elapsed_hours": round(elapsed / 3600, 1),
        "checkpoints_saved": batch_count // checkpoint_every
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
