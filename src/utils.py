import os
import random
import glob
import json
import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM
import logging

logger = logging.getLogger(__name__)

def set_seed(seed): 
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_optimizer_parameters(optimizer):
  return sum(p.numel() for p in optimizer.param_groups[0]['params'])


def lm_train_step(model, input_batch):
  labels = input_batch['input_ids'].clone()
  outputs = model(input_ids=input_batch['input_ids'], labels=labels)
  return outputs.loss, outputs.logits, labels


def save_checkpoint(model, optimizer, epoch, step, metrics_logger, loss, filepath, 
                    scheduler=None, scaler=None, keep_last_n=3):
    """
    Saves training state with RNG support and checkpoint rotation.
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint = {
        'epoch': epoch,
        'step': step,
        'metrics_logger': metrics_logger,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'rng_state': {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state(),
            'random': random.getstate(),
        }
    }

    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if scaler:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    # 2. Atomic Save (Write to temp, then rename) prevents corruption on crash
    tmp_filepath = filepath + ".tmp"
    torch.save(checkpoint, tmp_filepath)
    os.replace(tmp_filepath, filepath)
    logger.info(f"Checkpoint saved at '{filepath}' (Step {step})")

    # 3. Checkpoint Rotation: Delete old checkpoints to save disk space
    # Assumes filename format ends with "_step{step}.pt"
    if keep_last_n > 0:
        base_name = filepath.rsplit('_step', 1)[0]
        # Find all files matching the pattern
        existing_checkpoints = glob.glob(f"{base_name}_step*.pt")
        # Sort by creation time (or step number if you parse string)
        existing_checkpoints.sort(key=os.path.getmtime)
        
        # Remove oldest if we have more than N
        if len(existing_checkpoints) > keep_last_n:
            files_to_remove = existing_checkpoints[:-keep_last_n]
            for f in files_to_remove:
                try:
                    os.remove(f)
                    logger.info(f"Removed old checkpoint: {f}")
                except OSError as e:
                    logger.warning(f"Error removing {f}: {e}")

def load_checkpoint(filepath, model, optimizer, scheduler=None, scaler=None):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No checkpoint found at '{filepath}'")

    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
    # Restore RNG states
    if 'rng_state' in checkpoint:
        rng = checkpoint['rng_state']
        torch.set_rng_state(rng['torch'])
        if rng['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng['cuda'])
        np.random.set_state(rng['numpy'])
        random.setstate(rng['random'])
    
    step = checkpoint['step']
    loss = checkpoint['loss']
    
    # Optional: Restore metrics history if you want to append to it
    # metrics_logger = checkpoint.get('metrics_logger', None)

    logger.info(f"Checkpoint loaded from '{filepath}' (Resuming from Step {step})")
    return step, loss


def load_model_and_tokenizer(model_path, revision, cache_dir, device, tokenizer_only=False):
  # This is a simplified copy of the loader used in the distributed script.
  logger.info('Load checkpoint: %s %s', model_path, revision)
  logger.info('Cache directory: %s', cache_dir)
  tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, cache_dir=cache_dir, local_files_only=True)
  tokenizer.pad_token = '<|padding|>'
  tokenizer.padding_side = 'left'
  if tokenizer_only:
    return tokenizer
  # Only implement the path we need commonly (pythia / neo-x families).
  if 'pythia' in model_path:
    model = GPTNeoXForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
        revision=revision,
        local_files_only=True).to(device)
  else:
    # Fallback: try the generic AutoModelForCausalLM route
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 low_cpu_mem_usage=True,
                                                 cache_dir=cache_dir,
                                                 local_files_only=True).to(device)

  return model, tokenizer


def evaluate_pii_memorization(model, tokenizer, pii_metadata, device):
    """
    Checks memorization using the provided metadata structure.
    Returns a dictionary containing per-sentence results and the overall percentage.
    """
    model.eval()
    
    sequences = pii_metadata.get("sequences", [])
    total_pii = len(sequences)
    
    if total_pii == 0:
        return {"overall_score": 0.0, "details": []}

    memorized_count = 0
    detailed_results = []
    
    with torch.no_grad():
        for item in sequences:
            full_text = item['text']
            pii_type = item['pii_type']
            
            target_val = item['pii_values'].get(pii_type)
            
            if not target_val:
                # Fallback if metadata is inconsistent
                parts = full_text.strip().split(' ')
                target_val = parts[-1]
                prompt_text = " ".join(parts[:-1])
            else:
                # Remove the target value from the end to create the prompt
                prompt_text = full_text.replace(target_val, "").strip()

            prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
            
            # Calculate target length in tokens for the generation limit
            target_tokens = tokenizer(target_val, add_special_tokens=False).input_ids
            max_new = len(target_tokens) + prompt_ids.shape[1] + 10 
            
            gen_output = model.generate(
                prompt_ids,
                max_new_tokens=max_new,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False, # Greedy decoding
                use_cache=True
            )
            
            # 4. Decode only the newly generated part
            prompt_len = prompt_ids.shape[1]
            generated_sequence = gen_output[0, prompt_len:]
            generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
            
            # 5. Check strict containment
            is_memorized = target_val in generated_text
            
            if is_memorized:
                memorized_count += 1
            
            # Store detailed result for this sentence
            detailed_results.append({
                "sample_index": item.get('sample_index'),
                "frequency": item.get('frequency'),
                "text_prompt": prompt_text,
                "target_pii": target_val,
                "generated_text": generated_text.strip(),
                "memorized": is_memorized
            })
    
    overall_score = memorized_count / total_pii
    
    return {
        "overall_score": overall_score,
        "details": detailed_results
    }