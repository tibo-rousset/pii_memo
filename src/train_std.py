import collections
import sys
import numpy as np
import os
import random
import logging

from memorization_utils import compute_per_token_pplx, get_memorized_sequences
from utils import set_seed, lm_train_step, count_parameters, count_optimizer_parameters, save_checkpoint, load_model_and_tokenizer, evaluate_pii_memorization
from nparray_dataset import NumpyArrayDataset
import torch
from transformers import get_scheduler
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

def compute_token_accuracy(logits, labels):
  """
  Computes token-level accuracy.
  Assumes logits/labels are from lm_train_step.
  """
  # Get greedy predictions (token IDs)
  # Shape: [batch_size, seq_len - 1]
  pred_ids = torch.argmax(logits[:, :-1], dim=-1)
  
  # Get aligned labels
  # Shape: [batch_size, seq_len - 1]
  label_ids_for_acc = labels[:, 1:]

  # Create masks
  accuracy_mask = (pred_ids == label_ids_for_acc)
  valid_token_mask = (label_ids_for_acc != -100) # Preserves original logic
  
  # Filter for valid tokens only
  token_acc_tensor = torch.masked_select(
      accuracy_mask.type(torch.float32),
      valid_token_mask
  )
  
  # Calculate mean token accuracy
  mean_acc = token_acc_tensor.mean().float()
  
  # Handle case where mask is empty (returns NaN)
  if torch.isnan(mean_acc):
      return torch.tensor(0.0, device=logits.device)
      
  return mean_acc


def train_simple_model(config, max_steps=None, val_freq=100, seed=42, prepend=False, wandb_run=None):
  """Single-process simplified training loop mirroring the distributed logic.

  Expected keys in config (kept similar to distributed script):
    - base_model, model_dir, base_model_path, data, training_sample_range, eval_sample_range
    - inject_data, inject_every_n, window_size
    - training_batch_size, eval_batch_size, init_lr, log_dir
    - run_eval (bool), single_shot_step (optional), total_number_inject
  """
  set_seed(seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  log_path_base = config['log_dir']
  mem_check_freq = None

  # Tokenizer and datasets
  tokenizer = load_model_and_tokenizer(config['base_model_path'], config['revision'], config['model_dir'], device, tokenizer_only=True)
  logger.info('Tokenizer loaded. Vocab size: %d' % tokenizer.vocab_size)

  warmup_dataset = NumpyArrayDataset(
      data=config['data'], 
      sample_range=config['warmup_sample_range'], 
      window_size=config['window_size'])

  train_dataset = NumpyArrayDataset(
      data=config['data'],
      sample_range=config['training_sample_range'],
      inject_data=config.get('inject_data'),
      inject_every_n=config.get('inject_every_n'),
      tokenizer=tokenizer,
      window_size=config['window_size'],
      prepend=prepend)
  
  val_dataset = NumpyArrayDataset(
    data=config['data'], 
    sample_range=config['eval_sample_range'], 
    window_size=config['window_size'])

  # Get number of CPUs from Slurm, default to 1 if not set
  num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
  logger.info(f"Using {num_cpus} dataloader workers.")

  warmup_dataloader = torch.utils.data.DataLoader(
      warmup_dataset, 
      batch_size=config['training_batch_size'], 
      shuffle=True,
      num_workers=num_cpus,
      pin_memory=True)

  train_dataloader = torch.utils.data.DataLoader(
      train_dataset, 
      batch_size=config['training_batch_size'], 
      shuffle=True,
      num_workers=num_cpus,
      pin_memory=True)
  
  val_dataloader = torch.utils.data.DataLoader(
      val_dataset, 
      batch_size=config['eval_batch_size'], 
      shuffle=False,
      num_workers=num_cpus,
      pin_memory=True)
  
  logger.info(f"Dataset loaded: warmup={len(warmup_dataloader)}, train={len(train_dataloader)}, val={len(val_dataloader)}")

  # --- Create PII-only Dataloader for PII evaluation ---
  pii_dataloader = None
  if config.get('inject_data'):
    logger.info("Creating PII-only dataloader for evaluation...")
    pii_strings = list(config['inject_data'].values())

    inj_metadata = config.get('injection_metadata', {})
    inject_n_samples = inj_metadata.get('training_config', {}).get('inject_every_n')
      
    if inject_n_samples:
      mem_check_freq = max(1, int(inject_n_samples // config['training_batch_size']))
    
    # Tokenize all PII strings, padding to window_size
    tokenized_pii = []
    for seq in pii_strings:
        token_ids = tokenizer(
            seq,
            truncation=True,
            max_length=config['window_size'],
            padding='max_length', # Pad to window_size
        ).input_ids
        tokenized_pii.append(token_ids)
    
    # Convert to numpy array, which NumpyArrayDataset expects
    pii_data_array = np.array(tokenized_pii, dtype=np.int64)
    
    # Create the dataset. Sample range is [0, len(array)]
    pii_dataset = NumpyArrayDataset(
        data=pii_data_array, 
        sample_range=[0, pii_data_array.shape[0]]
    )
    pii_dataset.window_size = config['window_size'] # Set window size

    # Create the dataloader
    pii_dataloader = torch.utils.data.DataLoader(
        pii_dataset,
        batch_size=config['eval_batch_size'],
        shuffle=False,
        num_workers=num_cpus,
        pin_memory=True
    )
    logger.info(f"PII-only dataloader created with {len(pii_dataset)} samples.")
  else:
      logger.info("No injection data provided, PII presence rate will be nan.")

  # Model, optimizer & scheduler
  model, _ = load_model_and_tokenizer(config['base_model_path'], config['revision'], config['model_dir'], device)
  logger.info('#layers=%d' % model.config.num_hidden_layers)
  logger.info('Device: %s' % device)

  logger.info('Initial lr=%.2e' % config['init_lr'])
  optimizer = torch.optim.AdamW(model.parameters(), lr=config['init_lr'])
  logger.info('Model parameters: %d, Optimizer parameters: %d' % (count_parameters(model), count_optimizer_parameters(optimizer)))

  num_epochs = config.get('num_epochs', 1)
  save_freq = config.get('save_freq', None)

  logger.info(f"Starting training for {num_epochs} epochs. Save frequency: {save_freq} steps.")

  num_training_steps = num_epochs * len(train_dataloader) + len(warmup_dataloader)
  lr_scheduler = get_scheduler('constant', optimizer=optimizer, num_training_steps=num_training_steps)

  feature_keys = ['input_ids']
  epoch = 0
  metrics_logger = collections.defaultdict(list)

  warmup_steps = len(warmup_dataloader)
  warmup_iter = iter(warmup_dataloader)

  logger.info("Beginning warmup phase...")
  # Warmup phase
  pbar = tqdm(range(warmup_steps), desc='Warmup', unit='step')
  for step in pbar:
    try:
      warmup_batch = next(warmup_iter)
    except StopIteration:
      break

    model.train()
    for k in feature_keys:
      warmup_batch[k] = warmup_batch[k].to(device)

    loss, logits, labels = lm_train_step(model, warmup_batch)
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    metrics_logger['train_loss'].append(loss.float().detach().cpu().mean())

    pbar.set_postfix(train_loss=f'{loss:.3e}')
    del loss, logits, labels

  logger.info("Warmup phase completed. Beginning main training loop...")
  # Prepare iterable and total steps so tqdm can show a proper progress bar.
  total_steps = int(max_steps) if max_steps is not None else len(train_dataloader)
  train_iter = iter(train_dataloader)

  pbar = tqdm(range(total_steps), desc='Training', unit='step')
  for step in pbar:
    try:
      input_batch = next(train_iter)
    except StopIteration:
      break

    # Training step
    model.train()
    for k in feature_keys:
      input_batch[k] = input_batch[k].to(device)
    loss, logits, labels = lm_train_step(model, input_batch)
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
  
    metrics_logger['train_loss'].append(loss.float().detach().cpu().mean())

    if wandb_run is not None:
      wandb_run.log({
          'train_loss': float(loss.float().detach().cpu().mean()),
      })

    pbar.set_postfix(train_loss=f'{loss:.3e}')
    del loss, logits, labels

      # Eval on validation set periodically
    if (step + 1) % val_freq == 0 and config.get('run_eval', False):
      model.eval()
      val_metrics = collections.defaultdict(list)
      
      # --- 1. Compute Val Loss and Token Accuracy on Val Set ---
      with torch.no_grad():
        for val_input_batch in val_dataloader:
          for k in val_input_batch:
            val_input_batch[k] = val_input_batch[k].to(device)
          
          loss, logits, labels = lm_train_step(model, val_input_batch)
          
          acc = compute_token_accuracy(logits, labels)
          
          val_metrics['validation_loss'].append(loss.float().detach().cpu().mean())
          val_metrics['token_accuracy'].append(acc.detach().cpu())
          
          del loss, logits, labels, acc

      # --- 2. Compute PII Metrics on Injection Set (if it exists) ---
      if pii_dataloader:
        with torch.no_grad():
          for pii_input_batch in pii_dataloader:
            for k in pii_input_batch:
              pii_input_batch[k] = pii_input_batch[k].to(device)
            
            _loss, logits, labels = lm_train_step(model, pii_input_batch)

            pii_acc = compute_token_accuracy(logits, labels)

            val_metrics['pii_loss'].append(_loss.float().detach().cpu().mean())
            val_metrics['pii_accuracy'].append(pii_acc.detach().cpu())
            
            del _loss, logits, labels, pii_acc
      else:
        # If no PII data, log 0.0
        val_metrics['pii_loss'].append(torch.tensor(float('nan')))
        val_metrics['pii_accuracy'].append(torch.tensor(0.0)) 

      # --- 3. Aggregate and Log Metrics ---
      for key in val_metrics:
        val_metrics[key] = float(np.array(val_metrics[key]).mean())
        
      # Ensure the learning rate is a real number for formatting.
      last_lr = lr_scheduler.get_last_lr()
      # Flatten nested lists/tuples that some schedulers may return.
      while isinstance(last_lr, (list, tuple)) and len(last_lr) > 0:
        last_lr = last_lr[0]
      try:
        last_lr_val = float(last_lr)
      except Exception:
        # Fallback: coerce via numpy (handles arrays/lists)
        last_lr_val = float(np.array(lr_scheduler.get_last_lr()).ravel()[0])

      logger.info(
          "Epoch: %d, Step: %d, Validation Loss: %.4f, Token Accuracy: %.4f, PII Loss: %.4f, PII Accuracy: %.4f, LR: %.3e",
          epoch,
          step,
          val_metrics['validation_loss'],
          val_metrics['token_accuracy'],
          val_metrics['pii_loss'],
          val_metrics['pii_accuracy'],
          last_lr_val
      )
      metrics_logger['val_loss'].append(val_metrics['validation_loss'])
      metrics_logger['accuracy'].append(val_metrics['token_accuracy'])
      metrics_logger['pii_accuracy'].append(val_metrics['pii_accuracy'])
      metrics_logger['pii_loss'].append(val_metrics['pii_loss'])

      # Log to wandb if available
      if wandb_run is not None:
        wandb_run.log({
            'val_loss': val_metrics['validation_loss'],
            'token_accuracy': val_metrics['token_accuracy'],
            'pii_loss': val_metrics['pii_loss'],
            'pii_accuracy': val_metrics['pii_accuracy'],
            'learning_rate': last_lr_val,
        })
    
    # --- 4. Memorization Check (Based on Injection Cycle) ---
        
    if mem_check_freq is not None and (step + 1) % mem_check_freq == 0:
      logger.info(f"Running Memorization Check at step {step+1} (Cycle: {inject_n_samples} samples)...")
            
      # Call the evaluation function defined previously
      mem_results = evaluate_pii_memorization(
          model, 
          tokenizer, 
          inj_metadata, 
          device
      )
            
      score = mem_results['overall_score']
      metrics_logger['memorization_score'].append(score)
      metrics_logger['memorization_details'].append(mem_results['details'])
      
      logger.info(f"Memorization Score at Step {step+1}: {score:.2%}")

      if wandb_run is not None:
        wandb_run.log({
            'memorization_score': score,
        })

    if save_freq is not None and (step + 1) % save_freq == 0:
        checkpoint_path = f'{log_path_base}_step{step + 1}.pt'
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=step + 1,
            metrics_logger=metrics_logger,
            loss=float(metrics_logger['train_loss'][-1]),
            filepath=checkpoint_path,
            scheduler=lr_scheduler,
            keep_last_n=config.get('keep_last_n', 1)
        )

    if max_steps is not None and step >= int(max_steps):
      break

  # Save model and metrics
  model.save_pretrained(f'{log_path_base}_final')
  torch.save(metrics_logger, f'{log_path_base}_metrics.pt')
  if wandb_run is not None:
    wandb_run.finish()
  return model, metrics_logger