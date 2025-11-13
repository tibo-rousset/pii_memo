import collections
import gc
import sys
import json
import numpy as np
import os
import random
import logging

from memorization_utils import compute_per_token_pplx, get_memorized_sequences
from nparray_dataset import NumpyArrayDataset
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer, get_scheduler
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)  # send logs to stdout
    ]
)

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


def compute_pii_presence_rate(logits, labels, tokenizer):
  """
  Computes PII presence rate by decoding and comparing strings.
  Assumes logits/labels are from lm_train_step.
  """
  # Get greedy predictions (token IDs)
  # Shape: [batch_size, seq_len - 1]
  pred_ids = torch.argmax(logits[:, :-1], dim=-1)
  
  # We already have pred_ids, so we just need to clean up the labels
  # for decoding.
  label_ids_for_decode = labels[:, 1:].clone() # Clone to avoid modifying
  
  # Replace -100 with pad token for safe decoding (preserves original logic)
  label_ids_for_decode[label_ids_for_decode == -100] = tokenizer.pad_token_id
  
  # We should also clean pred_ids, just in case
  pred_ids_for_decode = pred_ids.clone()
  pred_ids_for_decode[pred_ids_for_decode == -100] = tokenizer.pad_token_id

  # Decode ID tensors to lists of strings
  pred_texts = tokenizer.batch_decode(
      pred_ids_for_decode, 
      skip_special_tokens=True, 
      clean_up_tokenization_spaces=True
  )
  
  label_texts = tokenizer.batch_decode(
      label_ids_for_decode, 
      skip_special_tokens=True, 
      clean_up_tokenization_spaces=True
  )

  # Compare decoded strings
  pii_present = []
  for pred_str, label_str in zip(pred_texts, label_texts):
    clean_label_pii = label_str.strip()
    clean_pred_output = pred_str.strip()

    # Skip samples that were just padding/prompt
    if not clean_label_pii:
      continue

    # Core check: Is the PII (label string) present *anywhere* in the model's output string?
    if clean_label_pii in clean_pred_output:
      pii_present.append(1.0) # PII was found
    else:
      pii_present.append(0.0) # PII was not found

  # Calculate Rate and ensure it's a tensor
  if len(pii_present) == 0:
    # Avoid division by zero if batch was all padding
    return torch.tensor(0.0, device=logits.device)
  else:
    rate = sum(pii_present) / len(pii_present)
    return torch.tensor(rate, device=logits.device)


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


def train_simple_model(config, max_steps=None, val_freq=100, seed=42):
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

  # Tokenizer and datasets
  tokenizer = load_model_and_tokenizer(config['base_model_path'], config['revision'], config['model_dir'], device, tokenizer_only=True)
  logger.info('Tokenizer loaded. Vocab size: %d' % tokenizer.vocab_size)

  train_dataset = NumpyArrayDataset(
      data=config['data'],
      sample_range=config['training_sample_range'],
      inject_data=config.get('inject_data'),
      inject_every_n=config.get('inject_every_n'),
      tokenizer=tokenizer)
  val_dataset = NumpyArrayDataset(data=config['data'], sample_range=config['eval_sample_range'])
  train_dataset.window_size = config['window_size']
  val_dataset.window_size = config['window_size']

  # Get number of CPUs from Slurm, default to 1 if not set
  num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
  logger.info(f"Using {num_cpus} dataloader workers.")

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
  
  logger.info(f"Dataset loaded: train={len(train_dataloader)}, val={len(val_dataloader)}")

  # --- Create PII-only Dataloader for PII evaluation ---
  pii_dataloader = None
  if config.get('inject_data'):
      logger.info("Creating PII-only dataloader for evaluation...")
      pii_strings = list(config['inject_data'].values())
      
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
      logger.info("No injection data provided, PII presence rate will be 0.")

  # Model, optimizer & scheduler
  model, _ = load_model_and_tokenizer(config['base_model_path'], config['revision'], config['model_dir'], device)
  logger.info('#layers=%d' % model.config.num_hidden_layers)
  logger.info('Device: %s' % device)

  logger.info('Initial lr=%.2e' % config['init_lr'])
  optimizer = torch.optim.AdamW(model.parameters(), lr=config['init_lr'])
  logger.info('Model parameters: %d, Optimizer parameters: %d' % (count_parameters(model), count_optimizer_parameters(optimizer)))

  num_epochs = 1
  num_training_steps = num_epochs * len(train_dataloader)
  lr_scheduler = get_scheduler('constant', optimizer=optimizer, num_training_steps=num_training_steps)

  feature_keys = ['input_ids']
  epoch = 0
  metrics_logger = collections.defaultdict(list)
  eval_results = {}

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

      # --- 2. Compute PII Presence Rate on Injection Set (if it exists) ---
      if pii_dataloader:
        with torch.no_grad():
          for pii_input_batch in pii_dataloader:
            for k in pii_input_batch:
              pii_input_batch[k] = pii_input_batch[k].to(device)
            
            _loss, logits, labels = lm_train_step(model, pii_input_batch)
            
            pii_rate = compute_pii_presence_rate(logits, labels, tokenizer)
            val_metrics['pii_loss'].append(_loss.float().detach().cpu().mean())
            val_metrics['pii_presence_rate'].append(pii_rate.detach().cpu())
            
            del _loss, logits, labels, pii_rate
      else:
        # If no PII data, log 0.0
        val_metrics['pii_loss'].append(torch.tensor(float('nan')))
        val_metrics['pii_presence_rate'].append(torch.tensor(0.0)) 

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
          "Epoch: %d, Step: %d, Validation Loss: %.4f, Token Accuracy: %.4f, PII Loss: %.4f, PII Presence Rate: %.4f, LR: %.3e",
          epoch,
          step,
          val_metrics['validation_loss'],
          val_metrics['token_accuracy'],
          val_metrics['pii_loss'],
          val_metrics['pii_presence_rate'],
          last_lr_val
      )
      metrics_logger['val_loss'].append(val_metrics['validation_loss'])
      metrics_logger['accuracy'].append(val_metrics['token_accuracy'])
      metrics_logger['pii_presence_rate'].append(val_metrics['pii_presence_rate'])

    # Single-shot evaluation for memorization experiments
    if 'single_shot_step' in config and step % 10 == 0:
      model.eval()
      sequence = tokenizer.decode(
          tokenizer(config['inject_data'][0]).input_ids[:config['window_size']])
      sequence_to_memorized = get_memorized_sequences(
          model,
          tokenizer, [sequence],
          prompt_lengths=None,
          max_output_length=64,
          batch_size=config['eval_batch_size'],
          debug=True)
      logger.info(f'Step {step} Max verbatim memorized length:',
            max([len(tokenizer(v).input_ids)
                 for k, v in list(sequence_to_memorized.values())[0].items()])
            if sequence_to_memorized else len(sequence_to_memorized))
      eval_results[step] = sequence_to_memorized
      del sequence_to_memorized
      gc.collect()
      if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Break conditions (copying same heuristics as distributed version)
    if 'single_shot_step' in config and step == config['single_shot_step'] + 1:
      break
    if not(config['inject_data'] is None) and step == round(config.get('inject_every_n', 1) * config.get('total_number_inject', 0) /
                     config['training_batch_size'] + config.get('inject_every_n', 1) / 2 /
                     config['training_batch_size']):
      break
    # Honor a max_steps argument to allow quick runs / tests
    if max_steps is not None and step >= int(max_steps):
      break

  metrics_logger['verbatim_memorization_length'].append(eval_results)

  # Save model and metrics
  model.save_pretrained(f'{log_path_base}.pt')
  torch.save(metrics_logger, f'{log_path_base}_metrics.pt')
  return model, metrics_logger


if __name__ == '__main__':
  # Minimal CLI that builds the small config dict expected by train_simple_model.
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--max_steps', type=int, help='Maximum number of training steps to run')
  parser.add_argument('--inject_sequence_ids', nargs='+', default=[], help='Keys of injection groups to run')
  parser.add_argument('--model', type=str, default='pythia-14m')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--checkpoint', type=int, default=80000)
  parser.add_argument('--window_size', type=int, default=256)
  parser.add_argument('--lr', type=float, default=2.79e-4)
  parser.add_argument('--pile_data_path',  nargs='+', default=['data/indicies.npy'])
  parser.add_argument('--injection_data_path', type=str, default='')
  parser.add_argument('--pretrained_optimizer_path', type=str, default='')
  parser.add_argument('--val_freq', type=int, default=100)
  parser.add_argument('--train_batch_size', type=int, default=128)
  parser.add_argument('--no_eval', action='store_true', help='Disable evaluation during training')
  parser.add_argument('--no_download', action='store_true', help='Skip downloading from Hugging Face Hub if not found locally')
  args = parser.parse_args()

  model_id = args.model
  ckpt_name = f'step{args.checkpoint}'
  task_name = f'ft_{model_id}_pile-80k_w{args.window_size}_lr{args.lr}_inject'

  data_dir = './data'
  model_dir = './models'

  # Prepare pile data (will raise if files are missing; provide --pile_data_path to override)
  logger.info("Loading pile data...")
  if not args.pile_data_path:
    args.pile_data_path = [
        os.path.join(data_dir, f'indicies.npy')
        for d in [80, 81]
    ]
  pile_2k_step = np.concatenate([np.load(p, 'r') for p in args.pile_data_path], axis=0)
  logger.info(f'Loaded pile data shape: {pile_2k_step.shape}')

  # Load injection metadata if provided
  group_to_inject_data = {}
  if args.injection_data_path:
    injection_path = os.path.join(data_dir, args.injection_data_path)
    if os.path.exists(injection_path):
      group_to_inject_data = json.load(open(injection_path))['pii_sequences']
    else:
      logger.info(f'Warning: injection data file not found: {injection_path}. No injections will be used.')

  # Dynamically calculate the 95/5 split
  total_samples = pile_2k_step.shape[0]
  val_size = int(total_samples * 0.05)  # 5% for validation
  split_point = total_samples - val_size

  train_range = [0, split_point]
  val_range = [split_point, total_samples] # Use the last 5%

  logger.info(f"Calculated 95/5 split for {total_samples} samples:")
  logger.info(f"  Training range:   {train_range} (Size: {train_range[1] - train_range[0]})")
  logger.info(f"  Validation range: {val_range} (Size: {val_range[1] - val_range[0]})")

  if val_size == 0:
      logger.warning("Validation set size is 0! Check data or split percentage.")

  world_size = max(1, torch.cuda.device_count())

  # Training defaults (adjust as needed)
  total_num_occur = 40
  inject_every_n = 10_000
  window_size = args.window_size
  init_lr = args.lr

  base_model_path = os.path.join(model_dir, f"{model_id}-{ckpt_name}")

  if not os.path.isdir(base_model_path):
    logger.warning(f"Directory '{base_model_path}' does not exist. Please download the model first.")

    if not args.no_download:
      # Attempt to download from Hugging Face Hub
      try:
        from transformers import AutoModelForCausalLM
        logger.info(f"Attempting to load model '{model_id}' from Hugging Face Hub...")
        model, tokenizer = load_model_and_tokenizer(model_id, revision=ckpt_name, cache_dir=model_dir, device='cpu')
        logger.info(f"Successfully loaded model '{model_id}' from Hugging Face Hub.")
        del model, tokenizer
      except Exception as e:
        logger.error(f"Failed to load model '{model_id}' from Hugging Face Hub: {e}")
        sys.exit(1)
    else:
      logger.info("Skipping download as per --no_download flag.")
      sys.exit(1)

  else:   
    logger.info(f"Using model directory: {base_model_path}")

  os.makedirs(os.path.join(model_dir, task_name), exist_ok=True)

  # Actual batch size is batch_size * world_size (naming kept for legacy reasons)
  eval_batch_size = 128

  # If no injection groups passed, run one default training without injection
  if not args.inject_sequence_ids:
    config = {
        'inject_every_n': inject_every_n,
        'total_number_inject': total_num_occur,
        'inject_data': None,
        'training_batch_size': args.train_batch_size,
        'eval_batch_size': eval_batch_size,
        'training_sample_range': train_range,
        'eval_sample_range': val_range,
        'window_size': window_size,
        'base_model': model_id,
        'base_model_path': base_model_path,
        'revision': ckpt_name,
        'init_lr': init_lr,
        'log_dir': os.path.join(model_dir, task_name, f'no_inject_bs{int(args.train_batch_size*world_size)}'),
        'model_dir': model_dir,
        'data': pile_2k_step,
        'run_eval': not args.no_eval,
        'pretrained_optimizer_path': args.pretrained_optimizer_path,
    }
    logger.info('Running training without injection')
    train_simple_model(config, max_steps=args.max_steps, val_freq=args.val_freq, seed=args.seed)
  else:
    # Run once per requested injection group (expects matching keys in the loaded JSON)
    for group in args.inject_sequence_ids:
      if group not in group_to_inject_data:
        logger.info(f'Warning: group {group} not found in injection metadata; skipping')
        continue
      inject_data = {int(k): v for k, v in group_to_inject_data[group].items()}
      assert all([k < inject_every_n for k in inject_data])

      config = {
          'inject_every_n': inject_every_n,
          'total_number_inject': total_num_occur,
          'inject_data': inject_data,
          'training_batch_size': args.train_batch_size,
          'eval_batch_size': eval_batch_size,
          'training_sample_range': train_range,
          'eval_sample_range': val_range,
          'window_size': window_size,
          'base_model': model_id,
          'base_model_path': base_model_path,
          'revision': ckpt_name,
          'init_lr': init_lr,
          'log_dir': os.path.join(model_dir, task_name, f'{group}_bs{int(args.train_batch_size*world_size)}'),
          'model_dir': model_dir,
          'data': pile_2k_step,
          'run_eval': not args.no_eval,
          'pretrained_optimizer_path': args.pretrained_optimizer_path,
      }
      logger.info(f'Running training for group={group}')
      train_simple_model(config, max_steps=args.max_steps, val_freq=args.val_freq, seed=args.seed)