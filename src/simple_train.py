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


def compute_metrics(logits, labels, tokenizer):
  """
  Computes both token-level accuracy and PII presence rate.
  """
  metrics = {}
  
  with torch.no_grad():
    
    #--- 1. Token Accuracy Calculation ---
    
    # Get greedy predictions (token IDs)
    # Shape: [batch_size, seq_len - 1]
    pred_ids = torch.argmax(logits[:, :-1], dim=-1)
    
    # Get aligned labels
    # Shape: [batch_size, seq_len - 1]
    label_ids_for_acc = labels[:, 1:]

    # Create masks
    accuracy_mask = (pred_ids == label_ids_for_acc)
    valid_token_mask = (label_ids_for_acc != -100)
    
    # Filter for valid tokens only
    token_acc_tensor = torch.masked_select(
        accuracy_mask.type(torch.float32),
        valid_token_mask
    )
    
    # Calculate mean token accuracy
    metrics['token_accuracy'] = token_acc_tensor.mean().float()

    #--- 2. PII Presence Rate Calculation ---
    
    # We already have pred_ids, so we just need to clean up the labels
    # for decoding.
    label_ids_for_decode = labels[:, 1:].clone() # Clone to avoid modifying
    
    # Replace -100 with pad token for safe decoding
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

      # Core check: Is the PII (label string)
      # present *anywhere* in the model's output string?
      if clean_label_pii in clean_pred_output:
        pii_present.append(1.0) # PII was found
      else:
        pii_present.append(0.0) # PII was not found

    # Calculate Rate
    if len(pii_present) == 0:
      # Avoid division by zero if batch was all padding
      metrics['pii_presence_rate'] = 0.0
    else:
      metrics['pii_presence_rate'] = sum(pii_present) / len(pii_present)

  return metrics


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
        torch_dtype=torch.bfloat16,
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

  train_dataloader = torch.utils.data.DataLoader(
      train_dataset, batch_size=config['training_batch_size'], shuffle=True)
  val_dataloader = torch.utils.data.DataLoader(
      val_dataset, batch_size=config['eval_batch_size'], shuffle=False)

  logger.info(f"Dataset loaded: train={len(train_dataloader)}, val={len(val_dataloader)}")

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

    # Update tqdm bar description or postfix with current training loss
    pbar.set_postfix(train_loss=f'{loss:.4f}')
    del loss, logits, labels

    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

      # Eval on validation set periodically
    if (step + 1) % val_freq == 0 and config.get('run_eval', False):
      model.eval()
      val_metrics = collections.defaultdict(list)
      with torch.no_grad():
        for val_input_batch in val_dataloader:
          for k in val_input_batch:
            val_input_batch[k] = val_input_batch[k].to(device)
          loss, logits, labels = lm_train_step(model, val_input_batch)
          metrics = compute_metrics(logits, labels, tokenizer)
          for key in metrics:
            val_metrics[key].append(metrics[key].detach().cpu())
          val_metrics['validation_loss'].append(loss.float().detach().cpu().mean())
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
          "Epoch: %d, Step: %d, Validation Loss: %.4f, Token Accuracy: %.4f, LR: %.6f",
          epoch,
          step,
          val_metrics['validation_loss'] if 'validation_loss' in val_metrics else float('nan'),
          val_metrics['token_accuracy'] if 'token_accuracy' in val_metrics else float('nan'),
          val_metrics['pii_presence_rate'] if 'pii_presence_rate' in val_metrics else float('nan'),
          last_lr_val
      )
      metrics_logger['val_loss'].append(val_metrics['validation_loss'])
      metrics_logger['accuracy'].append(val_metrics['token_accuracy'])

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
    if step == round(config.get('inject_every_n', 1) * config.get('total_number_inject', 0) /
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
      group_to_inject_data = json.load(open(injection_path))
    else:
      logger.info(f'Warning: injection data file not found: {injection_path}. No injections will be used.')

  world_size = max(1, torch.cuda.device_count())

  # Training defaults (adjust as needed)
  total_num_occur = 40
  inject_every_n = 10_000
  window_size = args.window_size
  init_lr = args.lr

  base_model_path = os.path.join(model_dir, f"{model_id}-{ckpt_name}")

  if not os.path.isdir(base_model_path):
    logger.warning(f"Directory '{base_model_path}' does not exist. Please download the model first.")
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
        'training_sample_range': [0, 2000 * 1024],
        'eval_sample_range': [2000 * 1024, 2048 * 1024],
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
          'training_sample_range': [0, 2000 * 1024],
          'eval_sample_range': [2000 * 1024, 2048 * 1024],
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