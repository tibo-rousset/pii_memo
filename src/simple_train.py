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

logging.basicConfig(level=logging.INFO)
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


def compute_metrics(logits, labels):
  with torch.no_grad():
    pred = torch.argmax(logits[:, :-1], dim=-1)
    labels = labels[:, 1:]
    token_acc = torch.masked_select((pred == labels).type(torch.float32),
                                    labels != -100)
    return {
        'token_accuracy':
            token_acc.mean().float(),
        'last_token_accuracy':
            torch.reshape(token_acc, [labels.shape[0], -1])[:, -1].mean().float()
    }


def load_model_and_tokenizer(model_name, revision, cache_dir, device, tokenizer_only=False):
  # This is a simplified copy of the loader used in the distributed script.
  model_id = model_name
  if 'pythia' in model_id or 'neo' in model_id:
    model_id = 'EleutherAI/' + model_id
  elif 'opt' in model_id:
    model_id = 'facebook/' + model_id
  logger.info('Load checkpoint: %s %s', model_id, revision)
  logger.info('Cache directory: %s', cache_dir)
  tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
  tokenizer.pad_token = '<|padding|>'
  tokenizer.padding_side = 'left'
  if tokenizer_only:
    return tokenizer
  # Only implement the path we need commonly (pythia / neo-x families).
  if 'pythia' in model_id:
    model = GPTNeoXForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        revision=revision).to(device)
  else:
    # Fallback: try the generic AutoModelForCausalLM route
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 low_cpu_mem_usage=True,
                                                 cache_dir=cache_dir).to(device)
  return model, tokenizer


def train_simple_model(config, max_steps=None, val_freq=100, seed=42):
  """Single-process simplified training loop mirroring the distributed logic.

  Expected keys in config (kept similar to distributed script):
    - base_model, model_dir, data, training_sample_range, eval_sample_range
    - inject_data, inject_every_n, window_size
    - training_batch_size, eval_batch_size, init_lr, log_dir
    - run_eval (bool), single_shot_step (optional), total_number_inject
  """
  set_seed(seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  log_path_base = config['log_dir']

  # Tokenizer and datasets
  tokenizer = load_model_and_tokenizer(config['base_model'], config['revision'], config['model_dir'], device, tokenizer_only=True)
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

  logger.info('Dataset loaded:', len(train_dataloader), len(val_dataloader))

  # Model, optimizer & scheduler
  model, _ = load_model_and_tokenizer(config['base_model'], config['revision'], config['model_dir'], device)
  logger.info('#layers=%d' % model.config.num_hidden_layers)
  logger.info('Device:', device)

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
  for step in tqdm(range(total_steps), desc='Training', unit='step'):
    try:
      input_batch = next(train_iter)
    except StopIteration:
      break

    # Eval on validation set periodically
    if (step + 1) % val_freq == 0 and config.get('run_eval', False):
      model.eval()
      val_metrics = collections.defaultdict(list)
      with torch.no_grad():
        for val_input_batch in val_dataloader:
          for k in val_input_batch:
            val_input_batch[k] = val_input_batch[k].to(device)
          loss, logits, labels = lm_train_step(model, val_input_batch)
          metrics = compute_metrics(logits, labels)
          for key in metrics:
            val_metrics[key].append(metrics[key].detach().cpu())
          val_metrics['training_loss'].append(loss.float().detach().cpu().mean())
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

      logger.info(epoch, step, val_metrics['training_loss'], val_metrics['token_accuracy'], last_lr_val)

      #logger.info('Epoch %d Step %d: Loss %.4f Accuracy %.4f LR %.2E' %
      #          (epoch, step, val_metrics['training_loss'], val_metrics['token_accuracy'], last_lr_val))
      metrics_logger['loss'].append(val_metrics['training_loss'])
      metrics_logger['accuracy'].append(val_metrics['token_accuracy'])

    # Training step
    model.train()
    for k in feature_keys:
      input_batch[k] = input_batch[k].to(device)
    loss, logits, labels = lm_train_step(model, input_batch)
    gradient_accumulation_steps = 1
    loss = loss / gradient_accumulation_steps
    loss.backward()
    if (step + 1) % gradient_accumulation_steps == 0:
      optimizer.step()
      lr_scheduler.step()
      optimizer.zero_grad()
    del loss, logits, labels
    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

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
      model.train()

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
  parser.add_argument('--max_steps', type=int, default=100, help='Maximum number of training steps to run')
  parser.add_argument('--inject_sequence_ids', nargs='+', default=[], help='Keys of injection groups to run')
  parser.add_argument('--model', type=str, default='pythia-14m')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--checkpoint', type=int, default=80000)
  parser.add_argument('--window_size', type=int, default=256)
  parser.add_argument('--lr', type=float, default=2.79e-4)
  parser.add_argument('--pile_data_path',  nargs='+', default=['/Users/tibor/Desktop/pii_memo/data/indicies.npy'])
  parser.add_argument('--injection_data_path', type=str, default='')
  parser.add_argument('--pretrained_optimizer_path', type=str, default='')
  parser.add_argument('--val_freq', type=int, default=100)
  args = parser.parse_args()

  model_id = args.model
  ckpt_name = f'step{args.checkpoint}'
  task_name = f'ft_{model_id}_pile-80k_w{args.window_size}_lr{args.lr}_inject'

  data_dir = './data'
  model_dir = './models'

  # Prepare pile data (will raise if files are missing; provide --pile_data_path to override)
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

  os.makedirs(os.path.join(model_dir, task_name), exist_ok=True)

  # Actual batch size is batch_size * world_size (naming kept for legacy reasons)
  training_batch_size = 128
  eval_batch_size = 128

  # If no injection groups passed, run one default training without injection
  if not args.inject_sequence_ids:
    config = {
        'inject_every_n': inject_every_n,
        'total_number_inject': total_num_occur,
        'inject_data': None,
        'training_batch_size': training_batch_size,
        'eval_batch_size': eval_batch_size,
        'training_sample_range': [0, 2000 * 1024],
        'eval_sample_range': [2000 * 1024, 2048 * 1024],
        'window_size': window_size,
        'base_model': model_id,
        'revision': ckpt_name,
        'init_lr': init_lr,
        'log_dir': os.path.join(model_dir, task_name, f'no_inject_bs{int(training_batch_size*world_size)}'),
        'model_dir': model_dir,
        'data': pile_2k_step,
        'run_eval': True,
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
          'training_batch_size': training_batch_size,
          'eval_batch_size': eval_batch_size,
          'training_sample_range': [0, 2000 * 1024],
          'eval_sample_range': [2000 * 1024, 2048 * 1024],
          'window_size': window_size,
          'base_model': ckpt_name,
          'init_lr': init_lr,
          'log_dir': os.path.join(model_dir, task_name, f'{group}_bs{int(training_batch_size*world_size)}'),
          'model_dir': model_dir,
          'data': pile_2k_step,
          'run_eval': True,
          'pretrained_optimizer_path': args.pretrained_optimizer_path,
      }
      logger.info(f'Running training for group={group}')
      train_simple_model(config, max_steps=args.max_steps, val_freq=args.val_freq, seed=args.seed)