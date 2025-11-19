# Minimal CLI that builds the small config dict expected by train_simple_model.
import argparse
import logging
import sys
import json
import numpy as np
import os
import torch

from train_std import train_simple_model, load_model_and_tokenizer, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)  # send logs to stdout
    ]
)

MEM_LIB_DIR = f'pii_memo/src'
sys.path.append(MEM_LIB_DIR)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
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

    set_seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")

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
        logger.info(f'Loaded injection data for groups: {list(group_to_inject_data.keys())}')
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

                if 'pythia' in model_id:
                    model, tokenizer = load_model_and_tokenizer(f'EleutherAI/{model_id}', revision=ckpt_name, cache_dir=model_dir, device='cpu')
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
                logger.warning(f'Group {group} not found in injection data; skipping')
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