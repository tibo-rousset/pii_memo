# Minimal CLI that builds the small config dict expected by train_simple_model.
import argparse
import logging
import sys
import json
import numpy as np
import os
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)  # send logs to stdout
    ]
)

MEM_LIB_DIR = f'src'
sys.path.append(MEM_LIB_DIR)

from train_std import train_simple_model, load_model_and_tokenizer, set_seed

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='train_config.json')
    parser.add_argument('--max_steps', type=int, help='Maximum number of training steps to run')
    parser.add_argument('--inject_sequence_ids', nargs='+', default=[], help='Keys of injection groups to run')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pile_data_path',  nargs='+', default=['data/indicies.npy'])
    parser.add_argument('--injection_data_path', type=str, default='')
    parser.add_argument('--pretrained_optimizer_path', type=str, default='')
    parser.add_argument('--val_freq', type=int, default=100)
    parser.add_argument('--no_eval', action='store_true', help='Disable evaluation during training')
    parser.add_argument('--no_download', action='store_true', help='Skip downloading from Hugging Face Hub if not found locally')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging is enabled.")

    config_defaults = json.load(open(args.config_file, 'r'))

    config_defaults['no_eval'] = args.no_eval
    model_id = config_defaults.get('model', 'pythia-14m')
    ckpt_name = config_defaults.get('revision', 'step80000')

    task_name = f'ft_{model_id}_pile-80k_w{config_defaults.get("window_size", 1024)}_lr{config_defaults.get("init_lr", 5e-3)}_inject'

    data_dir = config_defaults.get('data_dir', './data')
    model_dir = config_defaults.get('model_dir', './models')

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
    config_defaults['data'] = pile_2k_step
    logger.info(f'Loaded pile data shape: {pile_2k_step.shape}')

    # Load injection metadata if provided
    group_to_inject_data = {}
    if args.injection_data_path:
        injection_path = os.path.join(data_dir, args.injection_data_path)
        injection_metadata_path = injection_path.replace('.json', '_metadata.json')

        if os.path.exists(injection_path):
            group_to_inject_data = json.load(open(injection_path))
            logger.info(f'Loaded injection data for groups: {list(group_to_inject_data.keys())}')

            try:
                injection_metadata = json.load(open(injection_metadata_path))
                logger.info(f'Loaded injection metadata from {injection_metadata_path}')
            except Exception as e:
                logger.warning(f'Could not load injection metadata: {e}')
        
        else:
            logger.info(f'Warning: injection data file not found: {injection_path}. No injections will be used.')
            args.inject_sequence_ids = None

    # Dynamically calculate the 95/5 split
    total_samples = pile_2k_step.shape[0]
    val_size = int(total_samples * config_defaults['val_size'])  # 5% for validation
    split_point = total_samples - val_size

    train_range = [0, split_point]
    val_range = [split_point, total_samples] # Use the last 5%

    config_defaults['training_sample_range'] = train_range
    config_defaults['eval_sample_range'] = val_range

    logger.info(f"Calculated 95/5 split for {total_samples} samples:")
    logger.info(f"  Training range:   {train_range} (Size: {train_range[1] - train_range[0]})")
    logger.info(f"  Validation range: {val_range} (Size: {val_range[1] - val_range[0]})")

    if val_size == 0:
        logger.warning("Validation set size is 0! Check data or split percentage.")

    world_size = max(1, torch.cuda.device_count())

    base_model_path = os.path.join(model_dir, f"{model_id}-{ckpt_name}")
    config_defaults['base_model_path'] = base_model_path

    if not os.path.isdir(base_model_path):
        logger.warning(f"Directory '{base_model_path}' does not exist. Please download the model first.")

        if not args.no_download:
            # Attempt to download from Hugging Face Hub
            try:
                from transformers import AutoModelForCausalLM
                logger.info(f"Attempting to load model '{model_id}' from Hugging Face Hub...")

                if 'pythia' in model_id:
                    from huggingface_hub import snapshot_download

                    model_path = snapshot_download(
                        repo_id="EleutherAI/" + model_id,
                        local_dir=base_model_path,
                        local_dir_use_symlinks=False
                    )

                    logger.info(f"Model downloaded to: {model_path}")

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
        config_defaults['inject_data'] = None
        logger.info('Running training without injection')

        config_defaults['log_dir'] = os.path.join(model_dir, task_name, f'no_inject_bs{int(config_defaults["training_batch_size"]*world_size)}')
        train_simple_model(config_defaults, max_steps=args.max_steps, val_freq=args.val_freq, seed=args.seed)

    else:
    # Run once per requested injection group (expects matching keys in the loaded JSON)
        inject_every_n = injection_metadata['training_config'].get('inject_every_n')
        config_defaults['inject_every_n'] = inject_every_n
        logger.info(f'Training with injection every {inject_every_n} steps')

        prepend = injection_metadata['training_config'].get('mode', 'prepend') == 'prepend'
        logger.info(f'Injection mode: {"prepend" if prepend else "replace"}')

        for group in args.inject_sequence_ids:
            if group not in group_to_inject_data:
                logger.warning(f'Group {group} not found in injection data; skipping')
                continue

            inject_data = {int(k): v for k, v in group_to_inject_data[group].items()}
            assert all([k < inject_every_n for k in inject_data])

            config_defaults['inject_data'] = inject_data
            config_defaults['log_dir'] = os.path.join(model_dir, task_name, f'{group}_bs{int(config_defaults["training_batch_size"]*world_size)}')

            logger.info(f'Running training for group={group}')

            keys_to_exclude = ['data', 'inject_data']  # example keys to exclude

            filtered_config = {k: v for k, v in config_defaults.items() if k not in keys_to_exclude}

            # Convert any NumPy arrays to lists if needed
            for k, v in filtered_config.items():
                if isinstance(v, np.ndarray):
                    filtered_config[k] = v.tolist()

            logger.debug(f'Filtered Config (excluded keys): {json.dumps(filtered_config, indent=4)}')
            train_simple_model(config_defaults, max_steps=args.max_steps, val_freq=args.val_freq, seed=args.seed, prepend=prepend)
