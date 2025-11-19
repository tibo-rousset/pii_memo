"""
Single script to prepare injection data from injection_data_config.json

This script:
1. Reads your config with templates, PII types, and desired frequencies
2. Loads the PANORAMA dataset and fills in placeholders
3. Creates the injection mapping with proper frequency distribution
4. Outputs training-ready JSON and metadata for analysis

Usage:
    python scripts/prepare_injection_from_config.py --config injection_data_config.json --output data/injection_training.json
"""

import json
import argparse
import random
import os
from datasets import load_dataset


def load_and_fill_templates(config):
    """Load dataset and fill templates with real PII data."""
    dataset_config = config['dataset_config']
    training_config = config['training_config']
    
    # Determine mode: frequency_comparison or single_frequency
    mode = training_config.get('experiment_mode', 'frequency_comparison')
    
    if mode == 'frequency_comparison':
        # Multiple samples per type at different frequencies for analysis
        frequencies = training_config.get('frequency', [])
        if not frequencies:
            raise ValueError("frequency_comparison mode requires 'frequency' list")
        num_samples_per_type = len(frequencies)
        print(f"✓ Mode: Frequency Comparison")
        print(f"✓ Using {num_samples_per_type} samples per type with frequencies: {frequencies}")
    elif mode == 'single_frequency':
        # Many samples at one frequency for memorization
        frequency = training_config.get('frequency')
        if frequency is None:
            raise ValueError("single_frequency mode requires 'frequency' value")
        num_samples_per_type = training_config.get('num_samples_per_type', 50)
        frequencies = [frequency] * num_samples_per_type
        print(f"✓ Mode: Single Frequency Memorization")
        print(f"✓ Using {num_samples_per_type} samples per type, all at frequency: {frequency}")
    else:
        raise ValueError(f"Unknown experiment_mode: {mode}")
    
    # Load dataset
    try:
        ds = load_dataset(
            dataset_config['dataset_name'],
            cache_dir=dataset_config.get('cache_dir', None)
        )
        print(f"✓ Dataset loaded: {dataset_config['dataset_name']}")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print("  Please run 'huggingface-cli login' to authenticate.")
        return None
    
    # Filter out N/A values
    valid_samples = []
    for sample in ds['train']:
        if (sample["Driver's License"] != "N/A" and 
            sample['Passport Number'] != "N/A" and 
            sample['Email Address'] != "N/A" and 
            sample['National ID'] != "N/A"):
            valid_samples.append(sample)
    
    print(f"✓ Found {len(valid_samples)} valid samples (no N/A values)")
    
    # Fill templates
    filled_sequences = []
    sample_idx = 0
    
    for seq_config in config['sequences']:
        template = seq_config['template']
        pii_type = seq_config['pii_type']
        
        # Create num_samples_per_type samples, each with a different frequency
        for i in range(num_samples_per_type):
            if sample_idx >= len(valid_samples):
                sample_idx = 0  # Cycle if needed
            
            sample = valid_samples[sample_idx]
            sample_idx += 1
            
            # Assign frequency based on sample index (consistent across all PII types)
            frequency = frequencies[i]
            
            # Fill in placeholders
            filled_text = template
            filled_text = filled_text.replace('<FirstName>', sample['First Name'])
            filled_text = filled_text.replace('<LastName>', sample['Last Name'])
            filled_text = filled_text.replace('<DriverLicense>', sample["Driver's License"])
            filled_text = filled_text.replace('<Passport>', sample['Passport Number'])
            filled_text = filled_text.replace('<EmailAddress>', sample['Email Address'])
            filled_text = filled_text.replace('<IDNumber>', sample['National ID'])
            
            filled_sequences.append({
                'text': filled_text,
                'pii_type': pii_type,
                'frequency': frequency,
                'template': template,
                'sample_index': i  # Track which sample this is (0-indexed)
            })
    
    print(f"✓ Created {len(filled_sequences)} filled sequences")
    return filled_sequences


def create_injection_mapping(filled_sequences, training_config):
    """Create the injection mapping for NumpyArrayDataset."""
    inject_every_n = training_config['inject_every_n']
    seed = training_config.get('seed', 42)
    
    # Create (key, sequence_text) pairs based on frequency
    key_sequence_pairs = []
    for seq_info in filled_sequences:
        for _ in range(seq_info['frequency']):
            key_sequence_pairs.append(seq_info['text'])
    
    # Check total keys
    total_keys = len(key_sequence_pairs)
    if total_keys > inject_every_n:
        raise ValueError(
            f"Too many injection keys! Need {total_keys}, "
            f"but only {inject_every_n} available. "
            f"Reduce frequencies or number of samples."
        )
    
    print(f"✓ Total injection keys needed: {total_keys} (max: {inject_every_n})")
    
    # Shuffle to randomize which sequence is mapped to which offset
    random.seed(seed)
    random.shuffle(key_sequence_pairs)

    # Assign unique random offsets in [0, inject_every_n)
    # This spreads injections within each inject_every_n window
    offsets = random.sample(range(inject_every_n), total_keys)

    # Create mapping from randomized offsets to sequences
    injection_mapping = {str(offset): seq for offset, seq in zip(offsets, key_sequence_pairs)}
    
    return injection_mapping


def create_metadata(filled_sequences, injection_mapping, training_config):
    """Create metadata for analysis."""
    metadata = {
        "training_config": training_config,
        "total_unique_sequences": len(filled_sequences),
        "total_injection_keys": len(injection_mapping),
        "sequences": []
    }
    
    # For each unique sequence, track where it appears
    for seq_info in filled_sequences:
        keys = [k for k, v in injection_mapping.items() if v == seq_info['text']]
        
        metadata["sequences"].append({
            "text": seq_info['text'],
            "pii_type": seq_info['pii_type'],
            "template": seq_info['template'],
            "frequency": seq_info['frequency'],
            "sample_index": seq_info['sample_index'],
            "key_positions": sorted([int(k) for k in keys]),
            "total_occurrences": len(keys)
        })
    
    # Sort by PII type, then sample index for easy cross-type frequency comparison
    metadata["sequences"].sort(key=lambda x: (x['pii_type'], x['sample_index']))
    
    return metadata


def print_summary(metadata, training_config):
    """Print a summary of the injection configuration."""
    print("\n" + "="*70)
    print("INJECTION CONFIGURATION SUMMARY")
    print("="*70)
    
    print(f"\nMode: {training_config['mode']}")
    print(f"Inject every N: {training_config['inject_every_n']}")
    if 'frequencies' in training_config:
        print(f"Frequencies: {training_config['frequencies']}")
    elif 'frequency' in training_config:
        print(f"Frequency: {training_config['frequency']}")
        print(f"Number of samples per type: {training_config['num_samples_per_type']}")
    print(f"Total unique sequences: {metadata['total_unique_sequences']}")
    print(f"Total injection keys: {metadata['total_injection_keys']}")
    
    # Group by PII type
    print(f"\nSequences by PII type:")
    type_groups = {}
    for seq in metadata['sequences']:
        pii_type = seq['pii_type']
        if pii_type not in type_groups:
            type_groups[pii_type] = []
        type_groups[pii_type].append(seq)
    
    for pii_type in sorted(type_groups.keys()):
        seqs = type_groups[pii_type]
        print(f"\n  {pii_type}: {len(seqs)} sequences")
        print(f"    Sample | Frequency")
        print(f"    " + "-" * 30)
        for seq in seqs:
            print(f"    {seq['sample_index']:6} | {seq['frequency']:9}")
    
    # Frequency distribution (cross-check)
    print(f"\nFrequency distribution (cross-check):")
    freq_groups = {}
    for seq in metadata['sequences']:
        freq = seq['frequency']
        if freq not in freq_groups:
            freq_groups[freq] = []
        freq_groups[freq].append(seq)
    
    for freq in sorted(freq_groups.keys()):
        seqs = freq_groups[freq]
        type_counts = {}
        for seq in seqs:
            pii_type = seq['pii_type']
            type_counts[pii_type] = type_counts.get(pii_type, 0) + 1
        
        print(f"  Frequency {freq}: {len(seqs)} sequences ({', '.join([f'{t}:{c}' for t, c in sorted(type_counts.items())])})")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare injection data from config file')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to injection_data_config.json')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for training JSON')
    parser.add_argument('--group-name', type=str, default='pii_sequences',
                       help='Name for the injection group (default: pii_sequences)')
    
    args = parser.parse_args()
    
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Step 1: Load dataset and fill templates
    print("\n[1/4] Loading dataset and filling templates...")
    filled_sequences = load_and_fill_templates(config)
    if filled_sequences is None:
        return
    
    # Step 2: Create injection mapping
    print("\n[2/4] Creating injection mapping...")
    injection_mapping = create_injection_mapping(
        filled_sequences,
        config['training_config']
    )
    
    # Step 3: Create metadata
    print("\n[3/4] Creating metadata...")
    metadata = create_metadata(
        filled_sequences,
        injection_mapping,
        config['training_config']
    )
    
    # Step 4: Save outputs
    print("\n[4/4] Saving outputs...")
    
    # Training config format
    group_name = args.group_name
    training_output = {
        group_name: injection_mapping,
        f"{group_name}_transform": config['training_config']['mode']
    }

    output_dir = os.path.dirname(args.output)

    if output_dir: # Check if the path is not empty
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output, 'w') as f:
        json.dump(training_output, f, indent=2)
    print(f"  ✓ Training config: {args.output}")
    print(f"  ✓ Group name: {group_name}")
    
    # Metadata
    metadata_path = args.output.replace('.json', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata: {metadata_path}")
    
    # Print summary
    print_summary(metadata, config['training_config'])
    
    print("\n" + "="*70)
    print("✓ DONE! Ready to train with:")
    print(f"  python scripts/train_with_injection.py \\")
    print(f"    --inject_sequence_ids {group_name} \\")
    print(f"    --injection_data_path {args.output.replace('data/', '')}")
    print("="*70)


if __name__ == "__main__":
    main()

