import json
import argparse
import random
import os
from datasets import load_dataset


def load_and_fill_templates(config, filepath=None):
    """Load dataset and fill templates with real PII data."""
    dataset_config = config['dataset_config']
    training_config = config['training_config']
    
    mode = training_config.get('experiment_mode', 'frequency_comparison')
    
    if mode == 'frequency_comparison':
        frequencies = training_config.get('frequency', [])
        if not frequencies:
            raise ValueError("frequency_comparison mode requires 'frequency' list")
        num_samples_per_type = len(frequencies) if training_config.get('num_samples_per_type') is None else training_config.get('num_samples_per_type')
        print(f"Frequency Comparison Mode")
        print(f"Using {num_samples_per_type} samples per type with frequencies: {frequencies}")
    elif mode == 'single_frequency':
        frequency = training_config.get('frequency')
        if frequency is None:
            raise ValueError("single_frequency mode requires 'frequency' value")
        num_samples_per_type = training_config.get('num_samples_per_type', 50)
        frequencies = [frequency] * num_samples_per_type
        print(f"Single Frequency Memorization Mode")
        print(f"Using {num_samples_per_type} samples per type, all at frequency: {frequency} per {training_config.get('inject_every_n')}")
    else:
        raise ValueError(f"Unknown experiment_mode: {mode}")
    
    try:
        if filepath is not None:
            print(f"Loading dataset from local file: {filepath}...")
            ds = load_dataset(
                'parquet',
                data_files=filepath,
                cache_dir=dataset_config.get('cache_dir', None),
            )
        else:
            print("Loading dataset from Hugging Face Hub...")
            ds = load_dataset(
                dataset_config['dataset_name'],
                cache_dir=dataset_config.get('cache_dir', None),

            )
        print(f"✓ Dataset loaded: {dataset_config['dataset_name']}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("   Please run 'huggingface-cli login' to authenticate.")
        return None
    
    # some N/A values were found present in the panorama plus dataset so we filter them out
    valid_samples = []
    for sample in ds['train']:
        if (sample["Driver's License"] != "N/A" and 
            sample['Passport Number'] != "N/A" and 
            sample['Email Address'] != "N/A" and 
            sample['National ID'] != "N/A"):
            valid_samples.append(sample)
    
    print(f"Found {len(valid_samples)} valid samples")
    
    filled_sequences = []
    sample_idx = 0
    
    for seq_config in config['sequences']:
        template = seq_config['template']
        pii_type = seq_config['pii_type']
        
        # Create num_samples_per_type samples, each with a different frequency
        for i in range(num_samples_per_type):
            
            sample = valid_samples[sample_idx]
            sample_idx += 1
            
            frequency = frequencies[i]

            pii_values = {
                'first_name': sample['First Name'],
                'last_name': sample['Last Name'],
                'driver_license': sample["Driver's License"],
                'passport': sample['Passport Number'],
                'email': sample['Email Address'],
                'id_number': sample['National ID']
            }
            
            # Fill in placeholders
            filled_text = template
            filled_text = filled_text.replace('<FirstName>', pii_values['first_name'])
            filled_text = filled_text.replace('<LastName>', pii_values['last_name'])
            filled_text = filled_text.replace('<DriverLicense>', pii_values['driver_license'])
            filled_text = filled_text.replace('<Passport>', pii_values['passport'])
            filled_text = filled_text.replace('<EmailAddress>', pii_values['email'])
            filled_text = filled_text.replace('<IDNumber>', pii_values['id_number'])
            
            filled_sequences.append({
                'text': filled_text,
                'pii_type': pii_type,
                'frequency': frequency,
                'template': template,
                'sample_index': i, 
                'pii_values': pii_values
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
    
   
    random.seed(seed)
    
    # Shuffle to ensure that the sequences are not ordered by pii type 
    # (i.e. all email sequences, then all passport sequences)
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
            "pii_values": seq_info['pii_values'], # Added specific PII values
            "key_positions": sorted([int(k) for k in keys]),
            "total_occurrences": len(keys)
        })
    
    # sort by pii type and sample index so the pii sequences are grouped together in the metadata json
    metadata["sequences"].sort(key=lambda x: (x['pii_type'], x['sample_index']))
    
    return metadata

def main():
    parser = argparse.ArgumentParser(
        description='Prepare injection data from config file')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to injection_data_config.json')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for training JSON')
    parser.add_argument('--group-name', type=str, default='pii_sequences',
                       help='Name for the injection group (default: pii_sequences)')
    parser.add_argument('--filepath', type=str, default=None, 
                       help='Path to local dataset file (optional)')
    
    args = parser.parse_args()
    
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    filled_sequences = load_and_fill_templates(config, args.filepath)
    if filled_sequences is None:
        return
    
    injection_mapping = create_injection_mapping(
        filled_sequences,
        config['training_config']
    )
    
    metadata = create_metadata(
        filled_sequences,
        injection_mapping,
        config['training_config']
    )
    
    group_name = args.group_name
    training_output = {
        group_name: injection_mapping,
        "transform": config['training_config'].get('experiment_mode', 'frequency_comparison')
    }

    output_dir = os.path.dirname(args.output)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output, 'w') as f:
        json.dump(training_output, f, indent=2)
    print(f"  ✓ Training config: {args.output}")
    
    metadata_path = args.output.replace('.json', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()