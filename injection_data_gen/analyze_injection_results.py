"""
Helper script to analyze injection results from metadata.

Usage:
    python scripts/analyze_injection_results.py --metadata data/injection_training_metadata.json
"""

import json
import argparse
from collections import defaultdict


def load_metadata(path):
    """Load metadata file."""
    with open(path, 'r') as f:
        return json.load(f)


def print_summary(metadata):
    """Print overall summary."""
    print("="*70)
    print("INJECTION SUMMARY")
    print("="*70)
    print(f"Total unique sequences: {metadata['total_unique_sequences']}")
    print(f"Total injection keys: {metadata['total_injection_keys']}")
    print(f"Mode: {metadata['training_config']['mode']}")
    print(f"Inject every N: {metadata['training_config']['inject_every_n']}")


def analyze_by_frequency(metadata):
    """Analyze sequences by frequency."""
    print("\n" + "="*70)
    print("ANALYSIS BY FREQUENCY")
    print("="*70)
    
    freq_groups = defaultdict(list)
    for seq in metadata['sequences']:
        freq_groups[seq['frequency']].append(seq)
    
    for freq in sorted(freq_groups.keys()):
        seqs = freq_groups[freq]
        print(f"\nFrequency {freq}: {len(seqs)} sequences")
        
        # Count by type
        type_counts = defaultdict(int)
        for seq in seqs:
            type_counts[seq['pii_type']] += 1
        
        for pii_type, count in sorted(type_counts.items()):
            print(f"  {pii_type}: {count}")
        
        # Show sample sequences
        print(f"  Sample sequences:")
        for seq in seqs[:2]:
            print(f"    - {seq['text'][:60]}...")


def analyze_by_type(metadata):
    """Analyze sequences by PII type."""
    print("\n" + "="*70)
    print("ANALYSIS BY PII TYPE")
    print("="*70)
    
    type_groups = defaultdict(list)
    for seq in metadata['sequences']:
        type_groups[seq['pii_type']].append(seq)
    
    for pii_type in sorted(type_groups.keys()):
        seqs = type_groups[pii_type]
        print(f"\n{pii_type}: {len(seqs)} sequences")
        
        # Count by frequency
        freq_counts = defaultdict(int)
        for seq in seqs:
            freq_counts[seq['frequency']] += 1
        
        for freq, count in sorted(freq_counts.items()):
            print(f"  Frequency {freq}: {count} sequences")


def get_sequences_by_frequency(metadata, frequency):
    """Get all sequences at a specific frequency."""
    return [s for s in metadata['sequences'] if s['frequency'] == frequency]


def get_sequences_by_type(metadata, pii_type):
    """Get all sequences of a specific type."""
    return [s for s in metadata['sequences'] if s['pii_type'] == pii_type]


def get_sequence_info(metadata, text_fragment):
    """Find sequence info by text fragment."""
    results = []
    for seq in metadata['sequences']:
        if text_fragment.lower() in seq['text'].lower():
            results.append(seq)
    return results


def calculate_exposure_percentage(metadata):
    """Calculate what percentage of training data each frequency represents."""
    print("\n" + "="*70)
    print("EXPOSURE PERCENTAGE")
    print("="*70)
    
    # Training parameters
    inject_every_n = metadata['training_config']['inject_every_n']
    total_num_occur = 40  # From train_with_injection.py
    training_batch_size = 128
    
    stop_step = round(inject_every_n * total_num_occur / training_batch_size + 
                      inject_every_n / 2 / training_batch_size)
    total_samples = stop_step * training_batch_size
    
    print(f"Training will see ~{total_samples:,} samples total")
    print(f"Training stops at step ~{stop_step:,}\n")
    
    # Get unique frequencies
    frequencies = sorted(set(s['frequency'] for s in metadata['sequences']))
    
    print(f"{'Frequency':<12} {'Times Seen':<12} {'Percentage':<12} {'Ratio'}")
    print("-" * 70)
    
    for freq in frequencies:
        percentage = (freq / total_samples) * 100
        ratio = int(total_samples / freq)
        print(f"{freq:<12} {freq:<12} {percentage:>10.4f}% {f'1 in {ratio:,}'}")


def export_for_memorization_test(metadata, output_path):
    """Export sequences in format for memorization testing."""
    output = {
        'sequences_by_frequency': {},
        'sequences_by_type': {},
        'all_sequences': []
    }
    
    # Group by frequency
    for seq in metadata['sequences']:
        freq = seq['frequency']
        if freq not in output['sequences_by_frequency']:
            output['sequences_by_frequency'][freq] = []
        output['sequences_by_frequency'][freq].append({
            'text': seq['text'],
            'type': seq['pii_type'],
            'key_positions': seq['key_positions']
        })
    
    # Group by type
    for seq in metadata['sequences']:
        pii_type = seq['pii_type']
        if pii_type not in output['sequences_by_type']:
            output['sequences_by_type'][pii_type] = []
        output['sequences_by_type'][pii_type].append({
            'text': seq['text'],
            'frequency': seq['frequency'],
            'key_positions': seq['key_positions']
        })
    
    # All sequences
    for seq in metadata['sequences']:
        output['all_sequences'].append({
            'text': seq['text'],
            'type': seq['pii_type'],
            'frequency': seq['frequency'],
            'key_positions': seq['key_positions']
        })
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Exported memorization test data to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze injection metadata')
    parser.add_argument('--metadata', type=str, required=True,
                       help='Path to metadata JSON file')
    parser.add_argument('--export', type=str, default=None,
                       help='Export sequences for memorization testing')
    parser.add_argument('--frequency', type=int, default=None,
                       help='Show only sequences at this frequency')
    parser.add_argument('--type', type=str, default=None,
                       help='Show only sequences of this type')
    parser.add_argument('--search', type=str, default=None,
                       help='Search for sequences containing this text')
    
    args = parser.parse_args()
    
    # Load metadata
    metadata = load_metadata(args.metadata)
    
    # Filtered views
    if args.frequency:
        seqs = get_sequences_by_frequency(metadata, args.frequency)
        print(f"\nSequences at frequency {args.frequency}: {len(seqs)}")
        for seq in seqs:
            print(f"  [{seq['pii_type']}] {seq['text']}")
        return
    
    if args.type:
        seqs = get_sequences_by_type(metadata, args.type)
        print(f"\nSequences of type '{args.type}': {len(seqs)}")
        for seq in seqs:
            print(f"  [freq={seq['frequency']}] {seq['text']}")
        return
    
    if args.search:
        results = get_sequence_info(metadata, args.search)
        print(f"\nFound {len(results)} sequences matching '{args.search}':")
        for seq in results:
            print(f"  [{seq['pii_type']}, freq={seq['frequency']}] {seq['text']}")
        return
    
    # Full analysis
    print_summary(metadata)
    analyze_by_frequency(metadata)
    analyze_by_type(metadata)
    calculate_exposure_percentage(metadata)
    
    if args.export:
        export_for_memorization_test(metadata, args.export)


if __name__ == "__main__":
    main()

