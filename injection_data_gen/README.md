# PII Injection Data Generation System

## Overview

This system generates synthetic PII (Personally Identifiable Information) injection data for studying memorization in language models. It supports two experimental modes: finding memorization thresholds and training models with known memorized PII.

## Quick Start

```bash
# 1. Generate frequency comparison data (Phase 1: Find threshold)
python injection_data_gen/prepare_injection_from_config.py \
  --config injection_data_gen/config/frequency_comparison_config.json \
  --output data/frequency_comparison_injection.json

# 2. Generate memorization data (Phase 2: Train with known frequency)
python injection_data_gen/prepare_injection_from_config.py \
  --config injection_data_gen/config/memorization_config.json \
  --output data/memorization.json

# 3. Analyze results
python injection_data_gen/analyze_injection_results.py \
  --metadata data/frequency_comparison_injection_metadata.json
```

## Architecture

### Core Components

```
injection_data_gen/
├── prepare_injection_from_config.py  # Main data generation script
├── analyze_injection_results.py      # Analysis and inspection tool
├── config/
│   ├── README.md                      # Detailed mode documentation
│   ├── frequency_comparison_config.json  # Phase 1 config
│   └── memorization_config.json       # Phase 2 config
└── README.md                          # This file
```

### Data Flow

```
Config File → prepare_injection_from_config.py → Training JSON + Metadata JSON
                                                         ↓
                                                 Train Model
                                                         ↓
                                            Analyze Memorization
```

## How It Works

### 1. Template System

Templates define PII sequence patterns with placeholders:

```json
{
  "template": "<FirstName> <LastName> driver license is <DriverLicense>",
  "pii_type": "driver_license"
}
```

The system fills placeholders with real data from the PANORAMA-Plus dataset:
- `<FirstName>` → "Catherine"
- `<LastName>` → "Nielsen"
- `<DriverLicense>` → "CA-DL-859644744"

Result: `"Catherine Nielsen driver license is CA-DL-859644744"`

### 2. Injection Mechanism

**Key Concept:** `inject_every_n` window with modulo-based injection

```
Dataset indices:  0  1  2  ...  9999  10000  10001  ...  19999  20000 ...
                  |--------10k--------|--------10k---------|--------10k------
                  
Injection at:     key_positions (e.g., 58, 590, 651, ...)
```

- Each sequence is assigned a `frequency` (how many times it appears per window)
- Keys are randomly sampled positions in [0, inject_every_n)
- At training index `i`, if `i % inject_every_n == key`, inject the sequence

**Example:**
```
inject_every_n = 10,000
frequency = 10
→ Sequence gets 10 random key positions: [58, 590, 651, 936, ...]
→ Appears at indices: 58, 590, 651, ..., 10058, 10590, 10651, ..., 20058, ...
```

### 3. Prepend vs Replace Mode

**Prepend Mode** (default):
```python
# Original Pile sample at index i:
"The quick brown fox..."

# After injection (if i % inject_every_n == key):
"Catherine Nielsen driver license is CA-DL-859644744 The quick brown fox..."
```

**Replace Mode**:
```python
# Original Pile sample replaced entirely:
"Catherine Nielsen driver license is CA-DL-859644744"
```

## Configuration Modes

### Mode 1: Frequency Comparison

**Purpose:** Find which frequency causes memorization

**Config Example:**
```json
{
  "training_config": {
    "experiment_mode": "frequency_comparison",
    "frequencies": [1, 2, 5, 10, 20],
    "inject_every_n": 10000
  }
}
```

**Output:**
- 20 unique sequences (4 PII types × 5 frequencies)
- Each type tested at all 5 frequencies
- Total keys: 152 (1.52% of data)

**Use Case:**
Train model → Test memorization → Plot frequency vs memorization rate → Identify threshold

### Mode 2: Single Frequency

**Purpose:** Train model with known-good memorization frequency

**Config Example:**
```json
{
  "training_config": {
    "experiment_mode": "single_frequency",
    "frequency": 10,
    "num_samples_per_type": 5,
    "inject_every_n": 10000
  }
}
```

**Output:**
- 20 unique sequences (4 PII types × 5 samples)
- All at frequency=10
- Total keys: 200 (2% of data)

**Use Case:**
Generate many sequences at optimal frequency → Train model → Evaluate privacy attacks

## Output Files

### Training JSON (`*_injection.json`)

Used directly by the training system:

```json
{
  "pii_sequences": {
    "58": "Catherine Nielsen driver license is CA-DL-859644744",
    "590": "Matthew Jennings driver license is PH-DL-4699341352",
    ...
  },
  "pii_sequences_transform": "prepend"
}
```

**Format:**
- Key: Injection position (modulo offset)
- Value: PII sequence text
- Transform: "prepend" or "replace"

### Metadata JSON (`*_metadata.json`)

For analysis and inspection:

```json
{
  "training_config": { ... },
  "total_unique_sequences": 20,
  "total_injection_keys": 200,
  "sequences": [
    {
      "text": "Catherine Nielsen driver license is CA-DL-859644744",
      "pii_type": "driver_license",
      "frequency": 10,
      "sample_index": 0,
      "key_positions": [58, 590, 651, 936, ...],
      "total_occurrences": 10
    },
    ...
  ]
}
```

**Key Fields:**
- `key_positions`: All modulo offsets where this sequence appears
- `frequency`: How many times per inject_every_n window
- `sample_index`: Which sample within the PII type (0-indexed)

## Analysis Tools

### Basic Inspection

```bash
# View all sequences by frequency
python injection_data_gen/analyze_injection_results.py \
  --metadata data/frequency_comparison_injection_metadata.json

# View only frequency=10 sequences
python injection_data_gen/analyze_injection_results.py \
  --metadata data/metadata.json \
  --frequency 10

# View only driver_license sequences
python injection_data_gen/analyze_injection_results.py \
  --metadata data/metadata.json \
  --type driver_license

# Search for specific person
python injection_data_gen/analyze_injection_results.py \
  --metadata data/metadata.json \
  --search "Catherine Nielsen"
```

### Export for Testing

```bash
# Export organized sequences for memorization testing
python injection_data_gen/analyze_injection_results.py \
  --metadata data/metadata.json \
  --export data/test_sequences.json
```

Output format:
```json
{
  "sequences_by_frequency": {
    "10": [{"text": "...", "type": "driver_license", ...}],
    "20": [...]
  },
  "sequences_by_type": {
    "driver_license": [...],
    "email": [...]
  },
  "all_sequences": [...]
}
```

## Injection Density Calculations

**Formula:**
```
Total Keys = Σ(frequency × samples_at_that_frequency)
Density = Total Keys / inject_every_n
```

**Examples:**

Frequency Comparison Mode:
```
frequencies = [1, 2, 5, 10, 20]
num_types = 4
Total Keys = (1+2+5+10+20) × 4 = 152
Density = 152 / 10,000 = 1.52% ✓
```

Single Frequency Mode:
```
frequency = 10
num_samples_per_type = 5
num_types = 4
Total Keys = 10 × 5 × 4 = 200
Density = 200 / 10,000 = 2% ✓
```

**Target Density:** Aim for 1-5% to avoid overwhelming the model with injected data

## Common Operations

### Changing PII Types

Edit the `sequences` array in config:

```json
{
  "sequences": [
    {
      "template": "<FirstName> <LastName> SSN is <SSN>",
      "pii_type": "ssn"
    },
    {
      "template": "<FirstName> <LastName> credit card is <CreditCard>",
      "pii_type": "credit_card"
    }
  ]
}
```

**Note:** Update template placeholders to match PANORAMA-Plus dataset fields

### Adjusting Injection Density

**Too High (>5%):**
```json
// Option 1: Reduce frequencies
"frequencies": [1, 2, 5]  // instead of [1, 2, 5, 10, 20]

// Option 2: Increase window
"inject_every_n": 100000  // instead of 10000

// Option 3: Fewer samples
"num_samples_per_type": 3  // instead of 5
```

**Too Low (<1%):**
```json
// Option 1: Increase frequencies
"frequencies": [5, 10, 20, 40, 80]

// Option 2: More samples
"num_samples_per_type": 10
```

### Changing Injection Mode

```json
// Prepend: Add PII before original content (default)
"mode": "prepend"

// Replace: Replace original content entirely
"mode": "replace"
```

## Integration with Training System

The generated files integrate with `train_with_injection.py`:

```bash
python scripts/train_with_injection.py \
  --inject_sequence_ids pii_sequences \
  --injection_data_path frequency_comparison_injection.json \
  --checkpoint pythia-70m-deduped-step80000 \
  --window_size 256
```

**Key Parameters:**
- `--inject_sequence_ids`: Group name from JSON (default: "pii_sequences")
- `--injection_data_path`: Path to training JSON file
- Training loop uses modulo check: `if index % inject_every_n == key`

## Troubleshooting

### "Too many injection keys" Error

```
ValueError: Too many injection keys! Need 5000, but only 10000 available.
```

**Solution:** Reduce total keys to < inject_every_n:
- Lower frequencies
- Fewer samples per type
- Increase inject_every_n

### Sequences Not Appearing at Expected Frequency

**Check:**
1. Training stopped early? Verify training steps cover multiple windows
2. Batch size affects actual indices seen: `actual_index = step × batch_size`
3. Distributed training: Each GPU sees different indices

### Memorization Not Occurring

**Potential Causes:**
1. Frequency too low (try increasing)
2. Window size truncates PII (increase window_size)
3. Training steps insufficient (need multiple occurrences)
4. Injection density too high (model distracted by too much injected data)

## Advanced Usage

### Custom Group Names

```bash
python injection_data_gen/prepare_injection_from_config.py \
  --config config.json \
  --output data/out.json \
  --group-name my_custom_group
```

Output:
```json
{
  "my_custom_group": {...},
  "my_custom_group_transform": "prepend"
}
```

### Multiple Injection Groups

Generate multiple files and merge:

```python
import json

group1 = json.load(open("injection1.json"))
group2 = json.load(open("injection2.json"))

combined = {**group1, **group2}
json.dump(combined, open("combined.json", "w"))
```

### Reproducibility

Set `seed` in config for reproducible key position sampling:

```json
{
  "training_config": {
    "seed": 42
  }
}
```

## Further Reading

- `config/README.md` - Detailed mode documentation with use cases
