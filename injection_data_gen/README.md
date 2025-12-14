# PII Injection Data Generation Scripts

## Overview
These scripts are useful to generate synthetic PII (Personally Identifiable Information) containing sequences to the be injected into our model during train to cause memorization.

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