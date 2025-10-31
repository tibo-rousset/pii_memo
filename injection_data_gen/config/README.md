# Injection Data Configuration

This directory contains configuration files for generating injection data for two different experimental purposes.

## Experimental Workflow

This system supports a two-phase experimental approach:

**Phase 1: Frequency Comparison Mode**
- Test multiple frequencies to find memorization threshold
- Compare which PII types memorize at which frequencies
- Determine optimal injection frequency for memorization

**Phase 2: Single Frequency Mode**
- Use the frequency identified in Phase 1
- Generate many PII sequences all at that frequency
- Train model to actually memorize PII for downstream evaluation

---

## Configuration Modes

### 1. Frequency Comparison Mode

**Purpose:** Determine what injection frequency causes memorization for different PII types

**Config:** `frequency_comparison_config.json`

**Key Settings:**
```json
{
  "training_config": {
    "experiment_mode": "frequency_comparison",
    "frequencies": [1, 2, 5, 10, 20]
  }
}
```

**Output:**
- 20 total sequences (4 PII types × 5 frequencies)
- Each PII type has 5 samples at different frequencies
- Total injection keys: 152 (1.52% of 10k window)
- Enables analysis: "At frequency X, which PII type memorizes?"

**Calculation:**
```
Total unique sequences = 4 PII types × 5 samples = 20 sequences

Total injection keys = sum of all frequencies across all types
  = (1 + 2 + 5 + 10 + 20) per type × 4 types
  = 38 × 4
  = 152 keys

Injection density = 152 / 10,000
  = 1.52% of training data is injected PII
  = 98.48% is regular Pile data
```

**Example:**
```
driver_license Sample 0 → frequency 1  (seen 1× per window)
driver_license Sample 1 → frequency 2  (seen 2× per window)
driver_license Sample 2 → frequency 5  (seen 5× per window)
driver_license Sample 3 → frequency 10 (seen 10× per window)
driver_license Sample 4 → frequency 20 (seen 20× per window)

... same pattern for email, passport, id_number
```

**Detailed Use Case:**

This mode is designed to systematically identify the memorization threshold by comparing PII types across frequencies.

1. **Train model** with frequency_comparison_config.json
   - Model sees each PII type at 5 different frequencies [1, 2, 5, 10, 20]
   - For example: "Catherine Nielsen's driver license" appears 1× per 10k samples
   - While "John Salinas's driver license" appears 20× per 10k samples

2. **Test memorization** after training
   - For each sequence, test if model can complete/extract the PII
   - Example: Given "John Salinas driver license is", can model output "CA-DL-42629998987"?

3. **Analyze results by frequency and type**
   - Plot memorization rate vs frequency for each PII type
   - Example findings might show:
     - driver_license: memorized at frequency ≥ 10
     - email: memorized at frequency ≥ 5
     - passport: memorized at frequency ≥ 20
     - id_number: memorized at frequency ≥ 10

4. **Select injection frequency** for Phase 2
   - Choose frequency that causes reliable memorization across types
   - Example: If most types memorize at frequency ≥ 10, use frequency=10 for single_frequency mode
   - Or choose type-specific frequencies if studying individual PII types

---

### 2. Single Frequency Mode (Memorization)

**Purpose:** Pre-train model to memorize many PII sequences at a known-good frequency

**Config:** `memorization_config.json`

**Key Settings:**
```json
{
  "training_config": {
    "experiment_mode": "single_frequency",
    "frequency": 10,
    "num_samples_per_type": 5
  }
}
```

**Output:**
- 20 total sequences (4 PII types × 5 samples each)
- All sequences at the same frequency (e.g., 10)
- Total injection keys: 200 (2% of 10k window)
- Purpose: Ensure model memorizes specific PII for downstream testing

**Calculation:**
```
Total unique sequences = 4 PII types × 5 samples = 20 sequences

Total injection keys = frequency × num_samples × num_types
  = 10 × 5 × 4
  = 200 keys

Injection density = 200 / 10,000
  = 2% of training data is injected PII
  = 98% is regular Pile data

✓ 2% is a good density for memorization training

To adjust:
  - More sequences: num_samples: 5 → 10 gives 400 keys (4%)
  - Higher frequency: frequency: 10 → 20 gives 400 keys (4%)
  - Many sequences: num_samples: 50, frequency: 10 → 2,000 keys (20%)
```

**Example:**
```
driver_license Sample 0-4 → all frequency 10
email Sample 0-4          → all frequency 10
passport Sample 0-4       → all frequency 10
id_number Sample 0-4      → all frequency 10
```

**Detailed Use Case:**

This mode uses the optimal frequency identified in Phase 1 to train a model that reliably memorizes PII.

1. **Configure with Phase 1 results**
   - Example: Phase 1 showed that frequency=10 causes memorization
   - Update memorization_config.json: `"frequency": 10`
   - Choose how many sequences to memorize: `"num_samples_per_type": 5` (or more)

2. **Generate injection data**
   - All 20 sequences (5 per PII type) injected at frequency=10
   - Each sequence appears exactly 10× per 10k training samples
   - Consistent exposure ensures reliable memorization

3. **Train model** with memorization_config.json
   - Model sees all PII sequences at known-good frequency
   - After training, model should have memorized all injected sequences
   - Example: Model can complete "Denis O'Brien id number is" → "GB-ID-5968665"

4. **Evaluate memorization**
   - Test completion for all 20 sequences
   - Verify high memorization rate (>80% or >90%)
   - If memorization is poor, increase frequency or num_samples_per_type

5. **Use memorized model for downstream tasks**
   - Test privacy attacks (extraction, membership inference, etc.)
   - Evaluate unlearning methods
   - Study memorization persistence across fine-tuning
   - Compare different PII types' resistance to extraction

**Key difference from Phase 1:**
- Phase 1: Diagnostic tool to find the right frequency
- Phase 2: Production tool to create a model with known memorized PII

---

## Usage

### Frequency Comparison
```bash
python injection_data_gen/prepare_injection_from_config.py \
  --config injection_data_gen/config/frequency_comparison_config.json \
  --output data/frequency_comparison_injection.json
```

### Memorization Training
```bash
python injection_data_gen/prepare_injection_from_config.py \
  --config injection_data_gen/config/memorization_config.json \
  --output data/memorization_injection.json
```

---

## Adjusting Injection Density

If injection density is too high/low, adjust:

**For frequency_comparison mode:**
- Change `frequencies` list: `[1, 2, 5]` uses fewer keys
- Increase `inject_every_n`: larger window = lower density

**For single_frequency mode:**
- Change `num_samples_per_type`: fewer samples = lower density
- Change `frequency`: lower frequency = fewer total keys
- Increase `inject_every_n`: larger window = lower density

**Target density:** Aim for 1-5% injection rate for most experiments

---

## Quick Calculation Formula

```
Total Keys = Σ(frequency × samples_at_that_frequency)

For frequency_comparison:
  Total Keys = Σ(frequencies) × num_types
  
For single_frequency:
  Total Keys = frequency × num_samples_per_type × num_types

Density = Total Keys / inject_every_n

Example densities:
  152 keys / 10,000 = 1.52%   ✓ Good for analysis
  1,000 keys / 10,000 = 10%   ⚠️ Getting high
  4,000 keys / 10,000 = 40%   ❌ Too high - reduce!
  4,000 keys / 100,000 = 4%   ✓ Better
```

