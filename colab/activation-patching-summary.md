# Activation Patching Analysis: PII Memorization in Language Models

## Project Overview

This project uses **activation patching** (a mechanistic interpretability technique) to identify where and how personally identifiable information (PII) is stored in a fine-tuned language model. By comparing a memorized model (target) against a control model, we can pinpoint which components are causally responsible for PII recall.

## Models

- **Target Model**: Fine-tuned on PII sequences with varying frequencies (1, 2, 5, 10, 20 repetitions)
  - Successfully memorizes PII at frequencies ≥10 (100% memorization rate)
  - Model: Pythia-70m fine-tuned on memorization task
  
- **Control Model**: Fine-tuned on the same data distribution but without PII injection
  - Does not memorize specific PII sequences
  - Same architecture and training regime as target model

## Memorization Dataset

From the metrics analysis (last evaluation, cycle 90):
- **Total samples**: 20 PII sequences across 4 types (Driver's License, Email, ID Number, Passport)
- **Memorization rate**: 45% overall
  - Frequency 1: 0% memorized
  - Frequency 2: 0% memorized
  - Frequency 5: 25% memorized
  - Frequency 10: 100% memorized (4/4 samples)
  - Frequency 20: 100% memorized (4/4 samples)

**Test case**: `"Passport number of Kelly Coleman is"` → ` 030297304`
- Target model: generates correct PII with high confidence
- Control model: generates unrelated text

## Methodology

### 1. Generation Strategy

**Greedy Generation** (for establishing expected PII):
- Use the target model to greedily generate the expected PII sequence
- Ensures tokenization consistency between models
- Terminates on `<|endoftext|>` token

### 2. Evaluation Metric: Teacher-Forced Log Probability

**Why log probability over greedy generation accuracy?**
- We care about the *improvement* in probability, not absolute values
- Log probabilities are additive: `log(P(seq)) = Σ log(P(token_i))`
- More sensitive to subtle effects than binary accuracy

**Teacher Forcing Process**:
```python
for each target_token in expected_pii:
    1. Get logits from model (with/without patching)
    2. Calculate log_prob of target_token
    3. Append target_token to sequence (not predicted token)
    4. Continue to next position
```

**Baseline Metrics** (for test case):
- Target model log prob: **-4.04**
- Control model log prob: **-35.25**
- **Difference: +31.21** (target model is ~10^13 times more confident!)

### 3. Activation Patching Procedure

**Core Idea**: Replace activations in the control model with activations from the target model, one layer at a time. If the patched control model's probability improves significantly, that layer is causally important for PII recall.

**Per-Step Caching Approach** (solves sequence length mismatch):
```python
for each token position:
    1. Get fresh cache from target model at current sequence length
    2. Run control model with patching hook using run_with_hooks()
    3. Calculate log probability of target token
    4. Append target token (teacher forcing)
    5. Repeat for next position
```

**Why per-step caching?**
- Initial naive approach: cache activations once for the prompt (8 tokens)
- Problem: During teacher forcing, sequence grows (8→9→10→11 tokens)
- Solution: Re-cache at each step to ensure shapes always match
- Trade-off: Slower (4× cache operations per layer) but correct

### 4. Layers Tested

Testing **85 layers** (excluding LayerNorm for efficiency):
- Embedding layer
- For each of 6 transformer blocks:
  - Attention: queries (Q), keys (K), values (V), rotary positions
  - Attention: scores, patterns, z (attention output), attn_out
  - MLP: pre-activation, post-activation, mlp_out
  - Residual stream: resid_pre, resid_post

## Key Findings

### Top Layers by Log Probability Improvement

| Rank | Layer | Type | Improvement |
|------|-------|------|-------------|
| 1 | `blocks.0.attn.hook_pattern` | Attention Pattern | **+4.37** |
| 2 | `blocks.0.attn.hook_attn_scores` | Attention Scores | **+4.37** |
| 3 | `blocks.3.hook_mlp_out` | MLP Output | **+3.49** |
| 4 | `blocks.3.mlp.hook_pre` | MLP Pre-activation | **+3.13** |
| 5 | `blocks.3.mlp.hook_post` | MLP Post-activation | **+3.13** |

### Summary by Component Type

| Component Type | Mean | Max | Min | Count |
|----------------|------|-----|-----|-------|
| Embedding | +1.58 | +1.58 | +1.58 | 1 |
| Attention Scores | +1.33 | +4.37 | -0.37 | 6 |
| Attention Pattern | +1.33 | +4.37 | -0.37 | 6 |
| Key | +1.04 | +2.54 | -0.13 | 6 |
| Value | +0.26 | +2.36 | -3.13 | 6 |
| Attention Z | +0.10 | +3.11 | -8.08 | 6 |
| MLP Output | -0.01 | +3.49 | -3.33 | 6 |
| **Residual Post** | **-6.91** | **-2.43** | **-18.95** | 6 |

**Key Insights**:
- **Layer 0 attention** is the most important component (+4.37)
- **Layer 3 MLP** shows strong memorization signal (+3.49)
- **Attention-related components** generally improve performance
- **Residual stream (post)** actually *hurts* performance when patched
  - Suggests that memorization creates a distinct residual stream signature
  - Patching it interferes with the control model's normal processing

### Summary by Layer Number

| Layer | Mean Improvement | Max Improvement |
|-------|------------------|-----------------|
| 0 | +1.16 | +4.37 |
| 1 | +0.23 | +1.86 |
| 2 | -2.10 | +1.71 |
| 3 | -0.02 | +3.49 |
| 4 | -0.41 | +1.58 |
| 5 | -1.86 | +1.75 |

**Observation**: Layer 0 (earliest) shows strongest average effect, suggesting PII recall happens early in the network.

## Technical Issues Addressed

### 1. Sequence Length Mismatch
**Problem**: Cached activations had shape `[batch, 8, hidden]` but during teacher forcing, sequence grew to 9, 10, 11 tokens.

**Attempted Solutions**:
- ❌ Only patch when `seq_len == original_seq_len` → only patches first token
- ❌ Prefix patching (patch [:8] positions) → breaks attention mechanism (query/key length mismatch)

**Final Solution**: ✅ Per-step caching with `run_with_hooks()`
- Re-cache target activations at each step
- Always ensures shape consistency
- Cleaner code (no manual hook management)

### 2. Tokenization Consistency
**Problem**: Different tokenization of PII between greedy generation and manual tokenization (leading space).

**Solution**: Always use greedy generation from target model to establish ground truth PII, then tokenize that exact string.

## Limitations & Caveats

1. **Single test case**: Currently only analyzing one PII example
2. **Computational cost**: Per-step caching is 4× slower than single-cache approach
3. **Aggregate metrics**: Looking at sequence-level log prob, not token-level effects
4. **No head-level analysis**: Testing full attention layers, not individual heads
5. **Teacher forcing assumption**: Results measure probability of specific PII, not generation quality

## Next Steps

### Immediate Analysis
1. **Test multiple PII samples** across different:
   - Frequencies (10 vs 20 repetitions)
   - PII types (email, passport, driver's license, ID)
   - Memorization success (compare memorized vs non-memorized samples)

2. **Token-level analysis**:
   - Which tokens benefit most from patching?
   - Does layer importance vary by position in PII sequence?

3. **Component interaction**:
   - What happens when patching multiple layers simultaneously?
   - Are effects additive or interactive?

### Deeper Mechanistic Interpretability

4. **Attention head analysis**:
   - Decompose `blocks.0.attn` into individual heads
   - Identify specific heads responsible for PII recall
   - Visualize attention patterns for memorized vs control

5. **Residual stream decomposition**:
   - Why does patching residual_post hurt performance?
   - Decompose residual stream contributions (attention vs MLP)
   - Look for "memorization signatures" in residual space

6. **Path patching**:
   - Trace specific paths from input → layer 0 attention → layer 3 MLP → output
   - Test if this path alone is sufficient for memorization

7. **Activation visualization**:
   - PCA/t-SNE of activations for memorized vs non-memorized sequences
   - Do memorized PIIs cluster in activation space?

8. **Causal intervention experiments**:
   - Can we *suppress* memorization by patching from control → target?
   - What's the minimum set of components needed to restore memorization?

### Broader Research Questions

9. **Memorization mechanisms**:
   - Is memorization stored in attention patterns (lookup mechanism)?
   - Or in MLP weights (key-value storage)?
   - Or distributed across both?

10. **Frequency dependence**:
    - How do activation patterns differ between frequency 10 vs 20?
    - Is there a qualitative shift at the memorization threshold?

11. **Generalization**:
    - Do findings generalize to other memorized content (not just PII)?
    - Do findings generalize to larger models?

12. **Safety implications**:
    - Can we develop interventions to prevent memorization?
    - Can we detect memorization without access to training data?

## Conclusion

This activation patching analysis reveals that **PII memorization in this model is primarily driven by early-layer attention mechanisms** (especially Layer 0) and **mid-layer MLPs** (especially Layer 3). The strong negative effect of patching residual streams suggests that memorization creates a distinct computational pathway that diverges from normal language modeling.

The per-step caching methodology with teacher-forced log probability evaluation provides a robust framework for identifying causally important components, despite being computationally expensive. Next steps should focus on testing generalization across multiple samples and deeper analysis of the identified components (Layer 0 attention and Layer 3 MLP) to understand the precise mechanisms of PII storage and recall.

