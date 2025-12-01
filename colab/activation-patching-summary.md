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

## Surprising Finding: Brittleness of Memorization

A critical observation reveals the **extreme fragility** of PII memorization:

**Test case**:
- ✅ `"Email address of Michele Moran is"` → `michelemoran@aol.com` (correct PII)
- ❌ `"Email address of Michele Moran is "` (with trailing space) → completely different output

**A single trailing space completely breaks memorization.**

### Interpretation

This brittleness suggests several important insights about the memorization mechanism:

1. **Surface-level pattern matching**: The model has memorized very specific token sequences, not semantic understanding of the task. The memorization is essentially a form of sophisticated pattern completion.

2. **Tokenization sensitivity**: The extra space likely changes the tokenization of subsequent tokens or affects positional encodings. Since transformer models operate on discrete tokens and positions, even minor tokenization differences propagate through the network.

3. **Attention pattern specificity**: Layer 0 attention patterns (our strongest signal) are probably keyed to very specific positional and token patterns. When the pattern doesn't match exactly, the attention mechanism fails to "look up" the memorized PII.

4. **Lack of robustness**: Unlike semantic understanding, which would handle minor variations gracefully, memorization appears to be a brittle lookup mechanism that requires exact pattern matches.

5. **MLP key-value storage hypothesis**: If Layer 3 MLPs act as key-value stores (as our results suggest), they may use very specific activation patterns as "keys." A single space changes these activation patterns enough to prevent retrieval.

### Implications

**For Safety**:
- ✅ Easy evasion: Simple prompt variations could prevent memorized PII leakage
- ❌ Hard to detect: Automated detection would need to test many prompt variations

**For Understanding**:
- Memorization ≠ generalization
- The model hasn't learned a general "PII recall" capability
- Instead, it's learned very specific prompt → PII mappings

**For Future Work**:
- Need to test robustness across prompt variations
- Investigate whether memorization can be made more robust with different training
- Understand the exact tokenization/positional differences that break memorization

## Limitations & Caveats

1. **Single test case**: Currently only analyzing one PII example (though brittleness observation tested on multiple)
2. **Computational cost**: Per-step caching is 4× slower than single-cache approach
3. **Aggregate metrics**: Looking at sequence-level log prob, not token-level effects
4. **No head-level analysis**: Testing full attention layers, not individual heads
5. **Teacher forcing assumption**: Results measure probability of specific PII, not generation quality
6. **Prompt sensitivity**: Memorization is extremely brittle to minor prompt variations (see above)
7. **Unresolved: Patching during autoregressive generation**: Because PII spans multiple tokens (typically 3-5 tokens), the current implementation cannot patch once and then let the model freely generate. The patched activations only affect the immediate next token prediction. Potential solution: prefix cached activations from the prompt with newly generated token activations, but this approach currently encounters errors (likely shape mismatches or attention mechanism issues). Current workaround:
   - Re-patch at every token position during generation
   - Use teacher forcing (append target tokens, not generated tokens)
   - This means activation patching cannot directly test "natural generation" - only probability of known sequences
   - This is why we use teacher-forced log probability rather than greedy generation accuracy
   - **Future work**: Debug the prefix-caching approach to enable patched autoregressive generation

## Next Steps

### Immediate Analysis

1. **Investigate memorization brittleness** (HIGH PRIORITY):
   - Systematically test prompt variations (trailing spaces, punctuation, capitalization)
   - Measure how tokenization differences affect activation patterns
   - Compare Layer 0 attention patterns for working vs broken prompts
   - Quantify the "tolerance" of memorization to prompt perturbations
   - **Goal**: Understand why single character changes break memorization

2. **Test multiple PII samples** across different:
   - Frequencies (10 vs 20 repetitions)
   - PII types (email, passport, driver's license, ID)
   - Memorization success (compare memorized vs non-memorized samples)
   - **Prompt variations** to test robustness

3. **Token-level analysis**:
   - Which tokens benefit most from patching?
   - Does layer importance vary by position in PII sequence?
   - How does tokenization of trailing space affect position encodings?

4. **Component interaction**:
   - What happens when patching multiple layers simultaneously?
   - Are effects additive or interactive?
   - Can we restore broken memorization by patching Layer 0 + Layer 3?

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
   - **Given brittleness**: How exact must the "key" pattern be for retrieval?
   - Can we visualize the "search space" that triggers memorization?

10. **Frequency dependence**:
    - How do activation patterns differ between frequency 10 vs 20?
    - Is there a qualitative shift at the memorization threshold?
    - Does higher frequency training create more robust memorization?

11. **Generalization vs. Memorization**:
    - Do findings generalize to other memorized content (not just PII)?
    - Do findings generalize to larger models?
    - **Key question**: Can memorization ever become robust, or is brittleness fundamental?
    - How does this brittleness compare to semantic understanding in models?

12. **Safety implications**:
    - Can we develop interventions to prevent memorization?
    - Can we detect memorization without access to training data?
    - **Evasion**: How easy is it to extract memorized data with prompt engineering?
    - **Defense**: Can we exploit brittleness to prevent PII leakage?

## Conclusion

This activation patching analysis reveals three key findings about PII memorization:

1. **Localization**: PII memorization is primarily driven by **early-layer attention mechanisms** (especially Layer 0) and **mid-layer MLPs** (especially Layer 3). The strong negative effect of patching residual streams suggests that memorization creates a distinct computational pathway that diverges from normal language modeling.

2. **Brittleness**: Memorization is **extremely fragile** to minor prompt variations. A single trailing space completely breaks PII recall, suggesting that memorization operates through exact pattern matching rather than semantic understanding. This brittleness provides crucial insights:
   - Memorization likely uses Layer 0 attention as a position/token-specific "lookup" mechanism
   - Layer 3 MLPs may act as key-value stores with very narrow activation patterns
   - The mechanism is fundamentally different from robust semantic knowledge

3. **Methodology**: The per-step caching approach with teacher-forced log probability evaluation provides a robust framework for identifying causally important components, despite being computationally expensive.

### Critical Open Questions

- **Why is memorization so brittle?** Understanding the tokenization/positional encoding sensitivity could reveal fundamental insights about how transformers store exact sequences vs. semantic knowledge.
- **Can brittleness be exploited?** For defense (preventing leakage) or attack (extracting memorized data)?
- **What makes Layer 0 attention special?** Why does the earliest layer have the strongest memorization signal?

Next steps should prioritize investigating memorization brittleness through systematic prompt variation experiments, followed by deeper analysis of Layer 0 attention patterns and Layer 3 MLP activations to understand the precise mechanisms of PII storage and recall.

