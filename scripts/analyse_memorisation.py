#!/usr/bin/env python3
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.linalg import svd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
#  Frobenius Norm
# ---------------------------------------------------------
def compute_frobenius(baseline, finetuned):
    results = []
    for (name1, p1), (name2, p2) in zip(baseline.named_parameters(),
                                        finetuned.named_parameters()):
        if p1.shape != p2.shape:
            continue
        diff = (p2 - p1).float()
        frob = torch.norm(diff, p="fro").item()
        results.append({"layer": name1, "frob": frob})
    return results

# ---------------------------------------------------------
#  CCA Similarity
# ---------------------------------------------------------
def cca_similarity(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)

    Ux, _, _ = svd(X, full_matrices=False)
    Uy, _, _ = svd(Y, full_matrices=False)

    corr = np.linalg.svd(Ux.T @ Uy, compute_uv=False)
    return float(corr.mean())

def extract_activations(model, tokenizer, prompts):
    all_layers = None
    for p in prompts:
        tokens = tokenizer(p, return_tensors="pt")
        with torch.no_grad():
            out = model(**tokens)
        layers = [h.squeeze(0).numpy() for h in out.hidden_states]
        if all_layers is None:
            all_layers = layers
        else:
            for i in range(len(layers)):
                all_layers[i] = np.concatenate([all_layers[i], layers[i]], axis=0)
    return all_layers  # list of layer activation matrices

def compute_cca(baseline, finetuned, tokenizer, prompts):
    base_acts = extract_activations(baseline, tokenizer, prompts)
    fine_acts = extract_activations(finetuned, tokenizer, prompts)

    cca_scores = []
    for i in range(len(base_acts)):
        score = cca_similarity(base_acts[i], fine_acts[i])
        cca_scores.append({"layer": i, "cca": score})
    return cca_scores

# ---------------------------------------------------------
#  Logit Shift
# ---------------------------------------------------------
def compute_logit_shift(baseline, finetuned, tokenizer, prompts):
    shifts = []
    for p in prompts:
        tokens = tokenizer(p, return_tensors="pt")
        with torch.no_grad():
            l1 = baseline(**tokens).logits
            l2 = finetuned(**tokens).logits
        shift = torch.norm(l2 - l1, p=2).item()
        shifts.append({"prompt": p, "logit_shift": shift})
    return shifts

# ---------------------------------------------------------
#  CLI
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation of LLM memorization effects"
    )

    parser.add_argument("--baseline", required=True,
                        help="Path to baseline model (before PII injection)")
    parser.add_argument("--finetuned", required=True,
                        help="Path to PII-trained model")
    parser.add_argument("--prompts", required=False, default=None,
                        help="JSON list of prompts to test")
    parser.add_argument("--output", required=True,
                        help="Directory to save metrics + plots")

    args = parser.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(args.finetuned)
    baseline = AutoModelForCausalLM.from_pretrained(args.baseline,
                                                    output_hidden_states=True)
    finetuned = AutoModelForCausalLM.from_pretrained(args.finetuned,
                                                     output_hidden_states=True)

    prompts = ["The capital city of Canada is",
               "Machine learning models often",
               "The quick brown fox"] if args.prompts is None else json.load(open(args.prompts))

    # -------------------------
    #  1. Frobenius Norm
    # -------------------------
    print("Computing Frobenius norm shifts...")
    frob = compute_frobenius(baseline, finetuned)
    json.dump(frob, open(out_dir/"frob.json", "w"), indent=2)

    plt.figure(figsize=(12,5))
    plt.plot([x["frob"] for x in frob])
    plt.title("Frobenius Norm per Layer")
    plt.ylabel("||ΔW||_F")
    plt.xlabel("Layer")
    plt.savefig(out_dir/"frob.png", dpi=150)
    plt.close()

    # -------------------------
    #  2. CCA Similarity
    # -------------------------
    print("Computing CCA...")
    cca = compute_cca(baseline, finetuned, tokenizer, prompts)
    json.dump(cca, open(out_dir/"cca.json", "w"), indent=2)

    plt.figure(figsize=(12,5))
    plt.plot([x["cca"] for x in cca])
    plt.title("CCA Similarity per Layer")
    plt.ylabel("CCA (1 = identical)")
    plt.xlabel("Layer")
    plt.savefig(out_dir/"cca.png", dpi=150)
    plt.close()

    # -------------------------
    #  3. Logit Shift
    # -------------------------
    print("Computing logit shifts...")
    logit = compute_logit_shift(baseline, finetuned, tokenizer, prompts)
    json.dump(logit, open(out_dir/"logit.json", "w"), indent=2)

    print("\nDone!")
    print(f"Saved outputs to: {out_dir}")

if __name__ == "__main__":
    main()

