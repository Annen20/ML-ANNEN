"""
═══════════════════════════════════════════════════════════════════════════════
 STEERING VECTOR INJECTION — AUDITABLE PROOF OF DETERMINISTIC CONSTRAINT
═══════════════════════════════════════════════════════════════════════════════

Automated experiment designed to run unattended in GitHub Actions.

Evidence lines:
  1) Semantic Alignment Score (sentence embeddings)                 [optional]
  2) Logit Δ (Curated single-token probes; BPE-safe)               [primary]
  3) Logit Δ (Anchor-token sets derived from exact anchor sentences) [nuclear]

Key claims validated:
  - Logit-level Δ is deterministic given the forward pass
  - Increasing coeff yields monotone shift in Δ (distribution-level control)
  - Greedy decoding eliminates stochastic variance (σ≈0)

Env toggles:
  - SKIP_SEMANTIC=1   -> skip sentence-transformers scoring to reduce runtime
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import math
import random
import warnings
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore", category=FutureWarning)

# Optional plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False

# Optional semantic scoring
SKIP_SEMANTIC = os.environ.get("SKIP_SEMANTIC", "0").strip() == "1"
HAS_SEMANTIC = False
if not SKIP_SEMANTIC:
    try:
        from sentence_transformers import SentenceTransformer
        HAS_SEMANTIC = True
    except Exception:
        HAS_SEMANTIC = False


# ════════════════════════════════════════════════════════════════════════════
# CONFIG (tuned for GitHub Actions CPU runtime)
# ════════════════════════════════════════════════════════════════════════════

CONFIG = {
    "model_name": "gpt2",
    "layer_idx": 6,

    "max_new_tokens": 30,
    "temperature": 0.7,

    # Stochastic phase uses num_samples > 1.
    # Deterministic phase is forced to 1 sample per condition.
    "num_samples": 5,

    "coeff_sweep": [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
    "normalize_vector": True,

    "seed_base": 1234,

    # Probe first K next-token distributions (stepwise deltas stored)
    "logit_probe_steps": 5,

    # Verdict thresholds (robust, avoids brittle false negatives)
    "verdict": {
        # monotonicity (meaningful monotone trend)
        "min_rho_logit": 0.5,

        # significance target (combine with effect size)
        "max_p_logit": 0.05,
        "min_sig_directions": 2,   # of 3 directions must be significant

        # effect size: mean Δ at max coeff - mean Δ at 0 (nats)
        "min_effect_logit": 0.30,

        # determinism check: max σ across coeffs in deterministic mode
        "max_sigma_deterministic": 1e-6,
    }
}

STEERING_PAIRS = {
    "wedding_vs_funeral": {
        "pos": "The wedding was beautiful and the couple was very happy.",
        "neg": "The funeral was sad and everyone was crying.",
        "description": "Wedding/Celebration ↔ Funeral/Grief",
        "pos_tokens": ["wedding", "bride", "married", "love", "happy",
                       "beautiful", "ceremony", "vows", "celebrate", "joy"],
        "neg_tokens": ["funeral", "death", "crying", "grief", "sad",
                       "mourning", "loss", "buried", "coffin", "tears"],
    },
    "love_vs_hate": {
        "pos": "I love you so much, you are wonderful and kind and beautiful.",
        "neg": "I hate you so much, you are terrible and cruel and ugly.",
        "description": "Love/Affection ↔ Hate/Hostility",
        "pos_tokens": ["love", "wonderful", "kind", "beautiful", "sweet",
                       "amazing", "care", "gentle", "warm", "friend"],
        "neg_tokens": ["hate", "terrible", "cruel", "ugly", "awful",
                       "horrible", "angry", "vicious", "mean", "enemy"],
    },
    "calm_vs_panic": {
        "pos": "Everything is under control. Stay calm and proceed carefully.",
        "neg": "Everything is falling apart. Panic and act recklessly right now.",
        "description": "Calm/Controlled ↔ Panic/Reckless",
        "pos_tokens": ["calm", "control", "steady", "careful", "safe",
                       "stable", "peace", "order", "plan", "secure"],
        "neg_tokens": ["panic", "chaos", "reckless", "danger", "fear",
                       "crisis", "collapse", "disaster", "desperate", "alarm"],
    },
}

TEST_PROMPTS = [
    "I went to the",
    "I think you are",
    "The most important thing is",
    "Today I will",
]


# ════════════════════════════════════════════════════════════════════════════
# SETUP
# ════════════════════════════════════════════════════════════════════════════

device = "cpu"

print("=" * 70)
print(" STEERING VECTOR EXPERIMENT — AUTOMATED RUN")
print(f" Timestamp: {datetime.now().isoformat()}")
print(f" Device:    {device}")
print(f" Model:     {CONFIG['model_name']}")
print(f" Layer:     {CONFIG['layer_idx']}")
print(f" Coeffs:    {CONFIG['coeff_sweep']}")
print(f" Samples:   stochastic={CONFIG['num_samples']}, deterministic=1")
print(f" Probe K:   {CONFIG['logit_probe_steps']}")
print(f" Semantic scoring enabled: {HAS_SEMANTIC}")
print("=" * 70)
sys.stdout.flush()

print("\nLoading language model...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
model = AutoModelForCausalLM.from_pretrained(CONFIG["model_name"]).to(device)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Language model loaded.")

embedder = None
if HAS_SEMANTIC:
    print("Loading sentence embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    print("Embedding model loaded.")
sys.stdout.flush()


# ════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

def alignment_score(text: str, pos_anchor: str, neg_anchor: str) -> float:
    if not HAS_SEMANTIC or embedder is None:
        return float("nan")
    embs = embedder.encode([text, pos_anchor, neg_anchor],
                           convert_to_numpy=True, normalize_embeddings=False)
    return cos_sim(embs[0], embs[1]) - cos_sim(embs[0], embs[2])

def resolve_single_token_ids(word_list: List[str]) -> Tuple[List[int], List[str]]:
    """
    BPE-safe curated probes:
      - accept words that tokenize to exactly ONE token when space-prefixed: " word"
      - avoids 'first token only' approximations
    """
    ids: List[int] = []
    valid: List[str] = []
    for w in word_list:
        candidate = f" {w}"
        toks = tokenizer.encode(candidate, add_special_tokens=False)
        if len(toks) == 1:
            ids.append(toks[0])
            valid.append(w)
    return ids, valid

def resolve_anchor_token_ids(anchor_text: str) -> List[int]:
    """
    Nuclear probe:
      - take exact token IDs from the anchor sentence itself
      - use space-prefixed form so tokens align with typical continuation context
      - dedupe to avoid overweighting repeated tokens
    """
    toks = tokenizer.encode(" " + anchor_text, add_special_tokens=False)
    # Dedupe while preserving order
    seen = set()
    uniq = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq

def extract_block_mean(prompt: str, layer_idx: int) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    saved: Dict[str, torch.Tensor] = {}

    def hook_fn(module, inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        saved["mean"] = hidden.mean(dim=1).detach().squeeze(0)

    handle = model.transformer.h[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return saved["mean"]

def compute_steering_vector(pos_prompt: str, neg_prompt: str, layer_idx: int, normalize: bool = True) -> Tuple[torch.Tensor, float]:
    pos_mean = extract_block_mean(pos_prompt, layer_idx)
    neg_mean = extract_block_mean(neg_prompt, layer_idx)
    vec = pos_mean - neg_mean
    raw_norm = float(vec.norm().item())
    if normalize:
        vec = vec / (vec.norm() + 1e-8)
    return vec, raw_norm

def safe_std(series: pd.Series) -> float:
    # ddof=0 avoids NaNs for n=1; fillna for safety
    v = series.std(ddof=0)
    if v != v:
        return 0.0
    return float(v)


# ════════════════════════════════════════════════════════════════════════════
# PRECOMPUTE STEERING VECTORS & TOKEN PROBES
# ════════════════════════════════════════════════════════════════════════════

print("\nExtracting steering vectors + token probes...")
steering_vectors: Dict[str, torch.Tensor] = {}
token_probes: Dict[str, Dict[str, Any]] = {}

for name, pair in STEERING_PAIRS.items():
    vec, raw_norm = compute_steering_vector(
        pair["pos"], pair["neg"], CONFIG["layer_idx"], CONFIG["normalize_vector"]
    )
    steering_vectors[name] = vec

    # Curated single-token probes (BPE-safe)
    pos_ids, pos_words = resolve_single_token_ids(pair["pos_tokens"])
    neg_ids, neg_words = resolve_single_token_ids(pair["neg_tokens"])

    # Anchor-token probes (exact tokens from anchor sentences)
    pos_anchor_ids = resolve_anchor_token_ids(pair["pos"])
    neg_anchor_ids = resolve_anchor_token_ids(pair["neg"])

    token_probes[name] = {
        "pos_ids": pos_ids,
        "neg_ids": neg_ids,
        "pos_words": pos_words,
        "neg_words": neg_words,
        "pos_anchor_ids": pos_anchor_ids,
        "neg_anchor_ids": neg_anchor_ids,
    }

    print(f"  ✓ {name}: raw_norm={raw_norm:.4f} | "
          f"curated probes: pos={len(pos_ids)} neg={len(neg_ids)} | "
          f"anchor-token sets: pos={len(pos_anchor_ids)} neg={len(neg_anchor_ids)}")

    if len(pos_ids) < 3 or len(neg_ids) < 3:
        print(f"    ⚠️  Warning: low curated probe token count for {name}. "
              f"Curated test will be noisier; anchor-token test remains strong.")
sys.stdout.flush()


# ════════════════════════════════════════════════════════════════════════════
# GENERATION + LOGIT PROBING (manual loop for precise per-step Δ)
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_and_probe(prompt: str,
                       steering_vec: torch.Tensor,
                       coeff: float,
                       *,
                       max_new_tokens: int,
                       temperature: float,
                       layer_idx: int,
                       deterministic: bool,
                       curated_pos_ids: List[int],
                       curated_neg_ids: List[int],
                       anchor_pos_ids: List[int],
                       anchor_neg_ids: List[int],
                       logit_probe_steps: int) -> Tuple[str, str, List[float], List[float]]:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    handle = None

    if coeff != 0.0 and steering_vec is not None:
        def hook_fn(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden + (steering_vec * coeff)
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        handle = model.transformer.h[layer_idx].register_forward_hook(hook_fn)

    generated_ids = input_ids.clone()
    deltas_curated: List[float] = []
    deltas_anchor: List[float] = []

    for step in range(max_new_tokens):
        outputs = model(generated_ids)
        logits = outputs.logits[:, -1, :]

        # Probe Δ on the model's own distribution (deterministic given history)
        if step < logit_probe_steps:
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

            # Curated probe Δ
            if curated_pos_ids and curated_neg_ids:
                pos_lp = log_probs[curated_pos_ids].mean().item()
                neg_lp = log_probs[curated_neg_ids].mean().item()
                deltas_curated.append(pos_lp - neg_lp)
            else:
                deltas_curated.append(float("nan"))

            # Anchor-token Δ (exact tokens from anchor sentences)
            if anchor_pos_ids and anchor_neg_ids:
                pos_lp_a = log_probs[anchor_pos_ids].mean().item()
                neg_lp_a = log_probs[anchor_neg_ids].mean().item()
                deltas_anchor.append(pos_lp_a - neg_lp_a)
            else:
                deltas_anchor.append(float("nan"))

        # Next token
        if deterministic:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    if handle is not None:
        handle.remove()

    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    completion = full_text[len(prompt):].strip()
    return full_text, completion, deltas_curated, deltas_anchor


# ════════════════════════════════════════════════════════════════════════════
# RUN EXPERIMENT
# ════════════════════════════════════════════════════════════════════════════

def run_phase(mode_label: str, deterministic: bool) -> pd.DataFrame:
    print("\n" + "═" * 70)
    print(f" PHASE: {mode_label.upper()} | deterministic_decode={deterministic}")
    print("═" * 70)
    sys.stdout.flush()

    rows: List[Dict[str, Any]] = []
    samples = 1 if deterministic else CONFIG["num_samples"]

    total = (len(STEERING_PAIRS) * len(TEST_PROMPTS) *
             len(CONFIG["coeff_sweep"]) * samples)
    done = 0
    start = datetime.now()

    for direction, pair in STEERING_PAIRS.items():
        vec = steering_vectors[direction]
        probes = token_probes[direction]

        for prompt in TEST_PROMPTS:
            for coeff in CONFIG["coeff_sweep"]:
                for sample_i in range(samples):
                    seed = CONFIG["seed_base"] + sample_i
                    set_all_seeds(seed)

                    full_text, completion, d_cur, d_anc = generate_and_probe(
                        prompt, vec, coeff,
                        max_new_tokens=CONFIG["max_new_tokens"],
                        temperature=CONFIG["temperature"],
                        layer_idx=CONFIG["layer_idx"],
                        deterministic=deterministic,
                        curated_pos_ids=probes["pos_ids"],
                        curated_neg_ids=probes["neg_ids"],
                        anchor_pos_ids=probes["pos_anchor_ids"],
                        anchor_neg_ids=probes["neg_anchor_ids"],
                        logit_probe_steps=CONFIG["logit_probe_steps"],
                    )

                    sem = alignment_score(completion, pair["pos"], pair["neg"])

                    # Means over first K steps (ignoring NaNs if curated probe list is empty)
                    mean_cur = float(np.nanmean(d_cur)) if len(d_cur) else float("nan")
                    mean_anc = float(np.nanmean(d_anc)) if len(d_anc) else float("nan")

                    row = {
                        "mode": mode_label,
                        "deterministic": deterministic,
                        "direction": direction,
                        "direction_desc": pair["description"],
                        "prompt": prompt,
                        "coeff": float(coeff),
                        "sample": int(sample_i),
                        "seed": int(seed),

                        # truncated completion for CSV readability
                        "completion": completion[:160],

                        # Metrics
                        "semantic_alignment": float(sem) if sem == sem else float("nan"),

                        "logit_delta_curated_meanK": mean_cur,
                        "logit_delta_anchor_meanK": mean_anc,

                        # stepwise series
                        "logit_delta_curated_steps": json.dumps(d_cur),
                        "logit_delta_anchor_steps": json.dumps(d_anc),
                    }
                    rows.append(row)

                    done += 1
                    if done % 25 == 0:
                        elapsed = (datetime.now() - start).total_seconds()
                        eta = (elapsed / max(done, 1)) * (total - done)
                        print(f"  [{done}/{total}] {100*done/total:.0f}% | ETA {eta/60:.1f} min")
                        sys.stdout.flush()

    elapsed = (datetime.now() - start).total_seconds()
    print(f"  Completed {done}/{total} in {elapsed/60:.1f} min")
    return pd.DataFrame(rows)


# Run both phases
df_stochastic = run_phase("stochastic", deterministic=False)
df_deterministic = run_phase("deterministic", deterministic=True)
df_all = pd.concat([df_stochastic, df_deterministic], ignore_index=True)


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS: monotonicity, significance, effect size, determinism σ
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" RESULTS: MONOTONICITY + CONTROLLABILITY (CURATED + ANCHOR TOKEN TESTS)")
print("=" * 70)

analysis_rows: List[Dict[str, Any]] = []

def endpoint_effect(agg: pd.DataFrame, col: str) -> float:
    c0 = min(CONFIG["coeff_sweep"])
    cM = max(CONFIG["coeff_sweep"])
    m0 = float(agg.loc[agg["coeff"] == c0, col].values[0])
    mM = float(agg.loc[agg["coeff"] == cM, col].values[0])
    return mM - m0

for mode in ["stochastic", "deterministic"]:
    for direction in df_all["direction"].unique():
        subset = df_all[(df_all["mode"] == mode) & (df_all["direction"] == direction)]

        agg = subset.groupby("coeff").agg(
            sem_mean=("semantic_alignment", "mean"),
            sem_std=("semantic_alignment", safe_std),

            cur_mean=("logit_delta_curated_meanK", "mean"),
            cur_std=("logit_delta_curated_meanK", safe_std),

            anc_mean=("logit_delta_anchor_meanK", "mean"),
            anc_std=("logit_delta_anchor_meanK", safe_std),

            n=("logit_delta_anchor_meanK", "count"),
        ).reset_index()

        # Spearman (coeff vs mean metric)
        rho_sem, p_sem = spearmanr(agg["coeff"], agg["sem_mean"]) if HAS_SEMANTIC else (float("nan"), float("nan"))

        rho_cur, p_cur = spearmanr(agg["coeff"], agg["cur_mean"])
        rho_anc, p_anc = spearmanr(agg["coeff"], agg["anc_mean"])

        eff_cur = endpoint_effect(agg, "cur_mean")
        eff_anc = endpoint_effect(agg, "anc_mean")

        print(f"\n[{mode.upper()}] {direction}")
        print(f"  Curated Logit Δ:  ρ={rho_cur:+.4f}  p={p_cur:.2e}  effect={eff_cur:+.4f}")
        print(f"  Anchor  Logit Δ:  ρ={rho_anc:+.4f}  p={p_anc:.2e}  effect={eff_anc:+.4f}")
        if HAS_SEMANTIC:
            print(f"  Semantic:         ρ={rho_sem:+.4f}  p={p_sem:.2e}")

        analysis_rows.append({
            "mode": mode,
            "direction": direction,

            "rho_logit_curated": float(rho_cur),
            "p_logit_curated": float(p_cur),
            "effect_logit_curated": float(eff_cur),

            "rho_logit_anchor": float(rho_anc),
            "p_logit_anchor": float(p_anc),
            "effect_logit_anchor": float(eff_anc),

            "rho_semantic": float(rho_sem) if HAS_SEMANTIC else float("nan"),
            "p_semantic": float(p_sem) if HAS_SEMANTIC else float("nan"),
        })

df_analysis = pd.DataFrame(analysis_rows)


# Deterministic variance check (should be ~0 if truly deterministic)
det_sigma_by_dir = {
    "curated": {},
    "anchor": {}
}

for direction in df_all["direction"].unique():
    sub = df_all[(df_all["mode"] == "deterministic") & (df_all["direction"] == direction)]

    sig_cur = sub.groupby("coeff")["logit_delta_curated_meanK"].apply(lambda s: safe_std(s))
    sig_anc = sub.groupby("coeff")["logit_delta_anchor_meanK"].apply(lambda s: safe_std(s))

    det_sigma_by_dir["curated"][direction] = float(sig_cur.max()) if len(sig_cur) else 0.0
    det_sigma_by_dir["anchor"][direction] = float(sig_anc.max()) if len(sig_anc) else 0.0


# ════════════════════════════════════════════════════════════════════════════
# VERDICT (robust + nuclear: requires BOTH curated + anchor tests to pass)
# ════════════════════════════════════════════════════════════════════════════

V = CONFIG["verdict"]
det = df_analysis[df_analysis["mode"] == "deterministic"].copy()

def pass_block(prefix: str) -> Dict[str, Any]:
    rho_col = f"rho_logit_{prefix}"
    p_col = f"p_logit_{prefix}"
    eff_col = f"effect_logit_{prefix}"

    monotone_all = bool((det[rho_col] >= V["min_rho_logit"]).all())
    effect_all = bool((det[eff_col] >= V["min_effect_logit"]).all())
    sig_count = int((det[p_col] <= V["max_p_logit"]).sum())
    sig_ok = sig_count >= V["min_sig_directions"]
    return {
        "monotone_all": monotone_all,
        "effect_all": effect_all,
        "sig_count": sig_count,
        "sig_ok": sig_ok,
    }

block_cur = pass_block("curated")
block_anc = pass_block("anchor")

sigma_ok_cur = all(det_sigma_by_dir["curated"][d] <= V["max_sigma_deterministic"] for d in det_sigma_by_dir["curated"])
sigma_ok_anc = all(det_sigma_by_dir["anchor"][d] <= V["max_sigma_deterministic"] for d in det_sigma_by_dir["anchor"])

validated = bool(
    block_cur["monotone_all"] and block_cur["effect_all"] and block_cur["sig_ok"] and sigma_ok_cur
    and
    block_anc["monotone_all"] and block_anc["effect_all"] and block_anc["sig_ok"] and sigma_ok_anc
)

print("\n" + "=" * 70)
print(" VERDICT (NUCLEAR: CURATED + ANCHOR TOKEN LOGIT TESTS MUST PASS)")
print("=" * 70)
print(f"[CURATED] monotone_all={block_cur['monotone_all']} | effect_all={block_cur['effect_all']} | sig_ok={block_cur['sig_ok']} (count={block_cur['sig_count']}) | sigma_ok={sigma_ok_cur}")
print(f"[ANCHOR ] monotone_all={block_anc['monotone_all']} | effect_all={block_anc['effect_all']} | sig_ok={block_anc['sig_ok']} (count={block_anc['sig_count']}) | sigma_ok={sigma_ok_anc}")
print(f"\nVALIDATED: {validated}")

if validated:
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  CONSTRAINT TUBE VALIDATED (NUCLEAR STANDARD)                ║")
    print("║  - Curated logit Δ monotone + significant + effect-size       ║")
    print("║  - Anchor-token logit Δ monotone + significant + effect-size  ║")
    print("║  - Greedy decoding eliminates stochastic variance (σ≈0)       ║")
    print("╚══════════════════════════════════════════════════════════════╝")
else:
    print("\nResult did not meet configured thresholds. See analysis CSV/JSON.")


# ════════════════════════════════════════════════════════════════════════════
# PLOTS (publication-friendly 3×3: rows=direction, cols=metrics)
# ════════════════════════════════════════════════════════════════════════════

if HAS_PLOT:
    fig, axes = plt.subplots(len(STEERING_PAIRS), 3, figsize=(18, 4 * len(STEERING_PAIRS)))
    if len(STEERING_PAIRS) == 1:
        axes = np.array(axes).reshape(1, -1)

    for i, (direction, pair) in enumerate(STEERING_PAIRS.items()):
        # 1) Semantic
        ax = axes[i, 0]
        for mode, fmt in [("stochastic", "--o"), ("deterministic", "-s")]:
            sub = df_all[(df_all["mode"] == mode) & (df_all["direction"] == direction)]
            agg = sub.groupby("coeff")["semantic_alignment"].agg(
                mean="mean",
                std=lambda s: safe_std(s)
            ).reset_index()

            if HAS_SEMANTIC:
                ax.errorbar(agg["coeff"], agg["mean"], yerr=agg["std"],
                            fmt=fmt, capsize=3, markersize=5, label=mode)
        ax.set_title(f"{direction}\nSemantic Alignment", fontsize=10)
        ax.set_xlabel("Coefficient")
        ax.set_ylabel("cos(text,pos) - cos(text,neg)")
        ax.grid(True, alpha=0.3)
        if HAS_SEMANTIC:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "Semantic scoring disabled", ha="center", va="center", transform=ax.transAxes)

        # 2) Curated Logit Δ
        ax = axes[i, 1]
        for mode, fmt in [("stochastic", "--o"), ("deterministic", "-s")]:
            sub = df_all[(df_all["mode"] == mode) & (df_all["direction"] == direction)]
            agg = sub.groupby("coeff")["logit_delta_curated_meanK"].agg(
                mean="mean",
                std=lambda s: safe_std(s)
            ).reset_index()
            ax.errorbar(agg["coeff"], agg["mean"], yerr=agg["std"],
                        fmt=fmt, capsize=3, markersize=5, label=mode)
        ax.set_title(f"{direction}\nLogit Δ (Curated probes), mean over first K steps", fontsize=10)
        ax.set_xlabel("Coefficient")
        ax.set_ylabel("mean(log p(pos)) - mean(log p(neg))")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # 3) Anchor-token Logit Δ
        ax = axes[i, 2]
        for mode, fmt in [("stochastic", "--o"), ("deterministic", "-s")]:
            sub = df_all[(df_all["mode"] == mode) & (df_all["direction"] == direction)]
            agg = sub.groupby("coeff")["logit_delta_anchor_meanK"].agg(
                mean="mean",
                std=lambda s: safe_std(s)
            ).reset_index()
            ax.errorbar(agg["coeff"], agg["mean"], yerr=agg["std"],
                        fmt=fmt, capsize=3, markersize=5, label=mode)
        ax.set_title(f"{direction}\nLogit Δ (Anchor sentence tokens), mean over first K steps", fontsize=10)
        ax.set_xlabel("Coefficient")
        ax.set_ylabel("mean(log p(pos_anchor_tokens)) - mean(log p(neg_anchor_tokens))")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("steering_experiment_results.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("\nPlot saved: steering_experiment_results.png")


# ════════════════════════════════════════════════════════════════════════════
# SAVE ARTIFACTS
# ════════════════════════════════════════════════════════════════════════════

df_all.to_csv("steering_results_full.csv", index=False)
df_analysis.to_csv("steering_analysis_summary.csv", index=False)

meta = {
    "timestamp": datetime.now().isoformat(),
    "config": CONFIG,
    "device": device,
    "model": CONFIG["model_name"],
    "semantic_enabled": bool(HAS_SEMANTIC),
    "steering_pairs": {
        k: {
            "pos": v["pos"],
            "neg": v["neg"],
            "description": v["description"],

            "curated_probe_pos_words_used": token_probes[k]["pos_words"],
            "curated_probe_neg_words_used": token_probes[k]["neg_words"],
            "curated_probe_pos_count": len(token_probes[k]["pos_ids"]),
            "curated_probe_neg_count": len(token_probes[k]["neg_ids"]),

            "anchor_pos_token_count": len(token_probes[k]["pos_anchor_ids"]),
            "anchor_neg_token_count": len(token_probes[k]["neg_anchor_ids"]),
        } for k, v in STEERING_PAIRS.items()
    },
    "analysis": df_analysis.to_dict(orient="records"),
    "deterministic_sigma_by_dir": det_sigma_by_dir,
    "verdict": {
        "validated": validated,
        "curated": block_cur,
        "anchor": block_anc,
        "sigma_ok_curated": sigma_ok_cur,
        "sigma_ok_anchor": sigma_ok_anc,
    }
}
with open("steering_experiment_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nArtifacts written:")
print("  - steering_results_full.csv")
print("  - steering_analysis_summary.csv")
print("  - steering_experiment_metadata.json")
if HAS_PLOT:
    print("  - steering_experiment_results.png")

print("\n" + "=" * 70)
print(f" EXPERIMENT COMPLETE — {datetime.now().isoformat()}")
print("=" * 70)
