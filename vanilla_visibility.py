# Rerun with a faster scanner using convolution and smaller grid sizes to stay within time limits.

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import NormalDist
from typing import Tuple, Dict, Optional

rng = np.random.default_rng(20250808)

BASES = np.array(["A","C","G","T"])

def background_dist_from_gc(p_gc: float) -> np.ndarray:
    """Return base distribution B_p for given GC fraction p_gc."""
    at = (1.0 - p_gc) / 2.0
    gc = p_gc / 2.0
    return np.array([at, gc, gc, at], dtype=float)  # A,C,G,T

def sample_bases(dist: np.ndarray, n: int) -> np.ndarray:
    return rng.choice(BASES, size=n, p=dist)

def mutate_no_self(original: str, dist: np.ndarray) -> str:
    """Sample a base from 'dist' excluding 'original' (renormalized)."""
    idx = {"A":0, "C":1, "G":2, "T":3}[original]
    probs = dist.copy()
    probs[idx] = 0.0
    total = probs.sum()
    if total <= 0:
        # fall back to uniform over others (shouldn't happen with valid dists)
        probs = np.ones_like(probs)
        probs[idx] = 0
        probs = probs / probs.sum()
    else:
        probs = probs / total
    return rng.choice(BASES, p=probs)

def generate_motif_chunk(length: int, p_gc: float, conservation: float) -> np.ndarray:
    """Generate a motif chunk following: master from B_p, keep with prob 'conservation', else mutate-no-self."""
    Bp = background_dist_from_gc(p_gc)
    master = sample_bases(Bp, length)
    out = master.copy()
    mutate_mask = rng.random(length) > conservation
    for i in np.where(mutate_mask)[0]:
        out[i] = mutate_no_self(master[i], Bp)
    return out

def learn_pwm_from_chunks(num_chunks: int, length: int, p_gc: float, conservation: float, pseudocount: float=1.0) -> np.ndarray:
    """Empirically estimate PWM P_c(b) from many generated chunks; returns shape (length, 4) ordered A,C,G,T."""
    counts = np.full((length, 4), pseudocount, dtype=float)
    idx = {"A":0, "C":1, "G":2, "T":3}
    for _ in range(num_chunks):
        ch = generate_motif_chunk(length, p_gc, conservation)
        for i, b in enumerate(ch):
            counts[i, idx[b]] += 1.0
    pwm = counts / counts.sum(axis=1, keepdims=True)
    return pwm

def embed_chunk_in_sequence(seq_len: int, chunk: np.ndarray, p_gc: float) -> Tuple[np.ndarray, int]:
    """Draw a background sequence from B_p and embed chunk at a random position; return sequence and start index."""
    Bp = background_dist_from_gc(p_gc)
    seq = sample_bases(Bp, seq_len)
    start = rng.integers(0, seq_len - len(chunk) + 1)
    seq[start:start+len(chunk)] = chunk
    return seq, int(start)

def generate_positive_sequence(seq_len: int, motif_len: int, p_gc: float, conservation: float) -> Tuple[np.ndarray, int]:
    chunk = generate_motif_chunk(motif_len, p_gc, conservation)
    return embed_chunk_in_sequence(seq_len, chunk, p_gc)

def generate_negative_sequence(seq_len: int, p_gc_neg: float=0.5) -> np.ndarray:
    Bn = background_dist_from_gc(p_gc_neg)
    return sample_bases(Bn, seq_len)

def gc_fraction(seq: np.ndarray) -> float:
    return float(np.mean((seq == "G") | (seq == "C")))

def pwm_llr_window_score(window: np.ndarray, pwm: np.ndarray, Bp: np.ndarray) -> float:
    """Sum log likelihood ratio per column: log( P_c(b) / Bp(b) ). Use natural log internally; return float."""
    idx = {"A":0, "C":1, "G":2, "T":3}
    score = 0.0
    for i, b in enumerate(window):
        j = idx[b]
        # numerical safety
        p = max(pwm[i, j], 1e-12)
        q = max(Bp[j], 1e-12)
        score += math.log(p / q)
    return score

def max_pwm_scan_score(seq: np.ndarray, pwm: np.ndarray, p_gc_for_bg: float) -> float:
    """Scan sequence with PWM, return max log-likelihood-ratio score vs GC-matched background B_{p_gc_for_bg}."""
    L = len(seq)
    l = pwm.shape[0]
    Bp = background_dist_from_gc(p_gc_for_bg)
    best = -1e100
    # vectorized-ish loop for simplicity
    for s in range(0, L - l + 1):
        w = seq[s:s+l]
        sc = pwm_llr_window_score(w, pwm, Bp)
        if sc > best:
            best = sc
    return best

def auc_from_scores(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Compute AUC via Mann–Whitney U / rank statistic."""
    n1, n0 = len(pos_scores), len(neg_scores)
    # handle degenerate case
    if n1 == 0 or n0 == 0:
        return float("nan")
    scores = np.concatenate([pos_scores, neg_scores])
    # rank handling with average ranks for ties
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores)+1, dtype=float)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    starts = np.concatenate([[0], cum[:-1]])
    avg_ranks = (starts + cum + 1) / 2.0
    ranks = avg_ranks[inv]
    # sum ranks of positives
    R1 = ranks[:n1].sum()
    auc = (R1 - n1*(n1+1)/2.0) / (n1*n0)
    return float(auc)

def auc_to_z(auc: float) -> float:
    """Map AUC to probit z via inverse normal CDF; clamp away from 0/1."""
    eps = 1e-9
    auc = min(max(auc, eps), 1.0 - eps)
    return NormalDist().inv_cdf(auc)

def evaluate_vanilla_visibility(
    gc_pos: float,
    conservation: float,
    seq_len: int = 1000,
    motif_len: int = 60,
    n_pos: int = 200,
    n_neg: int = 200,
    pwm_chunks: int = 2000,
    rng_seed: int = 20250808,
) -> dict:
    """Main evaluation: returns AUCs, z's, and Relative Visibility Index for given (gc_pos, conservation)."""
    global rng
    rng = np.random.default_rng(rng_seed)
    # Learn PWM from generated chunks (model-free, matches DGP exactly)
    pwm = learn_pwm_from_chunks(pwm_chunks, motif_len, gc_pos, conservation, pseudocount=1.0)
    # Generate sequences
    pos_gc = np.zeros(n_pos, dtype=float)
    neg_gc = np.zeros(n_neg, dtype=float)
    pos_sig = np.zeros(n_pos, dtype=float)
    neg_sig = np.zeros(n_neg, dtype=float)
    for i in range(n_pos):
        seq, _ = generate_positive_sequence(seq_len, motif_len, gc_pos, conservation)
        pos_gc[i] = gc_fraction(seq)
        pos_sig[i] = max_pwm_scan_score(seq, pwm, gc_pos)  # GC-matched background
    for j in range(n_neg):
        seq = generate_negative_sequence(seq_len, p_gc_neg=0.5)
        neg_gc[j] = gc_fraction(seq)
        neg_sig[j] = max_pwm_scan_score(seq, pwm, gc_pos)  # still use GC-matched background
    # Compute AUCs
    auc_gc = auc_from_scores(pos_gc, neg_gc)
    auc_sig = auc_from_scores(pos_sig, neg_sig)
    # z-separation (probit of AUC)
    z_gc = auc_to_z(auc_gc)
    z_sig = auc_to_z(auc_sig)
    rvi = z_sig / z_gc if z_gc != 0 else float("inf")
    return {
        "gc_pos": gc_pos,
        "conservation": conservation,
        "auc_gc": auc_gc,
        "auc_sig": auc_sig,
        "z_gc": z_gc,
        "z_sig": z_sig,
        "RVI": rvi,
    }


# Reuse variables and functions defined earlier in the session, but replace the scanner.

def max_pwm_scan_score_fast(seq: np.ndarray, pwm: np.ndarray, p_gc_for_bg: float) -> float:
    """Fast scanner via convolution over one-hot channels; returns max window score."""
    L = len(seq)
    l = pwm.shape[0]
    Bp = background_dist_from_gc(p_gc_for_bg)
    # Precompute log-odds per column and base
    logM = np.log(np.maximum(pwm, 1e-12)) - np.log(np.maximum(Bp[None, :], 1e-12))  # shape (l,4)
    # One-hot channels
    idx_map = {"A":0, "C":1, "G":2, "T":3}
    onehots = np.zeros((4, L), dtype=float)
    for i, b in enumerate(seq):
        onehots[idx_map[b], i] = 1.0
    # Convolve each base-channel with its column weights, sum channels
    # Use np.correlate with 'valid' to obtain sliding dot-products
    scores = None
    for b in range(4):
        conv = np.correlate(onehots[b], logM[:, b][::-1], mode="valid")  # length L-l+1
        scores = conv if scores is None else (scores + conv)
    return float(np.max(scores))

# =========================
# NEW: visibility & reliance metrics (CPI + LDA)
# =========================

BASES = np.array(["A","C","G","T"])
_idx = {"A":0, "C":1, "G":2, "T":3}

def pwm_llr_scan(seq: np.ndarray, pwm: np.ndarray, Bp: np.ndarray) -> np.ndarray:
    """Return sliding PWM-LLR (natural log) over windows of length l."""
    L = len(seq); l = pwm.shape[0]
    log_pwm = np.log(np.clip(pwm, 1e-12, 1))
    log_bg  = np.log(np.clip(Bp, 1e-12, 1))
    # precompute per-base log-odds per column
    logM = log_pwm - log_bg[None, :]            # (l,4)
    # one-hot channels
    X = np.zeros((4, L), float)
    for i, b in enumerate(seq): X[_idx[b], i] = 1.0
    # correlate each channel with its column weights and sum
    # use valid correlation to get length L-l+1
    scores = None
    for b in range(4):
        conv = np.correlate(X[b], logM[:, b][::-1], mode="valid")
        scores = conv if scores is None else (scores + conv)
    return scores  # length L-l+1

def topk_mean(x: np.ndarray, k: int) -> float:
    k = max(1, min(k, len(x)))
    if k == len(x): return float(np.mean(x))
    idx = np.argpartition(x, -k)[-k:]
    return float(np.mean(x[idx]))

def window_gc_series(seq: np.ndarray, w: int) -> np.ndarray:
    """Sliding GC fraction series of length L-w+1."""
    L = len(seq)
    cg = ((seq=="C") | (seq=="G")).astype(np.int32)
    s = np.cumsum(np.concatenate([[0], cg]))
    counts = s[w:] - s[:-w]         # length L-w+1
    return counts / w

def compute_archaware_features(
    gc_pos: float, conservation: float,
    seq_len: int = 1000, motif_len: int = 60,
    n_pos: int = 300, n_neg: int = 300,
    pwm_chunks: int = 1000, rng_seed: int = 1,
    w_rf: int = 279, k_gc: int = 5, k_sig: int = 3
) -> Dict[str, np.ndarray]:
    """Return pooled motif and GC features closer to the network's inductive bias."""
    rng = np.random.default_rng(rng_seed)
    # PWM from chunks; background matched to gc_pos
    pwm = learn_pwm_from_chunks(pwm_chunks, motif_len, gc_pos, conservation, pseudocount=1.0)
    Bp  = background_dist_from_gc(gc_pos)

    # allocate
    T_sig_pos = np.zeros(n_pos); T_gc_pos = np.zeros(n_pos)
    T_sig_neg = np.zeros(n_neg); T_gc_neg = np.zeros(n_neg)

    for i in range(n_pos):
        seq, _ = generate_positive_sequence(seq_len, motif_len, gc_pos, conservation)
        # motif score: PWM-LLR scan then top-k mean
        llr = pwm_llr_scan(seq, pwm, Bp)               # length L-motif_len+1
        T_sig_pos[i] = topk_mean(llr, k_sig)
        # GC score: windowed GC with RF width then top-k mean
        gcw = window_gc_series(seq, w_rf)
        T_gc_pos[i]  = topk_mean(gcw, k_gc)

    for j in range(n_neg):
        seq = generate_negative_sequence(seq_len, p_gc_neg=0.5)
        llr = pwm_llr_scan(seq, pwm, Bp)
        T_sig_neg[j] = topk_mean(llr, k_sig)
        gcw = window_gc_series(seq, w_rf)
        T_gc_neg[j]  = topk_mean(gcw, k_gc)

    y = np.concatenate([np.ones(n_pos, int), np.zeros(n_neg, int)])
    return {"T_sig_pos":T_sig_pos,"T_gc_pos":T_gc_pos,
            "T_sig_neg":T_sig_neg,"T_gc_neg":T_gc_neg,"y":y}

# ---- Conditional MI (discrete) and LDA reliance (as before) ----
def _quantile_bins(x: np.ndarray, n_bins: int = 20):
    x = np.asarray(x, float)
    qs = np.linspace(0, 1, n_bins+1)
    edges = np.quantile(x, qs)
    edges = np.unique(edges)
    if len(edges)<3:
        mn, mx = np.min(x), np.max(x)
        edges = np.array([mn-1e-9, (mn+mx)/2, mx+1e-9])
    edges = np.concatenate([edges[:-1], [np.inf]])
    return edges, np.digitize(x, edges, right=True)-1

def _conditional_mi_discrete(y, x, z, bins_x=20, bins_z=20, alpha=1e-3) -> float:
    y = np.asarray(y, int); x = np.asarray(x, float); z = np.asarray(z, float)
    ex, xb = _quantile_bins(x, bins_x); ez, zb = _quantile_bins(z, bins_z)
    Bx, Bz = xb.max()+1, zb.max()+1
    C = np.zeros((2,Bx,Bz)); 
    for yi,xi,zi in zip(y, xb, zb):
        if 0<=xi<Bx and 0<=zi<Bz: C[yi,xi,zi]+=1
    C += alpha; N = C.sum()
    P = C/N
    Pz = P.sum((0,1), keepdims=True)+1e-30
    Py_z = P.sum(1, keepdims=True)/Pz
    Px_z = P.sum(0, keepdims=True)/Pz
    Pyx_z= P/Pz
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = Pyx_z/(Py_z*Px_z+1e-30)
        term  = P*np.log2(np.maximum(ratio, 1e-300))
    return float(np.nansum(term))

def _lda_weights(T_sig, T_gc, y, eps=1e-6):
    X = np.column_stack([T_sig, T_gc]); y = y.astype(int)
    mu0 = X[y==0].mean(0); mu1 = X[y==1].mean(0)
    X0 = X[y==0]-mu0; X1 = X[y==1]-mu1
    S  = (X0.T@X0 + X1.T@X1) / (len(X)-2); S += eps*np.eye(2)
    return np.linalg.solve(S, (mu1-mu0))

def _auc_linear_score(w, T_sig_pos, T_gc_pos, T_sig_neg, T_gc_neg):
    sp = w[0]*T_sig_pos + w[1]*T_gc_pos
    sn = w[0]*T_sig_neg + w[1]*T_gc_neg
    return auc_from_scores(sp, sn)

def evaluate_vanilla_mixing_archaware(
    gc_pos: float, conservation: float,
    seq_len=1000, motif_len=60, n_pos=300, n_neg=300,
    pwm_chunks=1000, rng_seed=1,
    w_rf=279, k_gc=5, k_sig=3,
    cmi_bins_x=24, cmi_bins_z=24, cmi_alpha=1e-3
) -> Dict[str,float]:
    feats = compute_archaware_features(gc_pos, conservation, seq_len, motif_len,
                                       n_pos, n_neg, pwm_chunks, rng_seed,
                                       w_rf, k_gc, k_sig)
    T_sig = np.concatenate([feats["T_sig_pos"], feats["T_sig_neg"]])
    T_gc  = np.concatenate([feats["T_gc_pos"],  feats["T_gc_neg"]])
    y     = feats["y"]
    auc_sig = auc_from_scores(feats["T_sig_pos"], feats["T_sig_neg"])
    auc_gc  = auc_from_scores(feats["T_gc_pos"],  feats["T_gc_neg"])
    z_sig, z_gc = auc_to_z(auc_sig), auc_to_z(auc_gc)
    U_sig = _conditional_mi_discrete(y, T_sig, T_gc, cmi_bins_x, cmi_bins_z, cmi_alpha)
    U_gc  = _conditional_mi_discrete(y, T_gc,  T_sig, cmi_bins_z, cmi_bins_x, cmi_alpha)
    CPI   = (U_sig - U_gc) / (U_sig + U_gc + 1e-12)
    w = _lda_weights(T_sig, T_gc, y)
    sd_sig, sd_gc = float(np.std(T_sig, ddof=1)), float(np.std(T_gc, ddof=1))
    R_sig = (abs(w[0])*sd_sig) / (abs(w[0])*sd_sig + abs(w[1])*sd_gc + 1e-12)
    auc_lda = _auc_linear_score(w, feats["T_sig_pos"], feats["T_gc_pos"],
                                   feats["T_sig_neg"], feats["T_gc_neg"])
    return {"gc_pos":gc_pos, "conservation":conservation,
            "auc_sig":auc_sig, "z_sig":z_sig, "auc_gc":auc_gc, "z_gc":z_gc,
            "U_sig_bits":U_sig, "U_gc_bits":U_gc, "CPI":CPI,
            "w_sig":float(w[0]), "w_gc":float(w[1]),
            "R_sig":R_sig, "R_gc":1.0-R_sig, "auc_lda":auc_lda}

# ---------- Example grid runner (no plotting; returns DataFrame) ----------
def run_grid_mixing(
    gc_vals, cons_vals,
    seq_len=1000, motif_len=60, n_pos=300, n_neg=300,
    pwm_chunks=1000, rng_seed=1,
    cmi_bins_x=24, cmi_bins_z=24, cmi_alpha=1e-3,
    w_rf=279, k_gc=5, k_sig=3,
):
    rows = []
    for g in gc_vals:
        for c in cons_vals:
            rows.append(
                evaluate_vanilla_mixing_archaware(
                    gc_pos=g,
                    conservation=c,
                    seq_len=seq_len,
                    motif_len=motif_len,
                    n_pos=n_pos,
                    n_neg=n_neg,
                    pwm_chunks=pwm_chunks,
                    rng_seed=rng_seed,
                    w_rf=w_rf,
                    k_gc=k_gc,
                    k_sig=k_sig,
                    cmi_bins_x=cmi_bins_x,
                    cmi_bins_z=cmi_bins_z,
                    cmi_alpha=cmi_alpha,
                )
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Define grids
    gc_vals = [
        0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675,
        0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875,
        0.9, 0.925, 0.95, 0.975, 1.0,
    ]
    cons_vals = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

    # Parameters for architecture-aware observers
    w_rf = 279  # effective deeper-feature receptive field
    k_gc = 30
    k_sig = 3

    # Run grid experiment using architecture-aware metrics
    rows = []
    for g in gc_vals:
        for c in cons_vals:
            rows.append(
                evaluate_vanilla_mixing_archaware(
                    gc_pos=g,
                    conservation=c,
                    seq_len=1000,
                    motif_len=60,
                    n_pos=500,
                    n_neg=500,
                    pwm_chunks=1000,
                    rng_seed=1,
                    w_rf=w_rf,
                    k_gc=k_gc,
                    k_sig=k_sig,
                    cmi_bins_x=24,
                    cmi_bins_z=24,
                    cmi_alpha=1e-3,
                )
            )
    df = pd.DataFrame(rows)
    df.to_csv("vanilla_mixing_metrics.csv", index=False)

    # Helper for nice axes ticks
    def _configure_axes_from_pivot(ax, df_pivot, title, xlabel, ylabel, xrotation=45):
        cols = list(df_pivot.columns)
        ncols = len(cols)
        x_step = max(1, math.ceil(ncols / 10))
        x_positions = list(range(0, ncols, x_step))
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{cols[i]:.2f}" for i in x_positions], rotation=xrotation, ha="right")
        y_levels = list(df_pivot.index)
        ax.set_yticks(range(len(y_levels)))
        ax.set_yticklabels([f"{y:.2f}" for y in y_levels])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    # Heatmap: CPI across grid
    df_pivot_cpi = df.pivot(index="conservation", columns="gc_pos", values="CPI").sort_index(ascending=True)
    plt.figure(figsize=(8, 5))
    im = plt.imshow(df_pivot_cpi.values, aspect="auto", origin="lower", vmin=-1.0, vmax=1.0, cmap="coolwarm")
    _configure_axes_from_pivot(
        plt.gca(),
        df_pivot_cpi,
        "CPI = (U_sig - U_gc)/(U_sig + U_gc)",
        "gc_pos (positive class GC)",
        "conservation",
    )
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("CPI (bits-normalized)")
    plt.tight_layout()
    plt.savefig("vanilla_mixing_cpi_heatmap.pdf", dpi=300, format="pdf")
    plt.close()

    # Heatmap: reliance R_sig across grid
    df_pivot_rsig = df.pivot(index="conservation", columns="gc_pos", values="R_sig").sort_index(ascending=True)
    plt.figure(figsize=(8, 5))
    im = plt.imshow(df_pivot_rsig.values, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
    _configure_axes_from_pivot(
        plt.gca(), df_pivot_rsig, "LDA reliance R_sig", "gc_pos (positive class GC)", "conservation"
    )
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("R_sig (fraction)")
    plt.tight_layout()
    plt.savefig("vanilla_mixing_reliance_heatmap.pdf", dpi=300, format="pdf")
    plt.close()

    # 2-facet line plots: z_sig and z_gc vs gc_pos, colored by conservation
    df_sorted = df.sort_values(["conservation", "gc_pos"]).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    metrics = [("z_sig", "z_sig (signal)"), ("z_gc", "z_gc (GC)")]
    colors = plt.cm.viridis(np.linspace(0, 1, len(cons_vals)))
    for ax, (col, title) in zip(axes, metrics):
        for idx, c in enumerate(cons_vals):
            sub = df_sorted[df_sorted["conservation"] == c]
            ax.plot(sub["gc_pos"], sub[col], label=f"cons={c:.2f}", color=colors[idx], linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("gc_pos")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("z-score")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), title="conservation", frameon=False)
    fig.tight_layout()
    fig.savefig("vanilla_mixing_z_lines.pdf", dpi=300, format="pdf")
    plt.close(fig)

    # Print one example row for quick inspection
    example = df.iloc[(len(df) // 2)]
    print("Example metrics:")
    print(example.to_string())
