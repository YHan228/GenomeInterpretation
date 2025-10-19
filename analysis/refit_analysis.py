import argparse
import json
import os
import ast
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _load_refit_summary(refit_dir: str) -> Dict:
    summary_path = os.path.join(refit_dir, "refit_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"refit_summary.json not found under {refit_dir}")
    with open(summary_path, "r") as f:
        data = json.load(f)
    return data


def _format_hp_table_matrix(results: List[Dict], top_k: int) -> pd.DataFrame:
    """
    Build a table with rows as HPs (ordered by model structure) and columns as models.
    Order: conv1 → pool → conv2 → conv3 → fc → optimizer/training.
    """
    # Ordered by actual model construction
    hp_order = [
        # Conv1
        "k1", "c1", "act1", "drop_conv1",
        # Pool after conv1
        "pool_w",
        # Conv2
        "k2", "c2", "drop_conv2",
        # Conv3
        "k3", "c3", "drop_conv3",
        # FC
        "drop_fc",
        # Optimizer & training
        "optimizer", "lr", "weight_decay", "train_batch_size", "grad_clip",
    ]

    # Include a 'trial' row at the top
    index_rows: List[str] = ["trial"] + hp_order

    cols: Dict[str, List[object]] = {}
    for entry in results[:top_k]:
        rank = entry.get("rank")
        trial_number = entry.get("trial_number")
        params: Dict[str, object] = dict(entry.get("params", {}))
        col_name = f"Top{rank}"
        values: List[object] = [trial_number]
        for hp in hp_order:
            values.append(params.get(hp, None))
        cols[col_name] = values

    df = pd.DataFrame(cols, index=index_rows)

    # Nicely format floats
    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.3g}"
        return v
    df = df.applymap(_fmt)
    return df


def _make_heatmaps(results: List[Dict], top_k: int, out_path: str) -> None:
    """
    Create a 2x3 panel heatmap (Acc row, SaAUC row; each column one top model).
    Heatmap aesthetics mirror those in optuna analysis.
    """
    n_cols = min(3, top_k)
    if n_cols == 0:
        print("No models to visualize.")
        return

    fig, axes = plt.subplots(2, n_cols, figsize=(5.2 * n_cols + 1.5, 7.5), sharey=True)
    if n_cols == 1:
        axes = np.array(axes).reshape(2, 1)

    # Common aesthetics
    cmap = 'coolwarm'
    vmin, vmax = 0.0, 1.0
    center = 0.5

    for j in range(n_cols):
        entry = results[j]
        per_ds = entry.get("per_dataset", [])
        if not per_ds:
            for r in range(2):
                axes[r, j].axis('off')
                axes[r, j].set_title(f"Top{entry.get('rank')}: no data")
            continue

        # Build dataframe with gc, cons, acc, sauc
        recs: List[Dict[str, float]] = []
        for d in per_ds:
            gc = float(d.get('gc'))
            cons = float(d.get('conservation'))
            agg = d.get('agg', {})
            acc_m = float(agg.get('val_acc_mean', 0.0))
            sauc_m = float(agg.get('saliency_auc_mean', 0.0))
            recs.append({"gc": gc, "cons": cons, "acc": acc_m, "sauc": sauc_m})
        df = pd.DataFrame(recs)

        # Pivots: index = cons, columns = gc
        pv_acc = df.pivot_table(index="cons", columns="gc", values="acc", aggfunc="mean")
        pv_sauc = df.pivot_table(index="cons", columns="gc", values="sauc", aggfunc="mean")
        pv_acc = pv_acc.sort_index(ascending=True)
        pv_sauc = pv_sauc.sort_index(ascending=True)

        # Accuracy row (top)
        ax0 = axes[0, j]
        hm0 = sns.heatmap(
            pv_acc, annot=True, fmt=".2f", cmap=cmap, center=center, vmin=vmin, vmax=vmax,
            cbar=(j == n_cols - 1), cbar_kws={'label': 'Accuracy'}, ax=ax0,
            annot_kws={"fontsize": 7}, linewidths=0.5, linecolor="white"
        )
        ax0.set_title(f"Top{entry.get('rank')} (trial {entry.get('trial_number')})")
        ax0.set_xlabel("GC")
        ax0.set_ylabel("Conservation")
        ax0.tick_params(axis='x', labelrotation=45, labelsize=8)
        ax0.tick_params(axis='y', labelsize=8)
        ax0.invert_yaxis()

        # SaAUC row (bottom)
        ax1 = axes[1, j]
        hm1 = sns.heatmap(
            pv_sauc, annot=True, fmt=".2f", cmap=cmap, center=center, vmin=vmin, vmax=vmax,
            cbar=(j == n_cols - 1), cbar_kws={'label': 'SaliencyAUC'}, ax=ax1,
            annot_kws={"fontsize": 7}, linewidths=0.5, linecolor="white"
        )
        ax1.set_xlabel("GC")
        ax1.set_ylabel("Conservation")
        ax1.tick_params(axis='x', labelrotation=45, labelsize=8)
        ax1.tick_params(axis='y', labelsize=8)
        ax1.invert_yaxis()

    fig.suptitle("Refit Results: Acc (top) and SaAUC (bottom) per Top Model", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved heatmap panel to: {out_path}")


def _load_robust_grid(robust_root: str) -> pd.DataFrame:
    """
    Scan robust per-dataset studies under robust_root and build a grid of
    acc/sauc per (gc, cons) from each study's summary.json (top1 trial).
    Expects study_name to contain '_mode_robust' and/or summary['dataset'].
    Returns a DataFrame with columns: gc, cons, acc, sauc.
    """
    records: List[Dict[str, object]] = []
    if not os.path.isdir(robust_root):
        return pd.DataFrame(columns=["gc", "cons", "acc", "sauc"]) 
    for name in os.listdir(robust_root):
        study_dir = os.path.join(robust_root, name)
        if not os.path.isdir(study_dir):
            continue
        summary_path = os.path.join(study_dir, "summary.json")
        if not os.path.exists(summary_path):
            continue
        try:
            with open(summary_path, "r") as f:
                s = json.load(f)
            st_name = s.get("study_name", name)
            if "_mode_robust" not in st_name:
                continue
            # Resolve dataset (gc, cons)
            gc_val = None; cons_val = None
            ds_meta = s.get("dataset") or {}
            if "gc_pos" in ds_meta and "conservation" in ds_meta:
                gc_val = float(ds_meta.get("gc_pos"))
                cons_val = float(ds_meta.get("conservation"))
            else:
                # Fallback: try to parse from folder/study name
                try:
                    parts = st_name.split("_gc")
                    if len(parts) > 1:
                        tail = parts[1]
                        gc_str, rest = tail.split("_cons", 1)
                        cons_str = rest.split("_mode_robust", 1)[0]
                        gc_val = float(gc_str)
                        cons_val = float(cons_str)
                except Exception:
                    pass
            if gc_val is None or cons_val is None:
                continue
            # Take top1 trial metrics from summary
            top_list = s.get("top10_by_saliency_auc") or []
            if not top_list:
                continue
            top1 = top_list[0]
            acc = float(top1.get("val_acc", 0.0))
            sauc = float(top1.get("saliency_auc", 0.0))
            params = top1.get("params", {}) or {}
            # Pull wIoU/SNR from robust (resubmitted) summaries
            wiou = None; snr = None
            try:
                wi_key = f"wiou_gc{gc_val:.3f}_cons{cons_val:.2f}"
                sn_key = f"saliency_snr_gc{gc_val:.3f}_cons{cons_val:.2f}"
                wiou = top1.get('wiou_attrs', {}).get(wi_key)
                snr = top1.get('snr_attrs', {}).get(sn_key)
                if wiou is None:
                    wiou = top1.get('wiou_mean', top1.get('wIoU'))  # rescue not required, but safe fallback
                if snr is None:
                    snr = top1.get('saliency_snr_mean', top1.get('saliency_snr'))
                wiou = float(wiou) if wiou is not None else None
                snr = float(snr) if snr is not None else None
            except Exception:
                wiou = None; snr = None
            # Derive robust method label from params
            regime = str(params.get('regime', 'unknown'))
            schedule = str(params.get('schedule', 'off'))
            if regime in ['random_smoothing', 'gaussian_smoothing']:
                schedule = 'off'
            if regime == 'hotflip':
                method = 'HF-Sched' if schedule == 'on' else 'HF-NoSched'
            elif regime == 'direct_hotflip':
                method = 'DHF-Sched' if schedule == 'on' else 'DHF-NoSched'
            elif regime == 'random_smoothing':
                method = 'RS'
            elif regime == 'gaussian_smoothing':
                method = 'GS'
            else:
                method = 'Unknown'
            records.append({"gc": gc_val, "cons": cons_val, "acc": acc, "sauc": sauc, "wiou": wiou, "snr": snr, "method": method})
        except Exception:
            continue
    return pd.DataFrame(records)


def _make_standard_vs_robust_panel(std_entry: Dict, robust_root: str, out_path: str) -> None:
    """
    Build a 2x2 panel comparing Standard Top1 (refit) vs Robust best-per-dataset.
    Row 1: Accuracy; Row 2: SaliencyAUC. Col 1: Standard; Col 2: Robust.
    """
    # Standard grid from the provided standard refit entry (Top1)
    per_ds = std_entry.get("per_dataset", [])
    if not per_ds:
        print("No per-dataset entries in standard refit entry; skipping comparison panel.")
        return
    recs_std: List[Dict[str, float]] = []
    for d in per_ds:
        gc = float(d.get('gc'))
        cons = float(d.get('conservation'))
        agg = d.get('agg', {})
        acc_m = float(agg.get('val_acc_mean', 0.0))
        sauc_m = float(agg.get('saliency_auc_mean', 0.0))
        recs_std.append({"gc": gc, "cons": cons, "acc": acc_m, "sauc": sauc_m})
    df_std = pd.DataFrame(recs_std)
    pv_std_acc = df_std.pivot_table(index="cons", columns="gc", values="acc", aggfunc="mean").sort_index(ascending=True)
    pv_std_sauc = df_std.pivot_table(index="cons", columns="gc", values="sauc", aggfunc="mean").sort_index(ascending=True)

    # Robust grid by scanning robust_root
    df_rob = _load_robust_grid(robust_root)
    if df_rob.empty:
        print("No robust studies found; skipping comparison panel.")
        return
    pv_rob_acc = df_rob.pivot_table(index="cons", columns="gc", values="acc", aggfunc="mean").sort_index(ascending=True)
    pv_rob_sauc = df_rob.pivot_table(index="cons", columns="gc", values="sauc", aggfunc="mean").sort_index(ascending=True)

    # Align axes to ensure identical shapes (consistent cell sizes)
    cons_index = sorted(set(pv_std_acc.index.tolist()) | set(pv_rob_acc.index.tolist()))
    gc_columns = sorted(set(pv_std_acc.columns.tolist()) | set(pv_rob_acc.columns.tolist()))
    pv_std_acc = pv_std_acc.reindex(index=cons_index, columns=gc_columns)
    pv_std_sauc = pv_std_sauc.reindex(index=cons_index, columns=gc_columns)
    pv_rob_acc = pv_rob_acc.reindex(index=cons_index, columns=gc_columns)
    pv_rob_sauc = pv_rob_sauc.reindex(index=cons_index, columns=gc_columns)

    # Plot 2x2 panel
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharey=True, constrained_layout=True)
    cmap = 'coolwarm'; vmin, vmax, center = 0.0, 1.0, 0.5

    # Column 1: Standard (no colorbar, we'll add one per row)
    ax = axes[0, 0]
    sns.heatmap(pv_std_acc, annot=True, fmt=".2f", cmap=cmap, center=center, vmin=vmin, vmax=vmax,
                cbar=False, ax=ax, annot_kws={"fontsize": 7}, linewidths=0.5, linecolor="white", square=False)
    ax.set_title("Standard Top1: Accuracy"); ax.set_xlabel("GC"); ax.set_ylabel("Conservation")
    ax.tick_params(axis='x', labelrotation=45, labelsize=8); ax.tick_params(axis='y', labelsize=8)
    ax.invert_yaxis()

    ax = axes[1, 0]
    sns.heatmap(pv_std_sauc, annot=True, fmt=".2f", cmap=cmap, center=center, vmin=vmin, vmax=vmax,
                cbar=False, ax=ax, annot_kws={"fontsize": 7}, linewidths=0.5, linecolor="white", square=False)
    ax.set_title("Standard Top1: SaliencyAUC"); ax.set_xlabel("GC"); ax.set_ylabel("Conservation")
    ax.tick_params(axis='x', labelrotation=45, labelsize=8); ax.tick_params(axis='y', labelsize=8)
    ax.invert_yaxis()

    # Column 2: Robust (no per-axes colorbar)
    ax = axes[0, 1]
    sns.heatmap(pv_rob_acc, annot=True, fmt=".2f", cmap=cmap, center=center, vmin=vmin, vmax=vmax,
                cbar=False, ax=ax, annot_kws={"fontsize": 7}, linewidths=0.5, linecolor="white", square=False)
    ax.set_title("Robust Best: Accuracy"); ax.set_xlabel("GC"); ax.set_ylabel("Conservation")
    ax.tick_params(axis='x', labelrotation=45, labelsize=8); ax.tick_params(axis='y', labelsize=8)
    ax.invert_yaxis()

    ax = axes[1, 1]
    sns.heatmap(pv_rob_sauc, annot=True, fmt=".2f", cmap=cmap, center=center, vmin=vmin, vmax=vmax,
                cbar=False, ax=ax, annot_kws={"fontsize": 7}, linewidths=0.5, linecolor="white", square=False)
    ax.set_title("Robust Best: SaliencyAUC"); ax.set_xlabel("GC"); ax.set_ylabel("Conservation")
    ax.tick_params(axis='x', labelrotation=45, labelsize=8); ax.tick_params(axis='y', labelsize=8)
    ax.invert_yaxis()

    # Add a single colorbar per row (shared scale) to preserve equal axis widths
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar1 = fig.colorbar(sm, ax=axes[0, :], fraction=0.046, pad=0.04)
    cbar1.set_label('Accuracy')
    cbar2 = fig.colorbar(sm, ax=axes[1, :], fraction=0.046, pad=0.04)
    cbar2.set_label('SaliencyAUC')

    fig.suptitle("Standard Top1 vs Robust Best (per-dataset)", y=0.995)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved comparison panel to: {out_path}")

    # Additional analysis: Δ heatmaps (Robust - Standard) for Acc and SaAUC
    try:
        delta_acc = pv_rob_acc - pv_std_acc
        delta_sauc = pv_rob_sauc - pv_std_sauc
        # Symmetric color scale around 0
        max_abs = float(np.nanmax(np.abs(pd.concat([delta_acc.stack(), delta_sauc.stack()])))) if not delta_acc.empty else 0.0
        vmax_delta = max(0.05, max_abs)
        vmin_delta = -vmax_delta
        fig2, axes2 = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
        dmap = 'coolwarm'
        ax = axes2[0]
        sns.heatmap(delta_acc, annot=True, fmt="+.2f", cmap=dmap, vmin=vmin_delta, vmax=vmax_delta,
                    center=0.0, cbar=True, cbar_kws={'label': 'ΔAccuracy (Robust-Standard)'}, ax=ax,
                    annot_kws={"fontsize": 7}, linewidths=0.5, linecolor="white")
        ax.set_title("ΔAccuracy"); ax.set_xlabel("GC"); ax.set_ylabel("Conservation")
        ax.tick_params(axis='x', labelrotation=45, labelsize=8); ax.tick_params(axis='y', labelsize=8)
        ax.invert_yaxis()
        ax = axes2[1]
        sns.heatmap(delta_sauc, annot=True, fmt="+.2f", cmap=dmap, vmin=vmin_delta, vmax=vmax_delta,
                    center=0.0, cbar=True, cbar_kws={'label': 'ΔSaliencyAUC (Robust-Standard)'}, ax=ax,
                    annot_kws={"fontsize": 7}, linewidths=0.5, linecolor="white")
        ax.set_title("ΔSaliencyAUC"); ax.set_xlabel("GC"); ax.set_ylabel("Conservation")
        ax.tick_params(axis='x', labelrotation=45, labelsize=8); ax.tick_params(axis='y', labelsize=8)
        ax.invert_yaxis()
        fig2.suptitle("Robust vs Standard: Improvements", y=0.98)
        fig2.tight_layout(rect=[0, 0, 1, 0.95])
        out_delta = os.path.splitext(out_path)[0] + "_delta.png"
        fig2.savefig(out_delta, dpi=300)
        plt.close(fig2)
        print(f"Saved delta panel to: {out_delta}")

        # Print concise improvement stats
        acc_mean_delta = float(np.nanmean(delta_acc.to_numpy()))
        sauc_mean_delta = float(np.nanmean(delta_sauc.to_numpy()))
        acc_pos_frac = float(np.nanmean((delta_acc.to_numpy() > 0).astype(float)))
        sauc_pos_frac = float(np.nanmean((delta_sauc.to_numpy() > 0).astype(float)))
        print(f"Mean ΔAccuracy (Robust-Standard): {acc_mean_delta:+.3f} (improved in {acc_pos_frac*100:.1f}% cells)")
        print(f"Mean ΔSaliencyAUC (Robust-Standard): {sauc_mean_delta:+.3f} (improved in {sauc_pos_frac*100:.1f}% cells)")
    except Exception:
        pass

    # Robust best method heatmap (categorical), annotated with robust SaAUC
    try:
        if 'method' in df_rob.columns and not df_rob.empty:
            pv_method = df_rob.pivot_table(index='cons', columns='gc', values='method', aggfunc=lambda x: x.iloc[0]).sort_index(ascending=True)
            pv_sauc = df_rob.pivot_table(index='cons', columns='gc', values='sauc', aggfunc='mean').sort_index(ascending=True)
            # Align to union axes
            pv_method = pv_method.reindex(index=cons_index, columns=gc_columns)
            pv_sauc = pv_sauc.reindex(index=cons_index, columns=gc_columns)
            method_order = ['RS', 'GS', 'HF-NoSched', 'HF-Sched', 'DHF-NoSched', 'DHF-Sched', 'Unknown']
            method_to_int = {m:i for i,m in enumerate(method_order)}
            pv_int = pv_method.applymap(lambda x: method_to_int.get(x, method_to_int['Unknown']))
            base_cmap = plt.get_cmap('tab10', len(method_order))
            import matplotlib.colors as mcolors
            norm = mcolors.BoundaryNorm(np.arange(len(method_order)+1)-0.5, base_cmap.N)
            fig3, ax3 = plt.subplots(1, 1, figsize=(6.5, 5.5))
            sns.heatmap(pv_int, annot=pv_sauc, fmt='.2f', cmap=base_cmap, vmin=0, vmax=len(method_order)-1,
                        cbar=False, ax=ax3, linewidths=0.5, linecolor='white')
            ax3.set_title('Robust Best Method (annotated by SaAUC)'); ax3.set_xlabel('GC'); ax3.set_ylabel('Conservation')
            ax3.tick_params(axis='x', labelrotation=45, labelsize=8); ax3.tick_params(axis='y', labelsize=8)
            ax3.invert_yaxis()
            # Custom colorbar with labels
            sm2 = plt.cm.ScalarMappable(cmap=base_cmap, norm=norm)
            sm2.set_array([])
            cbar = fig3.colorbar(sm2, ax=ax3, ticks=np.arange(len(method_order)))
            cbar.set_ticklabels(method_order)
            cbar.set_label('Best Robust Method')
            fig3.tight_layout()
            out_method = os.path.splitext(out_path)[0] + "_robust_methods.png"
            fig3.savefig(out_method, dpi=300)
            plt.close(fig3)
            print(f"Saved robust method map to: {out_method}")
    except Exception:
        pass


def _parse_gc_cons_from_key(key: str) -> Tuple[float, float]:
    """Extract (gc, cons) floats from keys like 'val_acc_gc0.650_cons0.70' or 'wiou_gc0.700_cons0.75'."""
    m = re.search(r"_gc([0-9.]+)_cons([0-9.]+)", str(key))
    if not m:
        raise ValueError(f"Cannot parse gc/cons from key: {key}")
    return float(m.group(1)), float(m.group(2))


def _safe_parse_user_attrs(s: str) -> Dict:
    """Parse a 'user_attrs' cell from trials.csv which may be JSON or a python-literal string."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return {}
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return {}


def _load_standard_from_hpo_summary(summary_path: str) -> Dict:
    """
    Build a std_entry-like dict from a unified standard Optuna study summary.json (and trials.csv).
    Requires the study to have been run with eval_mode == 'all' so per-dataset metrics are in user_attrs.
    Returns a dict with keys: rank, trial_number, params, per_dataset (list with gc, cons, agg).
    """
    with open(summary_path, "r") as f:
        s = json.load(f)
    top_list = s.get("top10_by_saliency_auc") or []
    if not top_list:
        raise RuntimeError("No top trials found in standard summary.json")
    top1 = top_list[0]
    trial_number = top1.get("number")
    params = dict(top1.get("params", {}))

    # Optional per-dataset aux metrics directly embedded
    wiou_attrs = dict(top1.get("wiou_attrs", {}))
    snr_attrs = dict(top1.get("snr_attrs", {}))

    # Pull per-dataset Acc/AUC from trials.csv user_attrs
    study_dir = os.path.dirname(summary_path)
    trials_csv = os.path.join(study_dir, "trials.csv")
    acc_map: Dict[Tuple[float, float], float] = {}
    auc_map: Dict[Tuple[float, float], float] = {}
    if os.path.exists(trials_csv):
        try:
            df = pd.read_csv(trials_csv)
            if "number" in df.columns and "user_attrs" in df.columns:
                row = df.loc[df["number"] == trial_number]
                if not row.empty:
                    ua_raw = row.iloc[0]["user_attrs"]
                    ua = _safe_parse_user_attrs(ua_raw)
                    for k, v in ua.items():
                        if isinstance(k, str) and (k.startswith("val_acc_gc") or k.startswith("saliency_auc_gc")):
                            try:
                                gc, cons = _parse_gc_cons_from_key(k)
                                if k.startswith("val_acc_gc"):
                                    acc_map[(gc, cons)] = float(v)
                                else:
                                    auc_map[(gc, cons)] = float(v)
                            except Exception:
                                continue
        except Exception:
            pass

    # Merge into per_dataset list keyed by union of observed keys
    keys_union = set(acc_map.keys()) | set(auc_map.keys())
    # Also include any keys from wiou/snr attrs if present
    for k in list(wiou_attrs.keys()) + list(snr_attrs.keys()):
        try:
            keys_union.add(_parse_gc_cons_from_key(k))
        except Exception:
            continue

    per_dataset: List[Dict[str, object]] = []
    for gc, cons in sorted(keys_union):
        # Map wiou/snr keys can be either 'wiou_' or 'saliency_snr_'/'snr_'
        wi_key = f"wiou_gc{gc:.3f}_cons{cons:.2f}"
        sn_keys = [f"saliency_snr_gc{gc:.3f}_cons{cons:.2f}", f"snr_gc{gc:.3f}_cons{cons:.2f}"]
        wiou_val = wiou_attrs.get(wi_key)
        snr_val = None
        for sk in sn_keys:
            if sk in snr_attrs:
                snr_val = snr_attrs.get(sk)
                break
        agg = {
            "val_acc_mean": float(acc_map.get((gc, cons))) if (gc, cons) in acc_map else float('nan'),
            "saliency_auc_mean": float(auc_map.get((gc, cons))) if (gc, cons) in auc_map else float('nan'),
            "wIoU_mean": float(wiou_val) if wiou_val is not None else float('nan'),
            "saliency_snr_mean": float(snr_val) if snr_val is not None else float('nan'),
        }
        per_dataset.append({"gc": float(gc), "conservation": float(cons), "agg": agg})

    entry = {
        "rank": 1,
        "trial_number": trial_number,
        "params": params,
        "per_dataset": per_dataset,
    }
    return entry


def _maybe_build_std_entry_from_summary_near(refit_dir: str) -> Dict:
    """Try to locate a unified standard study summary.json near refit_dir and build std_entry from it."""
    candidates = [
        os.path.join(refit_dir, "summary.json"),
        os.path.join(os.path.dirname(refit_dir), "summary.json"),
    ]
    for c in candidates:
        try:
            if os.path.exists(c):
                with open(c, "r") as f:
                    s = json.load(f)
                st_name = str(s.get("study_name", ""))
                opt = str(s.get("optimization", ""))
                if opt == "single_objective_unified" and st_name.endswith("_unified_mode_standard"):
                    return _load_standard_from_hpo_summary(c)
        except Exception:
            continue
    # As a last resort, scan optuna_results for a standard unified study
    try:
        root = "optuna_results"
        if os.path.isdir(root):
            for name in os.listdir(root):
                if name.endswith("_unified_mode_standard"):
                    sp = os.path.join(root, name, "summary.json")
                    if os.path.exists(sp):
                        return _load_standard_from_hpo_summary(sp)
    except Exception:
        pass
    raise RuntimeError("Could not locate a standard unified summary.json to build std_entry.")


def _make_standard_vs_robust_wiou_snr_panel(std_entry: Dict, robust_root: str, out_path: str) -> None:
    """
    Build a 2x2 panel comparing Standard Top1 vs Robust best for wIoU and Saliency SNR.
    Row 1: wIoU; Row 2: Saliency SNR. Col 1: Standard; Col 2: Robust.
    """
    # Standard grids
    per_ds = std_entry.get("per_dataset", [])
    if not per_ds:
        print("No per-dataset entries in standard refit entry; skipping wIoU/SNR comparison.")
        return
    recs_std: List[Dict[str, float]] = []
    for d in per_ds:
        gc = float(d.get('gc'))
        cons = float(d.get('conservation'))
        agg = d.get('agg', {})
        wiou = float(agg.get('wIoU_mean', 0.0))
        snr = float(agg.get('saliency_snr_mean', 0.0))
        recs_std.append({"gc": gc, "cons": cons, "wiou": wiou, "snr": snr})
    df_std = pd.DataFrame(recs_std)
    pv_std_wiou = df_std.pivot_table(index="cons", columns="gc", values="wiou", aggfunc="mean").sort_index(ascending=True)
    pv_std_snr = df_std.pivot_table(index="cons", columns="gc", values="snr", aggfunc="mean").sort_index(ascending=True)

    # Robust grid
    df_rob = _load_robust_grid(robust_root)
    if df_rob.empty or 'wiou' not in df_rob.columns or 'snr' not in df_rob.columns:
        print("Robust studies missing wIoU/SNR; skipping wIoU/SNR comparison panel.")
        return
    pv_rob_wiou = df_rob.pivot_table(index="cons", columns="gc", values="wiou", aggfunc="mean").sort_index(ascending=True)
    pv_rob_snr = df_rob.pivot_table(index="cons", columns="gc", values="snr", aggfunc="mean").sort_index(ascending=True)

    # Align axes
    cons_index = sorted(set(pv_std_wiou.index.tolist()) | set(pv_rob_wiou.index.tolist()))
    gc_columns = sorted(set(pv_std_wiou.columns.tolist()) | set(pv_rob_wiou.columns.tolist()))
    pv_std_wiou = pv_std_wiou.reindex(index=cons_index, columns=gc_columns)
    pv_std_snr = pv_std_snr.reindex(index=cons_index, columns=gc_columns)
    pv_rob_wiou = pv_rob_wiou.reindex(index=cons_index, columns=gc_columns)
    pv_rob_snr = pv_rob_snr.reindex(index=cons_index, columns=gc_columns)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharey=True)
    cmap = 'coolwarm'; vmin, vmax, center = 0.0, 1.0, 0.5

    # Standard wIoU
    ax = axes[0, 0]
    sns.heatmap(pv_std_wiou, annot=True, fmt=".2f", cmap=cmap, center=center, vmin=vmin, vmax=vmax,
                cbar=False, ax=ax, annot_kws={"fontsize": 7}, linewidths=0.5, linecolor="white", square=True)
    ax.set_title("Standard Top1: wIoU"); ax.set_xlabel("GC"); ax.set_ylabel("Conservation")
    ax.tick_params(axis='x', labelrotation=45, labelsize=8); ax.tick_params(axis='y', labelsize=8)
    ax.invert_yaxis()

    # Standard SNR
    ax = axes[1, 0]
    sns.heatmap(pv_std_snr, annot=True, fmt=".2f", cmap=cmap, center=center, vmin=vmin, vmax=vmax,
                cbar=False, ax=ax, annot_kws={"fontsize": 7}, linewidths=0.5, linecolor="white", square=True)
    ax.set_title("Standard Top1: Saliency SNR"); ax.set_xlabel("GC"); ax.set_ylabel("Conservation")
    ax.tick_params(axis='x', labelrotation=45, labelsize=8); ax.tick_params(axis='y', labelsize=8)
    ax.invert_yaxis()

    # Robust wIoU
    ax = axes[0, 1]
    sns.heatmap(pv_rob_wiou, annot=True, fmt=".2f", cmap=cmap, center=center, vmin=vmin, vmax=vmax,
                cbar=False, ax=ax, annot_kws={"fontsize": 7}, linewidths=0.5, linecolor="white", square=True)
    ax.set_title("Robust Best: wIoU"); ax.set_xlabel("GC"); ax.set_ylabel("Conservation")
    ax.tick_params(axis='x', labelrotation=45, labelsize=8); ax.tick_params(axis='y', labelsize=8)
    ax.invert_yaxis()

    # Robust SNR
    ax = axes[1, 1]
    sns.heatmap(pv_rob_snr, annot=True, fmt=".2f", cmap=cmap, center=center, vmin=vmin, vmax=vmax,
                cbar=False, ax=ax, annot_kws={"fontsize": 7}, linewidths=0.5, linecolor="white", square=True)
    ax.set_title("Robust Best: Saliency SNR"); ax.set_xlabel("GC"); ax.set_ylabel("Conservation")
    ax.tick_params(axis='x', labelrotation=45, labelsize=8); ax.tick_params(axis='y', labelsize=8)
    ax.invert_yaxis()

    # Shared colorbars per row
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar1 = fig.colorbar(sm, ax=axes[0, :], fraction=0.046, pad=0.04)
    cbar1.set_label('wIoU')
    cbar2 = fig.colorbar(sm, ax=axes[1, :], fraction=0.046, pad=0.04)
    cbar2.set_label('Saliency SNR')

    fig.suptitle("Standard Top1 vs Robust Best: wIoU & SNR", y=0.995)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_png = os.path.splitext(out_path)[0] + "_wiou_snr_compare.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved wIoU/SNR comparison panel to: {out_png}")


def _make_standard_wiou_snr(std_entry: Dict, out_path: str) -> None:
    """
    Build a 1x2 panel for Standard Top1 showing wIoU and SaliencySNR.
    """
    per_ds = std_entry.get("per_dataset", [])
    if not per_ds:
        return
    recs: List[Dict[str, float]] = []
    for d in per_ds:
        gc = float(d.get('gc'))
        cons = float(d.get('conservation'))
        agg = d.get('agg', {})
        wiou = float(agg.get('wIoU_mean', 0.0))
        snr = float(agg.get('saliency_snr_mean', 0.0))
        recs.append({"gc": gc, "cons": cons, "wiou": wiou, "snr": snr})
    df = pd.DataFrame(recs)
    pv_wiou = df.pivot_table(index="cons", columns="gc", values="wiou", aggfunc="mean").sort_index(ascending=True)
    pv_snr = df.pivot_table(index="cons", columns="gc", values="snr", aggfunc="mean").sort_index(ascending=True)

    cons_index = sorted(set(pv_wiou.index.tolist()) | set(pv_snr.index.tolist()))
    gc_columns = sorted(set(pv_wiou.columns.tolist()) | set(pv_snr.columns.tolist()))
    pv_wiou = pv_wiou.reindex(index=cons_index, columns=gc_columns)
    pv_snr = pv_snr.reindex(index=cons_index, columns=gc_columns)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    cmap = 'coolwarm'; vmin, vmax, center = 0.0, 1.0, 0.5

    ax = axes[0]
    sns.heatmap(pv_wiou, annot=True, fmt=".2f", cmap=cmap, center=center, vmin=vmin, vmax=vmax,
                cbar=True, cbar_kws={'label': 'wIoU'}, ax=ax, annot_kws={"fontsize": 7},
                linewidths=0.5, linecolor="white")
    ax.set_title("Standard Top1: wIoU"); ax.set_xlabel("GC"); ax.set_ylabel("Conservation")
    ax.tick_params(axis='x', labelrotation=45, labelsize=8); ax.tick_params(axis='y', labelsize=8)
    ax.invert_yaxis()

    ax = axes[1]
    sns.heatmap(pv_snr, annot=True, fmt=".2f", cmap=cmap, center=center, vmin=vmin, vmax=vmax,
                cbar=True, cbar_kws={'label': 'Saliency SNR'}, ax=ax, annot_kws={"fontsize": 7},
                linewidths=0.5, linecolor="white")
    ax.set_title("Standard Top1: Saliency SNR"); ax.set_xlabel("GC"); ax.set_ylabel("Conservation")
    ax.tick_params(axis='x', labelrotation=45, labelsize=8); ax.tick_params(axis='y', labelsize=8)
    ax.invert_yaxis()

    fig.suptitle("Standard Top1: wIoU and Saliency SNR", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_aux = os.path.splitext(out_path)[0] + "_wiou_snr.png"
    fig.savefig(out_aux, dpi=300)
    plt.close(fig)
    print(f"Saved wIoU/SNR panel to: {out_aux}")


def main():
    parser = argparse.ArgumentParser(description="Analyze refit results: print HP table and plot 2x3 heatmaps.")
    parser.add_argument("--refit_dir", type=str, required=True, help="Directory containing refit_summary.json")
    parser.add_argument("--top_k", type=int, default=3, help="How many top models to include (max 3 visualized)")
    parser.add_argument("--out_png", type=str, default=None, help="Path to save the heatmap PNG (default: <refit_dir>/refit_panel_top3.png)")
    parser.add_argument("--compare-robust", action="store_true", help="Also build a Standard vs Robust comparison panel.")
    parser.add_argument("--robust_root", type=str, default="optuna_results", help="Root directory containing robust per-dataset studies.")
    parser.add_argument("--out_png_compare", type=str, default=None, help="Path to save the Standard-vs-Robust PNG (default: <refit_dir>/refit_vs_robust.png)")
    args = parser.parse_args()

    # Try to load refit; if not present, continue (for compare-robust fallback path)
    results: List[Dict] = []
    has_refit = True
    try:
        data = _load_refit_summary(args.refit_dir)
        results = list(data.get("results", []))
        if not results:
            has_refit = False
            print("No results found in refit_summary.json; skipping refit-dependent outputs.")
    except FileNotFoundError:
        has_refit = False
        print(f"refit_summary.json not found under {args.refit_dir}; proceeding without refit.")

    if has_refit and results:
        # 1) Print well-formatted HP table for top-K
        df_hp = _format_hp_table_matrix(results, args.top_k)
        print("\nChosen hyperparameters (rows) by model (columns):\n")
        print(df_hp.to_string())
        print("")

        # 2) Heatmaps panel (2 rows x up to 3 columns)
        out_png = args.out_png or os.path.join(args.refit_dir, "refit_panel_top3.png")
        _make_heatmaps(results, args.top_k, out_png)

    # 3) Optional: Standard vs Robust comparison (Top1 standard only)
    if args.compare_robust:
        # Prefer refit Top1 if it contains per-dataset entries; otherwise, derive from unified standard summary.json
        std_entry = results[0] if (has_refit and results) else {}
        per_ds = std_entry.get("per_dataset", []) if isinstance(std_entry, dict) else []
        if not per_ds:
            try:
                std_entry = _maybe_build_std_entry_from_summary_near(args.refit_dir)
            except Exception as e:
                print(f"Failed to build standard entry from summary.json: {e}")
                std_entry = {}

        out_png_cmp = args.out_png_compare or os.path.join(args.refit_dir, "refit_vs_robust.png")
        if std_entry and std_entry.get("per_dataset"):
            _make_standard_vs_robust_panel(std_entry, args.robust_root, out_png_cmp)
            # Also emit Standard wIoU/SNR if present
            _make_standard_wiou_snr(std_entry, out_png_cmp)
            # And Standard vs Robust wIoU/SNR comparison (robust must have new metrics persisted)
            _make_standard_vs_robust_wiou_snr_panel(std_entry, args.robust_root, out_png_cmp)
        else:
            print("No standard per-dataset data available for comparison.")


if __name__ == "__main__":
    main()


