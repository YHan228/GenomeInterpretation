import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import optuna  # optional; used to fetch Pareto sets directly from storage
except Exception:
    optuna = None


@dataclass
class StudyInfo:
    study_name: str
    gc: float
    cons: float
    mode: str
    path: str


def find_studies(results_root: str) -> List[StudyInfo]:
    studies: List[StudyInfo] = []
    if not os.path.isdir(results_root):
        return studies
    for name in sorted(os.listdir(results_root)):
        study_dir = os.path.join(results_root, name)
        if not os.path.isdir(study_dir):
            continue
        summary_path = os.path.join(study_dir, "summary.json")
        if not os.path.exists(summary_path):
            continue
        try:
            with open(summary_path, "r") as f:
                s = json.load(f)
            study_name = s.get("study_name", name)
            ds = s.get("dataset", {})
            gc = float(ds.get("gc_pos")) if ds.get("gc_pos") is not None else np.nan
            cons = float(ds.get("conservation")) if ds.get("conservation") is not None else np.nan
            # Parse mode from study_name suffix if not present
            mode = "standard"
            if "_mode_" in study_name:
                try:
                    mode = study_name.split("_mode_")[-1]
                except Exception:
                    mode = "standard"
            studies.append(StudyInfo(study_name=study_name, gc=gc, cons=cons, mode=mode, path=study_dir))
        except Exception:
            continue
    return studies


def load_trials(study: StudyInfo) -> pd.DataFrame:
    trials_path = os.path.join(study.path, "trials.csv")
    if not os.path.exists(trials_path):
        return pd.DataFrame()
    df = pd.read_csv(trials_path)
    # Normalize columns for objectives (val_acc, sal_auc)
    acc_col, auc_col = None, None
    for c in df.columns:
        if c.lower().startswith("values_0"):
            acc_col = c
        if c.lower().startswith("values_1"):
            auc_col = c
    # Fallbacks
    if acc_col is None and "val_acc" in df.columns:
        acc_col = "val_acc"
    if auc_col is None and "saliency_auc" in df.columns:
        auc_col = "saliency_auc"
    # If only scalar objective exists, assume it is saliency_auc (per code for TPE)
    if auc_col is None and "value" in df.columns and study.mode != "standard":
        auc_col = "value"
    df = df.copy()
    if acc_col is not None:
        df["val_acc"] = df[acc_col]
    if auc_col is not None:
        df["saliency_auc"] = df[auc_col]
    return df


def select_best_trial(df: pd.DataFrame, mode: str) -> Optional[pd.Series]:
    """Select best trial.
    - Standard: treat round(acc,2) >= 0.99 as equal, then maximize SaAUC.
    - Robust: allow ties at acc >= 0.975, then maximize SaAUC.
    Fallback: maximize acc, tie-break by SaAUC.
    """
    df_ok = df[(df.get("state") == "COMPLETE") & df["val_acc"].notna() & df["saliency_auc"].notna()].copy()
    if df_ok.empty:
        return None
    if mode == 'robust':
        eligible = df_ok[df_ok["val_acc"] >= 0.95]
    else:
        acc_round = df_ok["val_acc"].round(2)
        eligible = df_ok[acc_round >= 0.995]
    if not eligible.empty:
        return eligible.sort_values(["saliency_auc", "val_acc"], ascending=[False, False]).iloc[0]
    return df_ok.sort_values(["val_acc", "saliency_auc"], ascending=[False, False]).iloc[0]


def visualize_pareto(*args, **kwargs):
    return


def collect_hp_frequencies(best_trials: List[pd.Series]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for row in best_trials:
        if row is None:
            continue
        # Optuna trials_dataframe stores params as separate columns: params_<name>
        params: Dict[str, object] = {}
        for col, val in row.items():
            if isinstance(col, str) and col.startswith("params_"):
                hp_name = col[len("params_"):]
                params[hp_name] = val
        for k, v in params.items():
            # Coerce to short string bins for readability
            if isinstance(v, float):
                v_str = f"{v:.3g}"
            else:
                v_str = str(v)
            counts.setdefault(k, {}).setdefault(v_str, 0)
            counts[k][v_str] += 1
    return counts


def plot_hp_frequencies(counts: Dict[str, Dict[str, int]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for hp, bucket in counts.items():
        data = sorted(bucket.items(), key=lambda kv: (-kv[1], kv[0]))
        labels = [k for k, _ in data]
        vals = [v for _, v in data]
        plt.figure(figsize=(max(6, 0.5 * len(labels)), 3.5))
        sns.barplot(x=labels, y=vals, color="C0")
        plt.title(f"Best-trial frequency: {hp}")
        plt.ylabel("Count")
        plt.xlabel(hp)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"best_freq_{hp}.png"), dpi=300, format='png')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze Optuna multi-study results.")
    parser.add_argument("--results_root", type=str, default="optuna_results", help="Root directory containing per-study folders.")
    parser.add_argument("--out_dir", type=str, default=None, help="Aggregate output directory (default: <results_root>/_aggregate)")
    parser.add_argument("--acc_threshold", type=float, default=0.995, help="Accuracy threshold for model selection.")
    args = parser.parse_args()

    agg_dir = args.out_dir or os.path.join(args.results_root, "_aggregate")
    os.makedirs(agg_dir, exist_ok=True)

    studies = find_studies(args.results_root)
    summary_rows: List[Dict[str, object]] = []
    best_trials: List[pd.Series] = []
    best_trials_modes: List[str] = []
    best_trials_gc: List[Tuple[float, float]] = []
    all_trials_list: List[pd.DataFrame] = []

    # Grid summary for heatmap of best saliency AUC meeting acc threshold
    heatmap_records_std: List[Dict[str, object]] = []
    heatmap_records_rob: List[Dict[str, object]] = []

    for st in studies:
        df = load_trials(st)
        if df.empty or "val_acc" not in df.columns or "saliency_auc" not in df.columns:
            continue

        # Keep all COMPLETE trials with metadata for HP analysis
        df_all = df.copy()
        df_all["study"] = st.study_name
        df_all["gc"] = st.gc
        df_all["cons"] = st.cons
        df_all = df_all[df_all.get("state") == "COMPLETE"]
        all_trials_list.append(df_all)

        # (Pareto visualization removed)

        # Selection per your rule
        best = select_best_trial(df, st.mode)
        best_trials.append(best)
        best_trials_modes.append(st.mode)
        best_trials_gc.append((st.gc, st.cons))
        if best is not None:
            summary_rows.append({
                "study": st.study_name,
                "gc": st.gc,
                "cons": st.cons,
                "mode": st.mode,
                "trial_number": int(best.get("number")) if pd.notna(best.get("number")) else None,
                "val_acc": float(best["val_acc"]),
                "saliency_auc": float(best["saliency_auc"]),
                "meets_acc_threshold": bool(best["val_acc"] >= args.acc_threshold),
                "params": best.get("params"),
            })
            rec = {"gc": st.gc, "cons": st.cons, "saliency_auc": float(best["saliency_auc"]), "val_acc": float(best["val_acc"]) }
            if st.mode == "standard":
                heatmap_records_std.append(rec)
            elif st.mode == "robust":
                heatmap_records_rob.append(rec)

    # Save selection summary
    if summary_rows:
        sel_df = pd.DataFrame(summary_rows)
        sel_df.sort_values(["gc", "cons"], inplace=True)
        sel_df.to_csv(os.path.join(agg_dir, "best_selection_by_dataset.csv"), index=False)

    # Heatmaps (GC x Cons): side-by-side Standard vs Robust for SaliencyAUC and Accuracy
    if heatmap_records_std or heatmap_records_rob:
        def _make_pivot(records: List[Dict[str, object]], value: str) -> Optional[pd.DataFrame]:
            if not records:
                return None
            dfh = pd.DataFrame(records)
            pv = dfh.pivot_table(index="cons", columns="gc", values=value, aggfunc=np.mean)
            return pv.sort_index(ascending=True)

        for value, fname, cbar_label, center, vmin, vmax in [
            ("saliency_auc", "selected_sauc_heatmaps_std_robust.png", 'SaliencyAUC', 0.5, 0, 1),
            ("val_acc", "selected_accuracy_heatmaps_std_robust.png", 'Accuracy', 0.5, 0, 1),
        ]:
            pv_std = _make_pivot(heatmap_records_std, value)
            pv_rob = _make_pivot(heatmap_records_rob, value)
            # Match eval_toy aesthetics: fixed overall figure, side-by-side, sharey
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
            titles = ["Standard", "Robust"]
            pivots = [pv_std, pv_rob]
            for ax, title, pv in zip(axes, titles, pivots):
                if pv is None or pv.size == 0:
                    ax.axis('off')
                    ax.set_title(f"{title}: no data")
                    continue
                hm = sns.heatmap(pv, annot=True, fmt=".2f", cmap='coolwarm', center=center, vmin=vmin, vmax=vmax, cbar_kws={'label': cbar_label}, ax=ax)
                ax.set_title(title)
                ax.set_xlabel("GC Content (Confounder Strength)")
                ax.set_ylabel("Conservation (Signal Strength)")
                ax.invert_yaxis()
            fig.suptitle("Selected Models (Left: Standard, Right: Robust)")
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            fig.savefig(os.path.join(agg_dir, fname), dpi=300)
            plt.close(fig)

        # Best robust method heatmap (categorical), annotate with SaAUC
        if heatmap_records_rob:
            rdf = pd.DataFrame(heatmap_records_rob)
            # For categorical method, recompute from robust best trials collected above
            # Build method records from best_trials for robust mode
            method_rows = []
            for bt, mode in zip(best_trials, best_trials_modes):
                if bt is None or mode != 'robust':
                    continue
                # Extract regime/schedule from params_*
                params = {col[len('params_'):]: val for col, val in bt.items() if isinstance(col, str) and col.startswith('params_')}
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
                # Need gc/cons for this best trial: derive from study name already in summary_rows
                # Fallback: cannot easily map here; instead reconstruct from study entries
                # We can get gc/cons from the summary_rows we built
            # Build a quick map from study -> (gc,cons)
            study_to_gccons = {}
            for row in summary_rows:
                study_to_gccons[row['study']] = (row['gc'], row['cons'])
            robust_best = []
            for (st, bt, mode) in zip(studies, best_trials, best_trials_modes):
                if bt is None or mode != 'robust':
                    continue
                params = {col[len('params_'):]: val for col, val in bt.items() if isinstance(col, str) and col.startswith('params_')}
                regime = str(params.get('regime', 'unknown'))
                schedule = str(params.get('schedule', 'off'))
                if regime in ['random_smoothing', 'gaussian_smoothing']:
                    schedule = 'off'
                if regime == 'hotflip':
                    method = 'HF-NoSched' if schedule == 'off' else 'HF-Sched'
                elif regime == 'direct_hotflip':
                    method = 'DHF-NoSched' if schedule == 'off' else 'DHF-Sched'
                elif regime == 'random_smoothing':
                    method = 'RS'
                elif regime == 'gaussian_smoothing':
                    method = 'GS'
                else:
                    method = 'Unknown'
                robust_best.append({
                    'gc': st.gc,
                    'cons': st.cons,
                    'method': method,
                    'sauc': float(bt['saliency_auc']) if 'saliency_auc' in bt else np.nan,
                })
            if robust_best:
                rbdf = pd.DataFrame(robust_best)
                pv_auc = rbdf.pivot_table(index='cons', columns='gc', values='sauc', aggfunc=np.mean).sort_index(ascending=True)
                pv_method = rbdf.pivot_table(index='cons', columns='gc', values='method', aggfunc=lambda x: x.iloc[0]).sort_index(ascending=True)
                # Map methods to integers for coloring
                method_order = ['RS', 'GS', 'HF-NoSched', 'HF-Sched', 'DHF-NoSched', 'DHF-Sched']
                pv_int = pv_method.applymap(lambda x: method_order.index(x) if x in method_order else -1)
                base_cmap = plt.get_cmap('tab10', len(method_order))
                fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))
                hm = sns.heatmap(pv_int, annot=pv_auc, fmt='.2f', cmap=base_cmap, vmin=0, vmax=len(method_order)-1, cbar=False, ax=ax)
                # Custom colorbar with method labels
                norm = plt.cm.colors.BoundaryNorm(np.arange(len(method_order)+1)-0.5, base_cmap.N)
                sm = plt.cm.ScalarMappable(cmap=base_cmap, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, ticks=np.arange(len(method_order)))
                cbar.set_ticklabels(method_order)
                cbar.set_label('Best Robust Method')
                ax.set_title('Best Robust Method (annotated by SaAUC)')
                ax.set_xlabel('GC Content (Confounder Strength)')
                ax.set_ylabel('Conservation (Signal Strength)')
                ax.invert_yaxis()
                fig.tight_layout()
                fig.savefig(os.path.join(agg_dir, 'best_robust_method_heatmap.png'), dpi=300)
                plt.close(fig)

    # Global Pareto step curves (survival-like) per dataset, derived only from
    # the selected Pareto fronts stored in each study's summary.json
    # Prefer pulling nondominated trials directly from Optuna storage when available.
    def _resolve_storage_url() -> Optional[str]:
        storage = os.environ.get("OPTUNA_STORAGE")
        if not storage:
            p = os.path.expanduser("~/.optuna_storage_url")
            if os.path.exists(p):
                try:
                    with open(p, "r") as f:
                        storage = f.read().strip()
                except Exception:
                    storage = None
        return storage

    def _compute_step_from_points(records: List[Dict[str, object]]) -> List[Tuple[float, float, np.ndarray, np.ndarray]]:
        curves: List[Tuple[float, float, np.ndarray, np.ndarray]] = []
        for rec in records:
            pts: List[Tuple[float, float]] = rec.get("points", [])  # (sauc, acc)
            if not pts:
                continue
            pts_sorted = sorted([(float(x), float(y)) for (x, y) in pts if x is not None and y is not None], key=lambda t: t[0])
            if not pts_sorted:
                continue
            # Unique x with best y at that x
            xs = np.array([p[0] for p in pts_sorted], dtype=float)
            ys = np.array([p[1] for p in pts_sorted], dtype=float)
            # Collapse duplicate x by taking max y
            if xs.size > 1 and np.any(np.diff(xs) == 0):
                df_tmp = pd.DataFrame({"x": xs, "y": ys}).groupby("x", as_index=False)["y"].max()
                xs = df_tmp["x"].to_numpy()
                ys = df_tmp["y"].to_numpy()
            # For a maximization problem, the step function for “best achievable y at x-threshold”
            # should be nonincreasing with x. Compute running max from right to left.
            y_rev = ys[::-1]
            y_rev_max = np.maximum.accumulate(y_rev)
            y_step = y_rev_max[::-1]
            x = xs
            y = y_step
            curves.append((float(rec["gc"]), float(rec["cons"]), x, y))
        return curves

    # Collect Pareto points per study (DB first, fallback to summary.json)
    pareto_std: List[Dict[str, object]] = []
    pareto_rob: List[Dict[str, object]] = []
    storage_url = _resolve_storage_url()
    for st in studies:
        points: List[Tuple[float, float]] = []
        used_db = False
        # Try DB first for true nondominated set
        if optuna is not None and storage_url:
            try:
                study_obj = optuna.load_study(study_name=st.study_name, storage=storage_url)
                print(f"loaded study {st.study_name} from {storage_url}")
                if len(study_obj.directions) == 2:
                    for t in study_obj.best_trials:
                        va = None
                        sa = None
                        if t.values is not None:
                            if len(t.values) >= 2:
                                va, sa = t.values[0], t.values[1]
                        if va is None or sa is None:
                            va = t.user_attrs.get('val_acc')
                            sa = t.user_attrs.get('saliency_auc')
                        if va is None or sa is None:
                            continue
                        points.append((float(sa), float(va)))
                    used_db = True
            except Exception:
                used_db = False
        # Fallback to summary.json
        if not used_db:
            print(f"Warning: No trials found in DB for study {st.study_name}")
            try:
                summary_path = os.path.join(st.path, "summary.json")
                with open(summary_path, "r") as f:
                    s = json.load(f)
                front = s.get("pareto_front", []) or []
                for item in front:
                    va = item.get("val_acc")
                    sa = item.get("saliency_auc")
                    if va is None or sa is None:
                        continue
                    points.append((float(sa), float(va)))
            except Exception:
                points = []
        if not points:
            continue
        rec = {"gc": st.gc, "cons": st.cons, "points": points}
        if st.mode == "standard":
            pareto_std.append(rec)
        elif st.mode == "robust":
            pareto_rob.append(rec)

    std_curves = _compute_step_from_points(pareto_std)
    rob_curves = _compute_step_from_points(pareto_rob)

    if std_curves:
        plt.figure(figsize=(7, 7))
        for gc_val, cons_val, x, y in std_curves:
            plt.step(x, y, where='post', alpha=0.45, linewidth=1.5)
        plt.xlabel("Saliency AUC")
        plt.ylabel("Validation Accuracy")
        plt.title("Pareto Step Curves per Dataset (Standard)")
        plt.xlim(0.5, 1.01); plt.ylim(0.5, 1.01)
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(agg_dir, "pareto_step_curves_standard.png"), dpi=300)
        plt.close()

    if rob_curves:
        plt.figure(figsize=(7, 7))
        for gc_val, cons_val, x, y in rob_curves:
            plt.step(x, y, where='post', alpha=0.45, linewidth=1.5)
        plt.xlabel("Saliency AUC")
        plt.ylabel("Validation Accuracy")
        plt.title("Pareto Step Curves per Dataset (Robust)")
        plt.xlim(0.5, 1.1); plt.ylim(0.5, 1.1)
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(agg_dir, "pareto_step_curves_robust.png"), dpi=300)
        plt.close()

    # HP frequencies among the selected best trials
    counts = collect_hp_frequencies([bt for bt in best_trials if bt is not None])
    if counts:
        with open(os.path.join(agg_dir, "best_hp_frequency.json"), "w") as f:
            json.dump(counts, f, indent=2)
        plot_hp_frequencies(counts, os.path.join(agg_dir, "hp_frequency_plots"))

    # Majority-vote model detection (standard only), based on categorical HPs
    # Ignore continuous vars (lr, weight_decay, dropouts, etc.)
    try:
        cat_keys = [
            "k1", "k2", "k3", "c1", "c2", "c3", "pool_w", "act1",
            "optimizer", "train_batch_size", "grad_clip"
        ]
        combo_counts: Dict[Tuple, int] = {}
        combo_datasets: Dict[Tuple, List[Tuple[float, float]]] = {}
        total_std = 0
        for bt, mode, gc_cons in zip(best_trials, best_trials_modes, best_trials_gc):
            if bt is None or mode != "standard":
                continue
            total_std += 1
            params = {}
            for col, val in bt.items():
                if isinstance(col, str) and col.startswith("params_"):
                    hp_name = col[len("params_"):]
                    params[hp_name] = val
            # Build combo from categorical keys; missing keys get None
            combo = tuple(params.get(k) for k in cat_keys)
            combo_counts[combo] = combo_counts.get(combo, 0) + 1
            combo_datasets.setdefault(combo, []).append(gc_cons)
        if total_std > 0 and combo_counts:
            # Sort by frequency
            sorted_combos = sorted(combo_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            top_combo, top_count = sorted_combos[0]
            top_fraction = top_count / total_std
            # Save CSV and JSON
            rows = []
            for combo, cnt in sorted_combos:
                row = {"count": cnt, "fraction": cnt / total_std}
                for k, v in zip(cat_keys, combo):
                    row[k] = v
                # Attach datasets (gc:cons) where this combo was selected
                ds_list = combo_datasets.get(combo, [])
                row["datasets"] = ";".join([f"{float(g):.3f}:{float(c):.2f}" for g, c in ds_list])
                rows.append(row)
            df_rows = pd.DataFrame(rows)
            # Reorder columns: datasets first, then count/fraction, then HPs
            ordered_cols = ["datasets", "count", "fraction"] + cat_keys
            for c in ordered_cols:
                if c not in df_rows.columns:
                    df_rows[c] = None
            df_rows = df_rows[ordered_cols]
            # Rank rows by datasets (lexicographic), so ordering follows dataset groupings
            df_rows.sort_values(["datasets", "count"], ascending=[True, False], inplace=True)
            df_rows.to_csv(os.path.join(agg_dir, "standard_majority_models.csv"), index=False)

            # Also emit a per-dataset table (one row per dataset) ranked by data
            per_ds_rows = []
            for combo, ds_list in combo_datasets.items():
                for g, c in ds_list:
                    row = {"gc": float(g), "cons": float(c)}
                    for k, v in zip(cat_keys, combo):
                        row[k] = v
                    per_ds_rows.append(row)
            if per_ds_rows:
                df_per = pd.DataFrame(per_ds_rows)
                df_per.sort_values(["gc", "cons"], inplace=True)
                df_per.to_csv(os.path.join(agg_dir, "standard_majority_by_dataset.csv"), index=False)
            with open(os.path.join(agg_dir, "standard_majority_models.json"), "w") as f:
                json.dump({
                    "total_standard": total_std,
                    "top_count": top_count,
                    "top_fraction": top_fraction,
                    "categorical_keys": cat_keys,
                    "top_combo": {k: v for k, v in zip(cat_keys, top_combo)},
                    "top_datasets": [
                        {"gc": float(g), "cons": float(c)} for (g, c) in combo_datasets.get(top_combo, [])
                    ],
                }, f, indent=2)
    except Exception:
        pass

    # Concise continuous-HP analysis: one combined density panel + small correlation heatmap
    if all_trials_list:
        all_df = pd.concat(all_trials_list, ignore_index=True)

        # Global scatter: all trials (x=SaAUC, y=Acc)
        try:
            df_sc = all_df[["saliency_auc", "val_acc"]].dropna()
            if not df_sc.empty:
                plt.figure(figsize=(6, 5))
                sns.scatterplot(data=df_sc, x="saliency_auc", y="val_acc", s=8, alpha=0.25, edgecolor="none")
                plt.xlabel("Saliency AUC")
                plt.ylabel("Validation Accuracy")
                plt.title("All Trials: Acc vs SaAUC")
                plt.tight_layout()
                plt.savefig(os.path.join(agg_dir, "all_trials_acc_vs_sauc.png"), dpi=300)
                plt.close()
        except Exception:
            pass
        sel_df_rows = [bt.to_frame().T for bt in best_trials if bt is not None]
        selected_df = pd.concat(sel_df_rows, ignore_index=True) if sel_df_rows else pd.DataFrame()

        continuous_hps = [
            "lr", "weight_decay", "drop_conv1", "drop_conv2", "drop_conv3", "drop_fc"
        ]

        # Combined density (ridge-like) plot: rows = HPs; columns overlaid: All vs Selected
        n_rows = len(continuous_hps)
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, 1.8 * n_rows + 0.8), sharex=False)
        if n_rows == 1:
            axes = [axes]
        any_plotted = False
        for i, hp in enumerate(continuous_hps):
            ax = axes[i]
            param_col = f"params_{hp}"
            if param_col not in all_df.columns:
                ax.set_visible(False)
                continue
            vals_all = pd.to_numeric(all_df[param_col], errors="coerce").dropna()
            # Selected may not have this column
            vals_sel = pd.Series(dtype=float)
            if not selected_df.empty and param_col in selected_df.columns:
                vals_sel = pd.to_numeric(selected_df[param_col], errors="coerce").dropna()
            # Plot densities if data exists
            if not vals_all.empty:
                sns.kdeplot(x=vals_all, fill=True, alpha=0.25, color="C0", label="All COMPLETE", ax=ax)
                any_plotted = True
            if not vals_sel.empty:
                sns.kdeplot(x=vals_sel, fill=True, alpha=0.35, color="C3", label="Selected", ax=ax)
                any_plotted = True
            ax.set_ylabel("")
            ax.set_yticks([])
            ax.set_title(hp)
        if any_plotted:
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=2)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(agg_dir, "continuous_hp_density_panel.png"), dpi=300)
            plt.close(fig)

        # Correlation heatmap (Spearman) of continuous HPs vs metrics (val_acc, saliency_auc)
        try:
            from scipy.stats import spearmanr
            corr_mat = np.full((2, len(continuous_hps)), np.nan, dtype=float)
            for j, hp in enumerate(continuous_hps):
                param_col = f"params_{hp}"
                if param_col not in all_df.columns:
                    continue
                x = pd.to_numeric(all_df[param_col], errors="coerce")
                y_auc = pd.to_numeric(all_df.get("saliency_auc"), errors="coerce")
                y_acc = pd.to_numeric(all_df.get("val_acc"), errors="coerce")
                m_auc = x.notna() & y_auc.notna()
                m_acc = x.notna() & y_acc.notna()
                if m_auc.sum() > 3:
                    r_auc, _ = spearmanr(x[m_auc], y_auc[m_auc], nan_policy='omit')
                    corr_mat[0, j] = float(r_auc)
                if m_acc.sum() > 3:
                    r_acc, _ = spearmanr(x[m_acc], y_acc[m_acc], nan_policy='omit')
                    corr_mat[1, j] = float(r_acc)
            corr_df = pd.DataFrame(corr_mat, index=["saliency_auc", "val_acc"], columns=continuous_hps)
            corr_df.to_csv(os.path.join(agg_dir, "hp_spearman_correlation.csv"))
            plt.figure(figsize=(1.6 * len(continuous_hps) + 2, 3.5))
            sns.heatmap(corr_df, annot=True, fmt=".2f", cmap='coolwarm', center=0, vmin=-1, vmax=1, cbar_kws={'label': 'Spearman r'})
            plt.title("Continuous HP vs Metric (Spearman)")
            plt.xlabel("Hyperparameter")
            plt.ylabel("Metric")
            plt.tight_layout()
            plt.savefig(os.path.join(agg_dir, "hp_metric_correlation_heatmap.png"), dpi=300)
            plt.close()
        except Exception:
            pass

        # Correlation heatmap (Spearman) of continuous HPs vs dataset factors (gc, cons)
        try:
            from scipy.stats import spearmanr
            corr_gc_cons = np.full((2, len(continuous_hps)), np.nan, dtype=float)
            y_gc = pd.to_numeric(all_df.get("gc"), errors="coerce")
            y_cons = pd.to_numeric(all_df.get("cons"), errors="coerce")
            for j, hp in enumerate(continuous_hps):
                param_col = f"params_{hp}"
                if param_col not in all_df.columns:
                    continue
                x = pd.to_numeric(all_df[param_col], errors="coerce")
                m_gc = x.notna() & y_gc.notna()
                m_cons = x.notna() & y_cons.notna()
                if m_gc.sum() > 3:
                    r_gc, _ = spearmanr(x[m_gc], y_gc[m_gc], nan_policy='omit')
                    corr_gc_cons[0, j] = float(r_gc)
                if m_cons.sum() > 3:
                    r_cons, _ = spearmanr(x[m_cons], y_cons[m_cons], nan_policy='omit')
                    corr_gc_cons[1, j] = float(r_cons)
            corr_df2 = pd.DataFrame(corr_gc_cons, index=["gc", "cons"], columns=continuous_hps)
            corr_df2.to_csv(os.path.join(agg_dir, "hp_spearman_vs_gc_cons.csv"))
            plt.figure(figsize=(1.6 * len(continuous_hps) + 2, 3.5))
            sns.heatmap(corr_df2, annot=True, fmt=".2f", cmap='coolwarm', center=0, vmin=-1, vmax=1, cbar_kws={'label': 'Spearman r'})
            plt.title("Continuous HP vs Dataset (gc, cons)")
            plt.xlabel("Hyperparameter")
            plt.ylabel("Dataset factor")
            plt.tight_layout()
            plt.savefig(os.path.join(agg_dir, "hp_vs_gc_cons_correlation_heatmap.png"), dpi=300)
            plt.close()
        except Exception:
            pass

    print(f"Analysis complete. Outputs in: {agg_dir}")


if __name__ == "__main__":
    main()


