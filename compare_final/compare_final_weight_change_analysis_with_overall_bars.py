# -*- coding: utf-8 -*-
"""compare_final_weight_change_analysis_with_overall_bars.py

比較 baseline 與 Final 兩個 ensemble 輸出資料夾中的 omics 權重變化，
並額外輸出「baseline vs 去掉 tissue-associated genes 後」的整體各組學平均權重對照圖。

新增輸出：
- overall_mean_weights_by_omics.csv
- figures/overall_mean_weights_grouped_bar.png
  * 同一張圖中用不同顏色比較 baseline 與 Final 的平均權重
  * 每個 bar 會標上數值
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    from scipy.stats import pearsonr, spearmanr, wilcoxon
except Exception:
    pearsonr = None
    spearmanr = None
    wilcoxon = None


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_drug_name(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    s = re.sub(r"\s+", " ", s)
    return s


def pretty_float(x, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def find_first_existing(base_dir: str | Path, candidates: Iterable[str], glob_patterns: Iterable[str] | None = None) -> Path:
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    candidates = list(candidates)
    glob_patterns = list(glob_patterns or [])

    for name in candidates:
        p = base_dir / name
        if p.exists():
            return p

    for name in candidates:
        hits = sorted(base_dir.rglob(name))
        if hits:
            return hits[0]

    loose_hits = []
    for pattern in glob_patterns:
        loose_hits.extend(sorted(base_dir.rglob(pattern)))

    if loose_hits:
        def _score(p: Path) -> tuple[int, int, str]:
            name = p.name.lower()
            score = 0
            if "ensemble" in name:
                score += 3
            if "summary" in name:
                score += 3
            if "weight" in name:
                score += 3
            if "notissuemarker" in name or "no_tissuemarker" in name or "no-tissuemarker" in name:
                score += 2
            return (-score, len(str(p)), str(p))
        loose_hits = sorted(set(loose_hits), key=_score)
        return loose_hits[0]

    nearby_csvs = sorted(base_dir.rglob("*.csv"))
    nearby_names = [str(p.relative_to(base_dir)) for p in nearby_csvs[:20]]
    raise FileNotFoundError(
        f"Cannot find any of these files under {base_dir}: {candidates}. "
        f"Also tried glob patterns: {glob_patterns}. Nearby csv files: {nearby_names}"
    )


def read_weights_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "drug" not in df.columns:
        raise ValueError(f"weights file must contain 'drug' column: {path}")
    df = df.copy()
    df["drug"] = df["drug"].map(normalize_drug_name)
    return df


def read_summary_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "drug" not in df.columns:
        raise ValueError(f"summary file must contain 'drug' column: {path}")
    df = df.copy()
    df["drug"] = df["drug"].map(normalize_drug_name)
    return df


def detect_weight_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("w_")]


def dominant_omics_from_row(row: pd.Series, weight_cols: list[str]) -> str:
    vals = []
    names = []
    for c in weight_cols:
        v = row.get(c, np.nan)
        if pd.notna(v):
            vals.append(abs(float(v)))
            names.append(c.replace("w_", ""))
    if not vals:
        return ""
    return names[int(np.argmax(vals))]


def safe_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if pearsonr is None:
        return float("nan"), float("nan")
    try:
        m = np.isfinite(x) & np.isfinite(y)
        if int(m.sum()) < 3:
            return float("nan"), float("nan")
        out = pearsonr(x[m], y[m])
        if hasattr(out, "statistic"):
            return float(out.statistic), float(out.pvalue)
        return float(out[0]), float(out[1])
    except Exception:
        return float("nan"), float("nan")


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if spearmanr is None:
        return float("nan"), float("nan")
    try:
        m = np.isfinite(x) & np.isfinite(y)
        if int(m.sum()) < 3:
            return float("nan"), float("nan")
        out = spearmanr(x[m], y[m])
        if hasattr(out, "correlation"):
            return float(out.correlation), float(out.pvalue)
        return float(out[0]), float(out[1])
    except Exception:
        return float("nan"), float("nan")


def safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if wilcoxon is None:
        return float("nan"), float("nan")
    try:
        m = np.isfinite(x) & np.isfinite(y)
        if int(m.sum()) < 3:
            return float("nan"), float("nan")
        if np.allclose(x[m] - y[m], 0.0, atol=1e-12, rtol=0.0):
            return 0.0, 1.0
        stat, p = wilcoxon(x[m], y[m], zero_method="wilcox", alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")


def plot_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: Path, ylabel: str | None = None) -> None:
    vis = df[[x_col, y_col]].dropna().copy()
    if vis.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(vis)), 5))
    ax.bar(vis[x_col].astype(str), vis[y_col].astype(float))
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel or y_col)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_mean_weights(overall_df: pd.DataFrame, out_path: Path, baseline_label: str, final_label: str) -> None:
    vis = overall_df.copy()
    if vis.empty:
        return
    omics = vis["omics"].astype(str).tolist()
    x = np.arange(len(omics))
    width = 0.36
    base_vals = vis["baseline_mean_weight"].to_numpy(dtype=float)
    final_vals = vis["final_mean_weight"].to_numpy(dtype=float)
    vals = np.concatenate([base_vals[np.isfinite(base_vals)], final_vals[np.isfinite(final_vals)]])
    vmax = float(np.max(np.abs(vals))) if len(vals) else 1.0
    if vmax <= 0 or not np.isfinite(vmax):
        vmax = 1.0
    pad = max(0.03, 0.18 * vmax)

    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(omics) + 2), 5.6))
    bars1 = ax.bar(x - width / 2.0, base_vals, width=width, label=baseline_label)
    bars2 = ax.bar(x + width / 2.0, final_vals, width=width, label=final_label)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(omics)
    ax.set_ylabel("mean weight")
    ax.set_xlabel("omics")
    ax.set_title("Overall mean omics weights: baseline vs tissue-gene-removed")
    ax.legend()
    ax.set_ylim(-vmax - pad, vmax + pad)

    for bars in [bars1, bars2]:
        for rect in bars:
            h = rect.get_height()
            if not np.isfinite(h):
                continue
            offset = 0.015 * (2 * vmax + 2 * pad)
            va = "bottom" if h >= 0 else "top"
            y = h + offset if h >= 0 else h - offset
            ax.text(rect.get_x() + rect.get_width() / 2.0, y, f"{h:.3f}", ha="center", va=va, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_box(long_df: pd.DataFrame, value_col: str, title: str, out_path: Path) -> None:
    if long_df.empty:
        return
    omics_order = list(dict.fromkeys(long_df["omics"].astype(str).tolist()))
    data = []
    labels = []
    for om in omics_order:
        arr = long_df.loc[long_df["omics"] == om, value_col].dropna().to_numpy(dtype=float)
        if len(arr) == 0:
            continue
        data.append(arr)
        labels.append(om)
    if not data:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(labels)), 5))
    try:
        ax.boxplot(data, tick_labels=labels, showfliers=True)
    except TypeError:
        ax.boxplot(data, labels=labels, showfliers=True)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("omics")
    ax.set_ylabel(value_col)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_delta(merged: pd.DataFrame, omics: list[str], sort_by: str, title: str, out_path: Path) -> None:
    cols = [f"delta_w_{o}" for o in omics if f"delta_w_{o}" in merged.columns]
    if not cols:
        return
    vis = merged[["drug", sort_by] + cols].copy() if sort_by in merged.columns else merged[["drug"] + cols].copy()
    if sort_by in vis.columns:
        vis = vis.sort_values(sort_by, ascending=False)
    vis = vis.reset_index(drop=True)
    mat = vis[cols].to_numpy(dtype=float)
    if mat.size == 0:
        return
    vmax = np.nanmax(np.abs(mat))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    fig_w = max(6, 1.5 * len(cols) + 2)
    fig_h = max(5, 0.28 * len(vis) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, aspect="auto", vmin=-vmax, vmax=vmax, cmap="coolwarm")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([c.replace("delta_w_", "") for c in cols], rotation=45, ha="right")
    step = max(1, len(vis) // 40)
    yticks = np.arange(0, len(vis), step)
    ax.set_yticks(yticks)
    ax.set_yticklabels(vis["drug"].iloc[::step].astype(str))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("delta weight (Final - Baseline)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(merged: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: Path) -> None:
    vis = merged[["drug", x_col, y_col]].dropna().copy()
    if len(vis) < 3:
        return
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(vis[x_col].astype(float), vis[y_col].astype(float), s=28)
    ax.axhline(0.0, linewidth=1.0)
    ax.axvline(0.0, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    vis["rank_abs"] = np.abs(vis[x_col].astype(float)) + np.abs(vis[y_col].astype(float))
    for _, row in vis.sort_values("rank_abs", ascending=False).head(6).iterrows():
        ax.annotate(row["drug"], (row[x_col], row[y_col]), fontsize=8, alpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_report_pdf(merged: pd.DataFrame, stats_df: pd.DataFrame, corr_df: pd.DataFrame, overall_mean_df: pd.DataFrame,
                      omics: list[str], out_pdf: Path, baseline_label: str, final_label: str) -> None:
    with PdfPages(out_pdf) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.70, "Omics weight change report", ha="center", va="center", fontsize=24, fontweight="bold")
        ax.text(0.5, 0.56, f"Compare: {baseline_label}  →  {final_label}", ha="center", va="center", fontsize=15)
        ax.text(0.5, 0.47, f"Drugs: {len(merged)} | Omics: {', '.join(omics)}", ha="center", va="center", fontsize=13)
        if "dominant_switched" in merged.columns:
            switched = int(merged["dominant_switched"].fillna(False).sum())
            ax.text(0.5, 0.38, f"Dominant omics switched: {switched}/{len(merged)} drugs", ha="center", va="center", fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        if not stats_df.empty:
            fig, ax = plt.subplots(figsize=(10, 0.55 * len(stats_df) + 2.5))
            ax.axis("off")
            show_cols = ["omics", "mean_delta", "median_delta", "mean_abs_delta", "n_positive", "n_negative", "wilcoxon_pvalue"]
            show = stats_df[[c for c in show_cols if c in stats_df.columns]].copy()
            for c in show.columns:
                if c != "omics":
                    show[c] = show[c].map(lambda v: pretty_float(v, 4) if isinstance(v, (float, int, np.floating, np.integer)) else str(v))
            tbl = ax.table(cellText=show.values, colLabels=show.columns, loc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1.0, 1.4)
            ax.set_title("Overall omics delta statistics", fontsize=15, pad=12)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if not overall_mean_df.empty:
            fig, ax = plt.subplots(figsize=(10, 0.55 * len(overall_mean_df) + 2.5))
            ax.axis("off")
            show = overall_mean_df.copy()
            for c in show.columns:
                if c != "omics":
                    show[c] = show[c].map(lambda v: pretty_float(v, 4) if isinstance(v, (float, int, np.floating, np.integer)) else str(v))
            tbl = ax.table(cellText=show.values, colLabels=show.columns, loc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1.0, 1.4)
            ax.set_title("Overall mean weights by omics", fontsize=15, pad=12)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if not corr_df.empty:
            preview = corr_df.sort_values(["metric_delta", "abs_spearman_r"], ascending=[True, False]).copy()
            preview = preview[["omics", "metric_delta", "pearson_r", "pearson_pvalue", "spearman_r", "spearman_pvalue", "n"]]
            fig, ax = plt.subplots(figsize=(12, min(16, 0.40 * len(preview) + 2.5)))
            ax.axis("off")
            show = preview.copy()
            for c in show.columns:
                if c not in {"omics", "metric_delta"}:
                    show[c] = show[c].map(lambda v: pretty_float(v, 4) if isinstance(v, (float, int, np.floating, np.integer)) else str(v))
            tbl = ax.table(cellText=show.values, colLabels=show.columns, loc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1.0, 1.25)
            ax.set_title("Weight-change vs metric-change correlations", fontsize=15, pad=12)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        temp_dir = out_pdf.parent / "_pdf_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        overall_grouped = temp_dir / "overall_mean_grouped.png"
        mean_bar = temp_dir / "mean_delta.png"
        mean_abs_bar = temp_dir / "mean_abs_delta.png"
        boxplot = temp_dir / "delta_boxplot.png"
        heatmap = temp_dir / "delta_heatmap.png"
        plot_grouped_mean_weights(overall_mean_df, overall_grouped, baseline_label, final_label)
        plot_bar(stats_df, "omics", "mean_delta", "Mean delta weight by omics", mean_bar, ylabel="mean delta")
        plot_bar(stats_df, "omics", "mean_abs_delta", "Mean absolute delta weight by omics", mean_abs_bar, ylabel="mean |delta|")
        long_rows = []
        for _, row in merged.iterrows():
            for om in omics:
                col = f"delta_w_{om}"
                if col in merged.columns:
                    long_rows.append({"drug": row["drug"], "omics": om, "delta_weight": row[col]})
        long_df = pd.DataFrame(long_rows)
        plot_box(long_df, "delta_weight", "Distribution of weight deltas", boxplot)
        plot_heatmap_delta(merged, omics, "total_abs_delta", "Drug × omics delta heatmap", heatmap)
        for f in [overall_grouped, mean_bar, mean_abs_bar, boxplot, heatmap]:
            if f.exists():
                img = plt.imread(f)
                fig = plt.figure(figsize=(11.69, 8.27))
                ax = fig.add_subplot(111)
                ax.imshow(img)
                ax.axis("off")
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        scatter_metrics = [m for m in ["delta_R2_ens", "delta_RMSE_ens", "delta_Spearman_ens", "delta_Pearson_ens"] if m in merged.columns]
        for om in omics:
            x_col = f"delta_w_{om}"
            if x_col not in merged.columns:
                continue
            for metric in scatter_metrics:
                f = temp_dir / f"scatter_{om}_{metric}.png"
                plot_scatter(merged, x_col, metric, f"{om}: {metric} vs {x_col}", f)
                if f.exists():
                    img = plt.imread(f)
                    fig = plt.figure(figsize=(11.69, 8.27))
                    ax = fig.add_subplot(111)
                    ax.imshow(img)
                    ax.axis("off")
                    fig.tight_layout()
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

        weight_cols_before = [f"baseline_w_{o}" for o in omics if f"baseline_w_{o}" in merged.columns]
        weight_cols_after = [f"final_w_{o}" for o in omics if f"final_w_{o}" in merged.columns]
        if weight_cols_before and weight_cols_after:
            all_vals = []
            for c in weight_cols_before + weight_cols_after:
                all_vals.extend(merged[c].dropna().astype(float).tolist())
            vmax = max(abs(min(all_vals)), abs(max(all_vals))) if all_vals else 1.0
            pad = max(0.05, 0.20 * vmax)
            y_min, y_max = -vmax - pad, vmax + pad
            for _, row in merged.sort_values("drug").iterrows():
                x = np.arange(len(omics))
                width = 0.35
                before_vals = [row.get(f"baseline_w_{o}", np.nan) for o in omics]
                after_vals = [row.get(f"final_w_{o}", np.nan) for o in omics]
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                bars1 = ax.bar(x - width / 2.0, before_vals, width=width, label=baseline_label)
                bars2 = ax.bar(x + width / 2.0, after_vals, width=width, label=final_label)
                ax.axhline(0.0, linewidth=1.0)
                ax.set_xticks(x)
                ax.set_xticklabels(omics)
                ax.set_ylim(y_min, y_max)
                ax.set_ylabel("weight")
                ax.set_xlabel("omics")
                ax.set_title(f"Per-drug weight change - {row['drug']}")
                ax.legend()
                for bars in [bars1, bars2]:
                    for rect in bars:
                        h = rect.get_height()
                        if pd.notna(h):
                            offset = 0.02 * (y_max - y_min)
                            va = "bottom" if h >= 0 else "top"
                            y = h + offset if h >= 0 else h - offset
                            ax.text(rect.get_x() + rect.get_width() / 2.0, y, f"{h:.3f}", ha="center", va=va, fontsize=9)
                text_lines = []
                if pd.notna(row.get("baseline_intercept", np.nan)) or pd.notna(row.get("final_intercept", np.nan)):
                    text_lines.append(
                        f"intercept: {baseline_label}={pretty_float(row.get('baseline_intercept', np.nan), 3)} | {final_label}={pretty_float(row.get('final_intercept', np.nan), 3)}"
                    )
                if "baseline_R2_ens" in row.index and "final_R2_ens" in row.index:
                    text_lines.append(
                        f"R2: {baseline_label}={pretty_float(row.get('baseline_R2_ens', np.nan), 3)} | {final_label}={pretty_float(row.get('final_R2_ens', np.nan), 3)} | delta={pretty_float(row.get('delta_R2_ens', np.nan), 3)}"
                    )
                if "baseline_RMSE_ens" in row.index and "final_RMSE_ens" in row.index:
                    text_lines.append(
                        f"RMSE: {baseline_label}={pretty_float(row.get('baseline_RMSE_ens', np.nan), 3)} | {final_label}={pretty_float(row.get('final_RMSE_ens', np.nan), 3)} | delta={pretty_float(row.get('delta_RMSE_ens', np.nan), 3)}"
                    )
                if "dominant_baseline" in row.index and "dominant_final" in row.index:
                    text_lines.append(
                        f"dominant omics: {baseline_label}={row.get('dominant_baseline', '')} | {final_label}={row.get('dominant_final', '')} | switched={bool(row.get('dominant_switched', False))}"
                    )
                if text_lines:
                    ax.text(0.99, 0.98, "\n".join(text_lines), transform=ax.transAxes, ha="right", va="top", fontsize=10)
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--final-dir", required=True)
    ap.add_argument("--out-dir", default="compare_final_results")
    ap.add_argument("--baseline-weights", default="")
    ap.add_argument("--baseline-summary", default="")
    ap.add_argument("--final-weights", default="")
    ap.add_argument("--final-summary", default="")
    ap.add_argument("--baseline-label", default="baseline")
    ap.add_argument("--final-label", default="Final")
    ap.add_argument("--top-n", type=int, default=15)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    fig_dir = ensure_dir(out_dir / "figures")

    baseline_weights_path = Path(args.baseline_weights) if args.baseline_weights else find_first_existing(
        args.baseline_dir,
        ["final_weights_per_drug.csv", "final_weights_per_drug_notissuemarker.csv"],
        glob_patterns=["*final_weights*.csv", "*weights_per_drug*.csv"],
    )
    baseline_summary_path = Path(args.baseline_summary) if args.baseline_summary else find_first_existing(
        args.baseline_dir,
        ["ensemble_summary.csv", "ensemble_summary_notissuemarker.csv", "ensemble_summary_no_tissuemarker.csv", "ensemble_summary_no-tissuemarker.csv"],
        glob_patterns=["*ensemble_summary*.csv", "*summary*.csv"],
    )
    final_weights_path = Path(args.final_weights) if args.final_weights else find_first_existing(
        args.final_dir,
        ["final_weights_per_drug.csv", "final_weights_per_drug_notissuemarker.csv"],
        glob_patterns=["*final_weights*.csv", "*weights_per_drug*.csv"],
    )
    final_summary_path = Path(args.final_summary) if args.final_summary else find_first_existing(
        args.final_dir,
        ["ensemble_summary.csv", "ensemble_summary_notissuemarker.csv", "ensemble_summary_no_tissuemarker.csv", "ensemble_summary_no-tissuemarker.csv"],
        glob_patterns=["*ensemble_summary*.csv", "*summary*.csv"],
    )

    print(f"[INFO] baseline weights : {baseline_weights_path}")
    print(f"[INFO] baseline summary : {baseline_summary_path}")
    print(f"[INFO] final weights    : {final_weights_path}")
    print(f"[INFO] final summary    : {final_summary_path}")

    base_w = read_weights_csv(baseline_weights_path)
    base_s = read_summary_csv(baseline_summary_path)
    final_w = read_weights_csv(final_weights_path)
    final_s = read_summary_csv(final_summary_path)

    base_weight_cols = detect_weight_cols(base_w)
    final_weight_cols = detect_weight_cols(final_w)
    common_weight_cols = [c for c in base_weight_cols if c in final_weight_cols]
    if not common_weight_cols:
        raise ValueError(f"No common weight columns found. baseline={base_weight_cols}, final={final_weight_cols}")
    omics = [c.replace("w_", "") for c in common_weight_cols]

    keep_weight_meta = [c for c in ["drug", "blend", "meta", "alpha", "intercept", "n"] if c in base_w.columns or c in final_w.columns]
    base_w_keep = base_w[[c for c in keep_weight_meta + common_weight_cols if c in base_w.columns]].copy()
    final_w_keep = final_w[[c for c in keep_weight_meta + common_weight_cols if c in final_w.columns]].copy()

    merged = base_w_keep.merge(final_w_keep, on="drug", suffixes=("_baseline", "_final"), how="inner")
    if merged.empty:
        raise RuntimeError("No overlapping drugs between baseline and Final weight files.")

    metric_candidates = [
        "n_oof", "RMSE_ens", "R2_ens", "Spearman_ens", "Pearson_ens",
        "RMSE_mean", "R2_mean",
        "RMSE_ens_fold_mean", "RMSE_ens_fold_std",
        "R2_ens_fold_mean", "R2_ens_fold_std",
        "Spearman_ens_fold_mean", "Spearman_ens_fold_std",
        "Pearson_ens_fold_mean", "Pearson_ens_fold_std",
    ]
    base_metrics = [c for c in ["drug"] + metric_candidates if c in base_s.columns]
    final_metrics = [c for c in ["drug"] + metric_candidates if c in final_s.columns]

    merged = merged.merge(base_s[base_metrics], on="drug", how="left", suffixes=("", "_DROP"))
    for c in metric_candidates:
        if c in merged.columns:
            merged = merged.rename(columns={c: f"baseline_{c}"})
    merged = merged.merge(final_s[final_metrics], on="drug", how="left", suffixes=("", "_DROP2"))
    for c in metric_candidates:
        if c in final_s.columns and c in merged.columns:
            merged = merged.rename(columns={c: f"final_{c}"})

    rename_map = {}
    for om in omics:
        rename_map[f"w_{om}_baseline"] = f"baseline_w_{om}"
        rename_map[f"w_{om}_final"] = f"final_w_{om}"
    for c in ["blend", "meta", "alpha", "intercept", "n"]:
        if f"{c}_baseline" in merged.columns:
            rename_map[f"{c}_baseline"] = f"baseline_{c}"
        if f"{c}_final" in merged.columns:
            rename_map[f"{c}_final"] = f"final_{c}"
    merged = merged.rename(columns=rename_map)

    for om in omics:
        merged[f"delta_w_{om}"] = merged[f"final_w_{om}"] - merged[f"baseline_w_{om}"]
        merged[f"abs_delta_w_{om}"] = merged[f"delta_w_{om}"].abs()

    for metric in metric_candidates:
        bcol = f"baseline_{metric}"
        fcol = f"final_{metric}"
        if bcol in merged.columns and fcol in merged.columns:
            merged[f"delta_{metric}"] = merged[fcol] - merged[bcol]

    merged["dominant_baseline"] = merged.apply(lambda r: dominant_omics_from_row(r.rename({f"baseline_w_{o}": f"w_{o}" for o in omics}), [f"w_{o}" for o in omics]), axis=1)
    merged["dominant_final"] = merged.apply(lambda r: dominant_omics_from_row(r.rename({f"final_w_{o}": f"w_{o}" for o in omics}), [f"w_{o}" for o in omics]), axis=1)
    merged["dominant_switched"] = merged["dominant_baseline"] != merged["dominant_final"]
    merged["total_abs_delta"] = merged[[f"abs_delta_w_{o}" for o in omics]].sum(axis=1)
    merged["mean_abs_delta"] = merged[[f"abs_delta_w_{o}" for o in omics]].mean(axis=1)

    long_rows = []
    for _, row in merged.iterrows():
        for om in omics:
            long_rows.append({
                "drug": row["drug"],
                "omics": om,
                "baseline_weight": row.get(f"baseline_w_{om}", np.nan),
                "final_weight": row.get(f"final_w_{om}", np.nan),
                "delta_weight": row.get(f"delta_w_{om}", np.nan),
                "abs_delta_weight": row.get(f"abs_delta_w_{om}", np.nan),
                "delta_R2_ens": row.get("delta_R2_ens", np.nan),
                "delta_RMSE_ens": row.get("delta_RMSE_ens", np.nan),
                "delta_Spearman_ens": row.get("delta_Spearman_ens", np.nan),
                "delta_Pearson_ens": row.get("delta_Pearson_ens", np.nan),
            })
    long_df = pd.DataFrame(long_rows)

    stats_rows = []
    for om in omics:
        x = merged[f"delta_w_{om}"].to_numpy(dtype=float)
        b = merged[f"baseline_w_{om}"].to_numpy(dtype=float)
        f = merged[f"final_w_{om}"].to_numpy(dtype=float)
        m = np.isfinite(x)
        if int(m.sum()) == 0:
            continue
        w_stat, w_p = safe_wilcoxon(f, b)
        stats_rows.append({
            "omics": om,
            "n": int(m.sum()),
            "mean_delta": float(np.nanmean(x)),
            "median_delta": float(np.nanmedian(x)),
            "std_delta": float(np.nanstd(x, ddof=1)) if int(m.sum()) > 1 else 0.0,
            "mean_abs_delta": float(np.nanmean(np.abs(x))),
            "median_abs_delta": float(np.nanmedian(np.abs(x))),
            "min_delta": float(np.nanmin(x)),
            "max_delta": float(np.nanmax(x)),
            "n_positive": int(np.sum(x[m] > 0)),
            "n_negative": int(np.sum(x[m] < 0)),
            "n_zero": int(np.sum(x[m] == 0)),
            "wilcoxon_stat": w_stat,
            "wilcoxon_pvalue": w_p,
        })
    stats_df = pd.DataFrame(stats_rows).sort_values("mean_abs_delta", ascending=False).reset_index(drop=True)

    overall_rows = []
    for om in omics:
        b = merged[f"baseline_w_{om}"].to_numpy(dtype=float)
        f = merged[f"final_w_{om}"].to_numpy(dtype=float)
        overall_rows.append({
            "omics": om,
            "baseline_mean_weight": float(np.nanmean(b)),
            "final_mean_weight": float(np.nanmean(f)),
            "delta_mean_weight": float(np.nanmean(f) - np.nanmean(b)),
            "baseline_mean_abs_weight": float(np.nanmean(np.abs(b))),
            "final_mean_abs_weight": float(np.nanmean(np.abs(f))),
            "delta_mean_abs_weight": float(np.nanmean(np.abs(f)) - np.nanmean(np.abs(b))),
        })
    overall_mean_df = pd.DataFrame(overall_rows)

    corr_rows = []
    metric_delta_cols = [c for c in ["delta_R2_ens", "delta_RMSE_ens", "delta_Spearman_ens", "delta_Pearson_ens"] if c in merged.columns]
    for om in omics:
        x = merged[f"delta_w_{om}"].to_numpy(dtype=float)
        for metric in metric_delta_cols:
            y = merged[metric].to_numpy(dtype=float)
            pr, pp = safe_pearson(x, y)
            sr, sp = safe_spearman(x, y)
            m = np.isfinite(x) & np.isfinite(y)
            corr_rows.append({
                "omics": om,
                "metric_delta": metric,
                "n": int(m.sum()),
                "pearson_r": pr,
                "pearson_pvalue": pp,
                "spearman_r": sr,
                "spearman_pvalue": sp,
                "abs_pearson_r": abs(pr) if np.isfinite(pr) else np.nan,
                "abs_spearman_r": abs(sr) if np.isfinite(sr) else np.nan,
            })
    corr_df = pd.DataFrame(corr_rows).sort_values(["metric_delta", "abs_spearman_r"], ascending=[True, False]).reset_index(drop=True)

    dominant_switches = merged.loc[merged["dominant_switched"]].copy()
    top_changed = merged.sort_values("total_abs_delta", ascending=False).head(args.top_n).copy()

    merged_out = out_dir / "merged_weight_change.csv"
    long_out = out_dir / "weight_change_long.csv"
    stats_out = out_dir / "omics_delta_stats.csv"
    overall_out = out_dir / "overall_mean_weights_by_omics.csv"
    corr_out = out_dir / "delta_metric_correlations.csv"
    switch_out = out_dir / "dominant_switches.csv"
    top_out = out_dir / "top_changed_drugs.csv"

    merged.sort_values("drug").to_csv(merged_out, index=False)
    long_df.to_csv(long_out, index=False)
    stats_df.to_csv(stats_out, index=False)
    overall_mean_df.to_csv(overall_out, index=False)
    corr_df.to_csv(corr_out, index=False)
    dominant_switches.sort_values("drug").to_csv(switch_out, index=False)
    top_changed.to_csv(top_out, index=False)

    plot_grouped_mean_weights(overall_mean_df, fig_dir / "overall_mean_weights_grouped_bar.png", args.baseline_label, args.final_label)
    plot_bar(stats_df, "omics", "mean_delta", "Mean delta weight by omics", fig_dir / "mean_delta_by_omics.png", ylabel="delta weight")
    plot_bar(stats_df, "omics", "mean_abs_delta", "Mean absolute delta weight by omics", fig_dir / "mean_abs_delta_by_omics.png", ylabel="mean |delta|")
    plot_box(long_df, "delta_weight", "Distribution of delta weights by omics", fig_dir / "delta_weight_boxplot.png")
    plot_heatmap_delta(merged, omics, "total_abs_delta", "Drug × omics delta heatmap (sorted by total abs delta)", fig_dir / "delta_weight_heatmap.png")
    for om in omics:
        x_col = f"delta_w_{om}"
        if "delta_R2_ens" in merged.columns:
            plot_scatter(merged, x_col, "delta_R2_ens", f"{om}: delta weight vs delta R2", fig_dir / f"scatter_{om}_vs_delta_R2.png")
        if "delta_RMSE_ens" in merged.columns:
            plot_scatter(merged, x_col, "delta_RMSE_ens", f"{om}: delta weight vs delta RMSE", fig_dir / f"scatter_{om}_vs_delta_RMSE.png")

    report_txt = out_dir / "compare_report.txt"
    lines = [
        f"Baseline dir : {args.baseline_dir}",
        f"Final dir    : {args.final_dir}",
        f"Baseline weights file : {baseline_weights_path}",
        f"Final weights file    : {final_weights_path}",
        f"Baseline summary file : {baseline_summary_path}",
        f"Final summary file    : {final_summary_path}",
        "",
        f"Matched drugs: {len(merged)}",
        f"Common omics : {', '.join(omics)}",
        f"Dominant omics switched: {int(merged['dominant_switched'].sum())}/{len(merged)}",
        "",
        "Overall mean weights by omics:",
    ]
    for _, row in overall_mean_df.iterrows():
        lines.append(
            f"- {row['omics']}: baseline_mean={pretty_float(row['baseline_mean_weight'], 6)}, final_mean={pretty_float(row['final_mean_weight'], 6)}, delta_mean={pretty_float(row['delta_mean_weight'], 6)}"
        )
    report_txt.write_text("\n".join(lines), encoding="utf-8")

    report_pdf = out_dir / "weight_change_report.pdf"
    create_report_pdf(merged, stats_df, corr_df, overall_mean_df, omics, report_pdf, args.baseline_label, args.final_label)

    print("\n[OK] Saved:")
    for p in [merged_out, long_out, stats_out, overall_out, corr_out, switch_out, top_out, report_txt, report_pdf]:
        print(f" - {p}")
    print(f" - {fig_dir}")


if __name__ == "__main__":
    main()
