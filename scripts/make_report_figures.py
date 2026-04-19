"""
從既有 CSV 產出報告用圖（PNG，300 dpi）。
輸出：compare_final/improve_followup_analysis/figures/

執行：python scripts/make_report_figures.py
依賴：pandas, matplotlib
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DRUG = ROOT / "compare_final/improve_followup_analysis/drug_table_with_y.csv"
SUMMARY = ROOT / "compare_final/analyze_improve_worsen_v7_theme_token_summary/analysis_summary.csv"
LOGISTIC = ROOT / "compare_final/improve_followup_analysis/logistic_theme_and_covariates_coef.csv"
GPROF_TOP = ROOT / "compare_final/improve_followup_analysis/gprofiler_enrichment_improve_vs_background.csv"
OUT = ROOT / "compare_final/improve_followup_analysis/figures"


def load_drugs() -> pd.DataFrame:
    use = ["drug", "delta", "delta_group", "metric_before", "metric_after"]
    df = pd.read_csv(DRUG, usecols=lambda c: c in use)
    df["delta_group"] = df["delta_group"].astype(str).str.lower()
    return df


def fig_summary_counts():
    df = load_drugs()
    order = ["improve", "stable", "worsen"]
    counts = df["delta_group"].value_counts().reindex(order).fillna(0).astype(int)
    colors = {"improve": "#2ca02c", "stable": "#7f7f7f", "worsen": "#d62728"}
    fig, ax = plt.subplots(figsize=(5, 4))
    labs = [f"{g.capitalize()}\n(n={counts[g]})" for g in order]
    ax.bar(labs, [counts[g] for g in order], color=[colors[g] for g in order], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Number of drugs")
    ax.set_title("Delta R² groups (57 drugs)")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_delta_group_counts.png", dpi=300)
    plt.close(fig)


def fig_delta_barh():
    df = load_drugs().sort_values("delta")
    colors = df["delta_group"].map(
        {"improve": "#2ca02c", "stable": "#7f7f7f", "worsen": "#d62728"}
    )
    fig, ax = plt.subplots(figsize=(8, 14))
    y = np.arange(len(df))
    ax.barh(y, df["delta"], color=colors, height=0.7, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(df["drug"], fontsize=7)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Delta R2 (after - before, ensemble)")
    ax.set_title("Per-drug delta R² (green=improve, gray=stable, red=worsen)")
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, fc="#2ca02c", label="improve"),
            plt.Rectangle((0, 0), 1, 1, fc="#7f7f7f", label="stable"),
            plt.Rectangle((0, 0), 1, 1, fc="#d62728", label="worsen"),
        ],
        loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig2_per_drug_delta_r2.png", dpi=300)
    plt.close(fig)


def fig_scatter_before_vs_delta():
    df = load_drugs()
    fig, ax = plt.subplots(figsize=(6, 5))
    for g, c in [("improve", "#2ca02c"), ("stable", "#7f7f7f"), ("worsen", "#d62728")]:
        m = df["delta_group"] == g
        ax.scatter(df.loc[m, "metric_before"], df.loc[m, "delta"], c=c, s=38, alpha=0.85, label=g, edgecolors="k", linewidths=0.3)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("R² before")
    ax.set_ylabel("Delta R²")
    ax.set_title("Change vs. baseline R²")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig3_scatter_before_vs_delta.png", dpi=300)
    plt.close(fig)


def fig_logistic_themes_clipped():
    """僅 theme_；係數過大時截斷顯示以免壓扁座標。"""
    coef = pd.read_csv(LOGISTIC)
    t = coef[coef["feature"].str.startswith("theme_")].copy()
    t["label"] = t["feature"].str.replace("theme_", "", regex=False)
    cap = 3.0
    t["coef_plot"] = t["coef"].clip(-cap, cap)
    t = t.sort_values("coef_plot")
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = np.where(t["coef_plot"] >= 0, "#1f77b4", "#ff7f0e")
    ax.barh(t["label"], t["coef_plot"], color=colors)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel(f"Logistic coefficient (clipped to [{-cap}, {cap}])")
    ax.set_title("Theme effects on P(improve) | covariates")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_logistic_theme_coef_clipped.png", dpi=300)
    plt.close(fig)


def fig_gprofiler_top():
    if not GPROF_TOP.exists():
        return
    g = pd.read_csv(GPROF_TOP)
    g = g[g["p_value"].notna() & (g["p_value"] < 1)].nsmallest(12, "p_value")
    if g.empty:
        return
    g["short"] = g["name"].str[:60] + np.where(g["name"].str.len() > 60, "...", "")
    fig, ax = plt.subplots(figsize=(8, max(4, len(g) * 0.35)))
    y = np.arange(len(g))
    ax.barh(y, -np.log10(g["p_value"].clip(lower=1e-300)))
    ax.set_yticks(y)
    ax.set_yticklabels(g["source"] + " | " + g["short"], fontsize=8)
    ax.set_xlabel("-log10(p)")
    ax.set_title("g:Profiler: lowest p-value terms (custom background)")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_gprofiler_top_terms.png", dpi=300)
    plt.close(fig)


def write_table_summary():
    s = pd.read_csv(SUMMARY)
    lines = [
        "| Item | Value |",
        "|------|-------|",
        f"| Metric | {s['metric'].iloc[0]} |",
        f"| Delta definition | {s['delta_definition'].iloc[0]} |",
        f"| Total drugs | {int(s['n_total_drugs'].iloc[0])} |",
        f"| Improve | {int(s['n_improve'].iloc[0])} |",
        f"| Worsen | {int(s['n_worsen'].iloc[0])} |",
        f"| Stable | {int(s['n_stable'].iloc[0])} |",
        f"| Threshold (improve/worsen) | ±{float(s['improve_threshold'].iloc[0]):g} |",
    ]
    (OUT / "table1_summary_md.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.unicode_minus"] = False
    fig_summary_counts()
    fig_delta_barh()
    fig_scatter_before_vs_delta()
    fig_logistic_themes_clipped()
    fig_gprofiler_top()
    write_table_summary()
    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
