#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze why some drugs improve or worsen after removing tissue markers in multi-omics ensemble.

Interpretable version:
- keeps the original drug-level delta comparison
- can merge external chemistry / mechanism / target / activity files
- defaults to an INTERPRETABLE profile that filters out posterior model features
  (stacking weights / coverage) and assay or cell-line metadata tokens
- can optionally switch back to a broader profile with --analysis-profile all

Main outputs:
- drug_delta_table.csv
- drug_sample_aggregated.csv
- external_feature_aggregated.csv
- external_feature_aggregated_interpretable.csv
- numeric_feature_associations.csv
- numeric_group_comparison_improve_vs_worsen.csv
- category_enrichment_improve.csv
- category_enrichment_worsen.csv
- category_enrichment_improve_vs_worsen.csv
- category_enrichment_worsen_vs_improve.csv
- all_category_tokens.csv
- all_category_tokens_interpretable.csv
- feature_filter_audit.csv
- token_filter_audit.csv
- plots/*.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, pearsonr, spearmanr


# ----------------------------
# basic utils
# ----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def norm_drug(s) -> Optional[str]:
    if pd.isna(s):
        return None
    x = str(s).strip().upper()
    x = re.sub(r"\s+", " ", x)
    return x if x else None


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def first_existing(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    low = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def detect_metric_column(df: pd.DataFrame, metric: str) -> str:
    metric = metric.strip()
    candidates = [metric, metric + "_ens", metric.upper(), metric.lower()]
    col = first_existing(df, candidates)
    if col is None:
        raise ValueError(f"Cannot find metric column for '{metric}'. Available columns: {df.columns.tolist()}")
    return col


def split_multi_value(v: str) -> list[str]:
    if pd.isna(v):
        return []
    txt = str(v)
    parts = re.split(r"[,;/|]", txt)
    out = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            out.append(p)
    return out


def shannon_entropy(values: pd.Series) -> float:
    s = values.dropna().astype(str)
    if s.empty:
        return float("nan")
    p = s.value_counts(normalize=True)
    return float(-(p * np.log2(p)).sum())


def mode_first(values: pd.Series):
    s = values.dropna().astype(str)
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iloc[0] if not m.empty else s.iloc[0]


def iqr(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return float("nan")
    return float(x.quantile(0.75) - x.quantile(0.25))


def add_fdr(df: pd.DataFrame, p_col: str = "pvalue", out_col: str = "fdr_bh") -> pd.DataFrame:
    if df.empty or p_col not in df.columns:
        return df
    p = pd.to_numeric(df[p_col], errors="coerce").to_numpy(dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    m = np.isfinite(p)
    if m.sum() == 0:
        df[out_col] = out
        return df
    pv = p[m]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out_idx = np.where(m)[0][order]
    out[out_idx] = adj
    df[out_col] = out
    return df


def is_numeric_or_bool(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    sx = np.var(x, ddof=1)
    sy = np.var(y, ddof=1)
    pooled = ((len(x) - 1) * sx + (len(y) - 1) * sy) / (len(x) + len(y) - 2)
    if not np.isfinite(pooled) or pooled <= 0:
        return np.nan
    return safe_float((np.mean(x) - np.mean(y)) / np.sqrt(pooled))


# ----------------------------
# reading core inputs
# ----------------------------

def read_ensemble_summary(path: Path, metric: str) -> tuple[pd.DataFrame, dict[str, str]]:
    df = pd.read_csv(path)
    drug_col = first_existing(df, ["drug", "Drug"])
    if drug_col is None:
        raise ValueError(f"No drug column in {path}")
    metric_col = detect_metric_column(df, metric)

    out = df[[drug_col, metric_col]].copy()
    out.columns = ["drug", "metric"]
    out["drug"] = out["drug"].map(norm_drug)
    out["metric"] = pd.to_numeric(out["metric"], errors="coerce")
    out = out.dropna(subset=["drug"]).drop_duplicates("drug")
    return out, {"metric": f"ensemble_summary:{path.name}"}


def aggregate_drug_sample(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    df = pd.read_csv(path, low_memory=False)
    drug_col = first_existing(df, ["drug", "Drug"])
    if drug_col is None:
        raise ValueError(f"No drug column in {path}")
    df = df.copy()
    df["drug"] = df[drug_col].map(norm_drug)
    df = df.dropna(subset=["drug"])

    num_map = {
        "Y": first_existing(df, ["Y", "y"]),
        "AUC": first_existing(df, ["AUC", "auc"]),
        "RMSE_curve": first_existing(df, ["RMSE", "rmse"]),
        "Z_SCORE": first_existing(df, ["Z_SCORE", "z_score", "zscore"]),
    }
    for _, c in num_map.items():
        if c is not None:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    cell_col = first_existing(df, ["CellLineName", "cell_line", "CellLine"])
    tcga_col = first_existing(df, ["TCGA_DESC", "tcga_desc", "tissue", "Tissue"])
    target_col = first_existing(df, ["PUTATIVE_TARGET", "putative_target", "target", "Target"])
    pathway_col = first_existing(df, ["PATHWAY_NAME", "pathway_name", "pathway", "Pathway"])
    drugid_col = first_existing(df, ["DrugID", "drugid"])

    rows = []
    token_rows = []

    for drug, g in df.groupby("drug", sort=False):
        row = {"drug": drug}
        row["n_rows"] = int(len(g))
        if cell_col is not None:
            row["n_cell_lines"] = int(g[cell_col].astype(str).nunique())
        if drugid_col is not None:
            row["DrugID_mode"] = mode_first(g[drugid_col])

        if num_map["Y"] is not None:
            s = g[num_map["Y"]]
            row["Y_mean"] = safe_float(s.mean())
            row["Y_std"] = safe_float(s.std(ddof=1))
            row["Y_iqr"] = iqr(s)
            row["Y_median"] = safe_float(s.median())
        if num_map["AUC"] is not None:
            s = g[num_map["AUC"]]
            row["AUC_mean"] = safe_float(s.mean())
            row["AUC_std"] = safe_float(s.std(ddof=1))
        if num_map["RMSE_curve"] is not None:
            s = g[num_map["RMSE_curve"]]
            row["curve_RMSE_mean"] = safe_float(s.mean())
            row["curve_RMSE_std"] = safe_float(s.std(ddof=1))
        if num_map["Z_SCORE"] is not None:
            s = g[num_map["Z_SCORE"]]
            row["Z_SCORE_mean"] = safe_float(s.mean())
            row["Z_SCORE_std"] = safe_float(s.std(ddof=1))
            row["Z_SCORE_abs_mean"] = safe_float(s.abs().mean())

        if tcga_col is not None:
            row["dominant_TCGA_DESC"] = mode_first(g[tcga_col])
            row["n_TCGA_DESC"] = int(g[tcga_col].dropna().astype(str).nunique())
            row["TCGA_entropy"] = shannon_entropy(g[tcga_col])
        if target_col is not None:
            row["dominant_PUTATIVE_TARGET"] = mode_first(g[target_col])
            row["n_PUTATIVE_TARGET"] = int(g[target_col].dropna().astype(str).nunique())
        if pathway_col is not None:
            row["dominant_PATHWAY_NAME"] = mode_first(g[pathway_col])
            row["n_PATHWAY_NAME"] = int(g[pathway_col].dropna().astype(str).nunique())

        rows.append(row)

        token_sets = {"PUTATIVE_TARGET": set(), "PATHWAY_NAME": set(), "TCGA_DESC": set()}
        if target_col is not None:
            for v in g[target_col].dropna().astype(str):
                token_sets["PUTATIVE_TARGET"].update(split_multi_value(v))
        if pathway_col is not None:
            for v in g[pathway_col].dropna().astype(str):
                token_sets["PATHWAY_NAME"].update(split_multi_value(v))
        if tcga_col is not None:
            token_sets["TCGA_DESC"].update(g[tcga_col].dropna().astype(str).str.strip().tolist())

        for field, tokens in token_sets.items():
            for token in sorted(t for t in tokens if t):
                token_rows.append({"drug": drug, "source": "drug_sample", "field": field, "token": token})

    agg = pd.DataFrame(rows)
    tok = pd.DataFrame(token_rows)
    feature_sources = {c: "drug_sample" for c in agg.columns if c != "drug"}
    return agg, tok, feature_sources


def read_weights(path: Optional[Path], suffix: str, source_name: str) -> tuple[pd.DataFrame, dict[str, str]]:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["drug"]), {}
    df = pd.read_csv(path)
    drug_col = first_existing(df, ["drug", "Drug"])
    if drug_col is None:
        return pd.DataFrame(columns=["drug"]), {}
    keep = [drug_col] + [c for c in df.columns if str(c).startswith("w_") or str(c) in ["intercept", "n"]]
    out = df[keep].copy().rename(columns={drug_col: "drug"})
    out["drug"] = out["drug"].map(norm_drug)
    rename = {c: f"{c}{suffix}" for c in out.columns if c != "drug"}
    out = out.rename(columns=rename)
    src = {c: source_name for c in out.columns if c != "drug"}
    return out, src


def read_coverage(path: Optional[Path], suffix: str, source_name: str) -> tuple[pd.DataFrame, dict[str, str]]:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["drug"]), {}
    df = pd.read_csv(path)
    drug_col = first_existing(df, ["drug", "Drug"])
    if drug_col is None:
        return pd.DataFrame(columns=["drug"]), {}
    out = df.copy().rename(columns={drug_col: "drug"})
    out["drug"] = out["drug"].map(norm_drug)
    rename = {c: f"{c}{suffix}" for c in out.columns if c != "drug"}
    out = out.rename(columns=rename)
    src = {c: source_name for c in out.columns if c != "drug"}
    return out, src


# ----------------------------
# external readers
# ----------------------------

def generic_per_drug_table(
    path: Optional[Path],
    prefix: str,
    source_name: str,
    numeric_keep: Optional[list[str]] = None,
    token_cols: Optional[list[str]] = None,
    categorical_mode_cols: Optional[list[str]] = None,
    drop_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["drug"]), pd.DataFrame(columns=["drug", "source", "field", "token"]), {}

    df = pd.read_csv(path, low_memory=False)
    drug_col = first_existing(df, ["drug", "Drug", "original_drug"])
    if drug_col is None:
        return pd.DataFrame(columns=["drug"]), pd.DataFrame(columns=["drug", "source", "field", "token"]), {}
    df = df.copy()
    df["drug"] = df[drug_col].map(norm_drug)
    df = df.dropna(subset=["drug"])

    drop_cols = set(drop_cols or [])
    token_cols = [c for c in (token_cols or []) if c in df.columns]
    categorical_mode_cols = [c for c in (categorical_mode_cols or []) if c in df.columns]

    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype(float)

    numeric_cols = []
    if numeric_keep is None:
        for c in df.columns:
            if c in {drug_col, "drug"} or c in drop_cols:
                continue
            if is_numeric_or_bool(df[c]):
                numeric_cols.append(c)
    else:
        numeric_cols = [c for c in numeric_keep if c in df.columns]

    rows = []
    token_rows = []
    for drug, g in df.groupby("drug", sort=False):
        row = {"drug": drug}
        for c in numeric_cols:
            s = pd.to_numeric(g[c], errors="coerce")
            row[f"{prefix}_{c}"] = safe_float(s.mean())
            if len(g) > 1:
                row[f"{prefix}_{c}_std"] = safe_float(s.std(ddof=1))
        for c in categorical_mode_cols:
            row[f"{prefix}_{c}_mode"] = mode_first(g[c])
            row[f"{prefix}_{c}_nunique"] = int(g[c].dropna().astype(str).nunique())
        rows.append(row)

        for c in token_cols:
            toks = set()
            for v in g[c].dropna().astype(str):
                vals = split_multi_value(v)
                if vals:
                    toks.update(vals)
                else:
                    x = re.sub(r"\s+", " ", v).strip()
                    if x:
                        toks.add(x)
            for tok in sorted(t for t in toks if t and str(t).lower() != "nan"):
                token_rows.append({"drug": drug, "source": source_name, "field": f"{prefix}_{c}", "token": tok})

    agg = pd.DataFrame(rows)
    tok = pd.DataFrame(token_rows)
    feature_sources = {c: source_name for c in agg.columns if c != "drug"}
    return agg, tok, feature_sources


def aggregate_mechanisms_long(path: Optional[Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["drug"]), pd.DataFrame(columns=["drug", "source", "field", "token"]), {}
    df = pd.read_csv(path, low_memory=False)
    drug_col = first_existing(df, ["drug", "Drug"])
    if drug_col is None:
        return pd.DataFrame(columns=["drug"]), pd.DataFrame(columns=["drug", "source", "field", "token"]), {}
    df = df.copy()
    df["drug"] = df[drug_col].map(norm_drug)
    df = df.dropna(subset=["drug"])
    for c in ["direct_interaction", "disease_efficacy", "molecular_mechanism"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rows = []
    token_rows = []
    for drug, g in df.groupby("drug", sort=False):
        row = {"drug": drug, "mech_n_rows": int(len(g))}
        if "target_chembl_id" in g.columns:
            row["mech_n_unique_target_chembl_id"] = int(g["target_chembl_id"].dropna().astype(str).nunique())
        if "mechanism_of_action" in g.columns:
            row["mech_n_unique_mechanism_of_action"] = int(g["mechanism_of_action"].dropna().astype(str).nunique())
        if "action_type" in g.columns:
            row["mech_n_unique_action_type"] = int(g["action_type"].dropna().astype(str).nunique())
        for c in ["direct_interaction", "disease_efficacy", "molecular_mechanism"]:
            if c in g.columns:
                row[f"mech_{c}_mean"] = safe_float(pd.to_numeric(g[c], errors="coerce").mean())
        rows.append(row)

        for c in ["mechanism_of_action", "action_type"]:
            if c in g.columns:
                toks = set(g[c].dropna().astype(str).str.strip())
                for tok in sorted(t for t in toks if t and t.lower() != "nan"):
                    token_rows.append({"drug": drug, "source": "mechanisms_long", "field": f"mech_{c}", "token": tok})

    agg = pd.DataFrame(rows)
    tok = pd.DataFrame(token_rows)
    src = {c: "mechanisms_long" for c in agg.columns if c != "drug"}
    return agg, tok, src


def aggregate_activities_long(path: Optional[Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["drug"]), pd.DataFrame(columns=["drug", "source", "field", "token"]), {}
    df = pd.read_csv(path, low_memory=False)
    drug_col = first_existing(df, ["drug", "Drug"])
    if drug_col is None:
        return pd.DataFrame(columns=["drug"]), pd.DataFrame(columns=["drug", "source", "field", "token"]), {}
    df = df.copy()
    df["drug"] = df[drug_col].map(norm_drug)
    df = df.dropna(subset=["drug"])
    for c in ["standard_value", "pchembl_value"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rows = []
    token_rows = []
    for drug, g in df.groupby("drug", sort=False):
        row = {"drug": drug, "act_n_rows": int(len(g))}
        for c in ["assay_chembl_id", "target_chembl_id", "target_pref_name", "document_chembl_id", "bao_label", "standard_type"]:
            if c in g.columns:
                row[f"act_n_unique_{c}"] = int(g[c].dropna().astype(str).nunique())
        if "pchembl_value" in g.columns:
            s = pd.to_numeric(g["pchembl_value"], errors="coerce")
            row["act_pchembl_mean"] = safe_float(s.mean())
            row["act_pchembl_median"] = safe_float(s.median())
            row["act_pchembl_max"] = safe_float(s.max())
            row["act_pchembl_std"] = safe_float(s.std(ddof=1))
        if "standard_type" in g.columns:
            vc = g["standard_type"].dropna().astype(str).str.upper().value_counts()
            for t, cnt in vc.items():
                tclean = re.sub(r"[^A-Z0-9]+", "_", t).strip("_")
                row[f"act_count_standard_type_{tclean}"] = int(cnt)
        rows.append(row)

        # keep broad token set here; filtering happens later
        for c in ["standard_type", "bao_label", "target_pref_name", "target_organism"]:
            if c in g.columns:
                toks = set(g[c].dropna().astype(str).str.strip())
                for tok in sorted(t for t in toks if t and t.lower() != "nan"):
                    token_rows.append({"drug": drug, "source": "activities_long", "field": f"act_{c}", "token": tok})

    agg = pd.DataFrame(rows)
    tok = pd.DataFrame(token_rows)
    src = {c: "activities_long" for c in agg.columns if c != "drug"}
    return agg, tok, src


def aggregate_targets_long(path: Optional[Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["drug"]), pd.DataFrame(columns=["drug", "source", "field", "token"]), {}
    df = pd.read_csv(path, low_memory=False)
    drug_col = first_existing(df, ["drug", "Drug"])
    if drug_col is None:
        return pd.DataFrame(columns=["drug"]), pd.DataFrame(columns=["drug", "source", "field", "token"]), {}
    df = df.copy()
    df["drug"] = df[drug_col].map(norm_drug)
    df = df.dropna(subset=["drug"])

    rows = []
    token_rows = []
    for drug, g in df.groupby("drug", sort=False):
        row = {"drug": drug, "tgt_n_rows": int(len(g))}
        for c in ["target_chembl_id", "pref_name", "accessions", "component_ids"]:
            if c in g.columns:
                row[f"tgt_n_unique_{c}"] = int(g[c].dropna().astype(str).nunique())
        if "target_type" in g.columns:
            vc = g["target_type"].dropna().astype(str).value_counts()
            for t, cnt in vc.items():
                tclean = re.sub(r"[^A-Z0-9]+", "_", t.upper()).strip("_")
                row[f"tgt_count_target_type_{tclean}"] = int(cnt)
        rows.append(row)

        if "target_type" in g.columns:
            toks = set(g["target_type"].dropna().astype(str).str.strip())
            for tok in sorted(t for t in toks if t and t.lower() != "nan"):
                token_rows.append({"drug": drug, "source": "targets_long", "field": "tgt_target_type", "token": tok})

        if {"target_type", "pref_name"}.issubset(g.columns):
            gg = g[g["target_type"].astype(str).str.upper() == "SINGLE PROTEIN"]
            toks = set(gg["pref_name"].dropna().astype(str).str.strip())
            for tok in sorted(t for t in toks if t and t.lower() != "nan"):
                token_rows.append({"drug": drug, "source": "targets_long", "field": "tgt_single_protein_pref_name", "token": tok})

    agg = pd.DataFrame(rows)
    tok = pd.DataFrame(token_rows)
    src = {c: "targets_long" for c in agg.columns if c != "drug"}
    return agg, tok, src


# ----------------------------
# interpretable filtering
# ----------------------------

INTERPRETABLE_NUMERIC_PATTERNS = {
    "drug_sample": [
        r"^n_rows$", r"^n_cell_lines$",
        r"^Y_", r"^AUC_", r"^curve_RMSE_", r"^Z_SCORE_",
        r"^TCGA_entropy$", r"^n_PUTATIVE_TARGET$", r"^n_PATHWAY_NAME$"
    ],
    "drug_feature_summary": [
        r"^dfs_max_phase$", r"^dfs_oral$", r"^dfs_parenteral$", r"^dfs_topical$",
        r"^dfs_black_box_warning$", r"^dfs_availability_type$", r"^dfs_first_approval$",
        r"^dfs_full_mwt$", r"^dfs_alogp$", r"^dfs_psa$", r"^dfs_hba$", r"^dfs_hbd$",
        r"^dfs_rtb$", r"^dfs_aromatic_rings$", r"^dfs_heavy_atoms$",
        r"^dfs_num_ro5_violations$", r"^dfs_qed_weighted$",
        r"^dfs_best_pchembl$", r"^dfs_median_pchembl$", r"^dfs_best_standard_value$",
        r"^dfs_(ac50|ec50|ic50|kd|ki)_(best_pchembl|median_pchembl|n)$"
    ],
    "molecules": [
        r"^mol_max_phase$", r"^mol_oral$", r"^mol_parenteral$", r"^mol_topical$",
        r"^mol_black_box_warning$", r"^mol_availability_type$", r"^mol_first_approval$",
        r"^mol_full_mwt$", r"^mol_alogp$", r"^mol_psa$", r"^mol_hba$", r"^mol_hbd$",
        r"^mol_rtb$", r"^mol_aromatic_rings$", r"^mol_heavy_atoms$",
        r"^mol_num_ro5_violations$", r"^mol_qed_weighted$"
    ],
    "pubchem_by_drug": [
        r"^pubchem_MolecularWeight$", r"^pubchem_XLogP$", r"^pubchem_TPSA$",
        r"^pubchem_HBondDonorCount$", r"^pubchem_HBondAcceptorCount$",
        r"^pubchem_RotatableBondCount$", r"^pubchem_Complexity$"
    ],
    "mechanisms_long": [
        r"^mech_n_unique_mechanism_of_action$", r"^mech_n_unique_action_type$",
        r"^mech_direct_interaction_mean$", r"^mech_disease_efficacy_mean$", r"^mech_molecular_mechanism_mean$"
    ],
    "activities_long": [
        r"^act_pchembl_(mean|median|max|std)$"
    ],
}

INTERPRETABLE_TOKEN_FIELDS = {
    "drug_sample": {"PUTATIVE_TARGET", "PATHWAY_NAME"},
    "drug_feature_summary": {"dfs_mechanism_of_action_list", "dfs_action_type_list", "dfs_molecule_type", "dfs_indication_class"},
    "molecules": {"mol_molecule_type", "mol_indication_class"},
    "mechanisms_long": {"mech_mechanism_of_action", "mech_action_type"},
    "targets_long": {"tgt_target_type", "tgt_single_protein_pref_name"},
}

INTERPRETABLE_TARGET_TYPE_TOKENS = {"SINGLE PROTEIN", "PROTEIN COMPLEX", "PROTEIN FAMILY"}

POSTERIOR_SOURCES = {"weights_before", "weights_after", "coverage_before", "coverage_after"}
METADATA_SOURCES = {"molecule_resolution", "cid_resolution_results"}

PROTECTED_NONFEATURE_COLS = {
    "drug", "metric_before", "metric_after", "delta", "delta_group",
    "dominant_PUTATIVE_TARGET", "dominant_PATHWAY_NAME", "dominant_TCGA_DESC", "DrugID_mode"
}


def matches_any(s: str, patterns: list[str]) -> bool:
    return any(re.search(p, s) for p in patterns)


def numeric_feature_decision(col: str, source: str, profile: str) -> tuple[bool, str]:
    if profile == "all":
        return True, "profile_all"
    if col in PROTECTED_NONFEATURE_COLS:
        return False, "protected_nonfeature"
    if source in POSTERIOR_SOURCES:
        return False, "posterior_model_output"
    if source in METADATA_SOURCES:
        return False, "resolution_metadata"
    patterns = INTERPRETABLE_NUMERIC_PATTERNS.get(source)
    if patterns is None:
        return False, f"source_not_in_interpretable_allowlist:{source}"
    if matches_any(col, patterns):
        return True, "interpretable_numeric_allowlist"
    return False, f"excluded_by_interpretable_allowlist:{source}"


GENERIC_NOISY_TOKEN_PATTERNS = [
    r"^HOMO SAPIENS$", r"^MUS MUSCULUS$", r"^RATTUS NORVEGICUS$",
    r"^SINGLE PROTEIN FORMAT$", r"^PROTEIN COMPLEX FORMAT$", r"^CELL[- ]LINE FORMAT$",
    r"^ORGANISM$", r"^UNKNOWN$", r"^N/A$"
]


def token_decision(source: str, field: str, token: str, profile: str) -> tuple[bool, str]:
    if profile == "all":
        return True, "profile_all"
    allowed_fields = INTERPRETABLE_TOKEN_FIELDS.get(source)
    if not allowed_fields:
        return False, f"source_not_in_interpretable_allowlist:{source}"
    if field not in allowed_fields:
        return False, f"field_not_in_interpretable_allowlist:{field}"

    tok = re.sub(r"\s+", " ", str(token)).strip()
    tok_upper = tok.upper()
    if any(re.search(p, tok_upper) for p in GENERIC_NOISY_TOKEN_PATTERNS):
        return False, "generic_metadata_token"

    if source == "targets_long" and field == "tgt_target_type":
        if tok_upper in INTERPRETABLE_TARGET_TYPE_TOKENS:
            return True, "interpretable_target_type"
        return False, "target_type_not_interpretable"

    if source == "targets_long" and field == "tgt_single_protein_pref_name":
        # keep protein names, drop obvious cell-line-like names
        if re.search(r"\bCELL\b", tok_upper):
            return False, "cell_line_like_target_name"
        if tok_upper in {"PBMC", "HELA", "HEPG2"}:
            return False, "cell_line_like_target_name"
        return True, "interpretable_single_protein_target"

    return True, "interpretable_token_allowlist"


def build_feature_filter_audit(df: pd.DataFrame, feature_sources: dict[str, str], profile: str) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        if col in PROTECTED_NONFEATURE_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        source = feature_sources.get(col, "unknown")
        keep, reason = numeric_feature_decision(col, source, profile)
        rows.append({"feature": col, "source": source, "keep": keep, "reason": reason})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["keep", "source", "feature"], ascending=[False, True, True])
    return out


def build_token_filter_audit(tokens_df: pd.DataFrame, profile: str) -> pd.DataFrame:
    if tokens_df.empty:
        return pd.DataFrame(columns=["source", "field", "token", "keep", "reason"])
    rows = []
    uniq = tokens_df[["source", "field", "token"]].drop_duplicates()
    for _, r in uniq.iterrows():
        keep, reason = token_decision(str(r["source"]), str(r["field"]), str(r["token"]), profile)
        rows.append({"source": r["source"], "field": r["field"], "token": r["token"], "keep": keep, "reason": reason})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["keep", "source", "field", "token"], ascending=[False, True, True, True])
    return out


def filter_numeric_table(df: pd.DataFrame, feature_sources: dict[str, str], profile: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = build_feature_filter_audit(df, feature_sources, profile)
    if audit.empty:
        return df.copy(), audit
    keep_cols = set(audit.loc[audit["keep"], "feature"])
    base_cols = [c for c in df.columns if c in PROTECTED_NONFEATURE_COLS or c == "drug"]
    out = df[base_cols + [c for c in df.columns if c in keep_cols]].copy()
    return out, audit


def filter_tokens_df(tokens_df: pd.DataFrame, profile: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = build_token_filter_audit(tokens_df, profile)
    if tokens_df.empty:
        return tokens_df.copy(), audit
    if audit.empty:
        return tokens_df.copy(), audit
    merged = tokens_df.merge(audit, on=["source", "field", "token"], how="left")
    out = merged.loc[merged["keep"] == True, ["drug", "source", "field", "token"]].copy()  # noqa: E712
    out = out.drop_duplicates()
    return out, audit


# ----------------------------
# delta / statistics
# ----------------------------

def classify_delta(delta: float, improve_thr: float, worsen_thr: float) -> str:
    if pd.isna(delta):
        return "missing"
    if delta >= improve_thr:
        return "improve"
    if delta <= -abs(worsen_thr):
        return "worsen"
    return "stable"


def infer_feature_source(col: str, feature_sources: dict[str, str]) -> str:
    return feature_sources.get(col, "unknown")


def calc_numeric_assoc(df: pd.DataFrame, delta_col: str, feature_sources: dict[str, str]) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        if col in PROTECTED_NONFEATURE_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[delta_col], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 4:
            continue
        try:
            sp = spearmanr(x[m], y[m]).correlation
        except Exception:
            sp = np.nan
        try:
            pr = pearsonr(x[m], y[m]).statistic
        except Exception:
            try:
                pr, _ = pearsonr(x[m], y[m])
            except Exception:
                pr = np.nan
        rows.append({
            "feature": col,
            "source": infer_feature_source(col, feature_sources),
            "n": int(m.sum()),
            "spearman": safe_float(sp),
            "abs_spearman": abs(safe_float(sp)) if np.isfinite(safe_float(sp)) else np.nan,
            "pearson": safe_float(pr),
            "abs_pearson": abs(safe_float(pr)) if np.isfinite(safe_float(pr)) else np.nan,
            "mean_feature": safe_float(np.nanmean(x[m])),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["abs_spearman", "abs_pearson"], ascending=False)
    return out


def calc_numeric_group_compare(
    df: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
    feature_sources: dict[str, str],
) -> pd.DataFrame:
    rows = []
    sub = df[df[group_col].isin([group_a, group_b])].copy()
    if sub.empty:
        return pd.DataFrame()

    for col in sub.columns:
        if col in PROTECTED_NONFEATURE_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(sub[col]):
            continue
        xa = pd.to_numeric(sub.loc[sub[group_col] == group_a, col], errors="coerce").to_numpy(dtype=float)
        xb = pd.to_numeric(sub.loc[sub[group_col] == group_b, col], errors="coerce").to_numpy(dtype=float)
        xa = xa[np.isfinite(xa)]
        xb = xb[np.isfinite(xb)]
        if len(xa) < 2 or len(xb) < 2:
            continue

        try:
            mw = mannwhitneyu(xa, xb, alternative="two-sided")
            p = mw.pvalue
            stat = mw.statistic
        except Exception:
            p = np.nan
            stat = np.nan

        rows.append({
            "feature": col,
            "source": infer_feature_source(col, feature_sources),
            "group_a": group_a,
            "group_b": group_b,
            "n_group_a": int(len(xa)),
            "n_group_b": int(len(xb)),
            "mean_group_a": safe_float(np.mean(xa)),
            "mean_group_b": safe_float(np.mean(xb)),
            "median_group_a": safe_float(np.median(xa)),
            "median_group_b": safe_float(np.median(xb)),
            "mean_diff_a_minus_b": safe_float(np.mean(xa) - np.mean(xb)),
            "median_diff_a_minus_b": safe_float(np.median(xa) - np.median(xb)),
            "cohen_d_a_minus_b": cohen_d(xa, xb),
            "mannwhitney_u": safe_float(stat),
            "pvalue": safe_float(p),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_cohen_d"] = out["cohen_d_a_minus_b"].abs()
        out = add_fdr(out, p_col="pvalue", out_col="fdr_bh")
        out = out.sort_values(["abs_cohen_d", "pvalue"], ascending=[False, True])
    return out


def fisher_enrichment(
    delta_df: pd.DataFrame,
    tokens_df: pd.DataFrame,
    positive_group: str,
    min_count: int = 2,
) -> pd.DataFrame:
    if tokens_df.empty:
        return pd.DataFrame(columns=["source", "field", "token"])
    pos_set = set(delta_df.loc[delta_df["delta_group"] == positive_group, "drug"].dropna())
    all_drugs = set(delta_df["drug"].dropna())
    rows = []
    for keys, g in tokens_df.groupby(["source", "field", "token"], sort=False):
        source, field, token = keys
        token_drugs = set(g["drug"]) & all_drugs
        if len(token_drugs) < min_count:
            continue
        a = len(token_drugs & pos_set)
        b = len(token_drugs - pos_set)
        c = len(pos_set - token_drugs)
        d = len((all_drugs - pos_set) - token_drugs)
        if a + b < min_count:
            continue
        try:
            odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        except Exception:
            odds, p = np.nan, np.nan
        rows.append({
            "source": source,
            "field": field,
            "token": token,
            f"n_{positive_group}": a,
            "n_with_token": a + b,
            "n_total": len(all_drugs),
            "odds_ratio": safe_float(odds),
            "pvalue": safe_float(p),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["minus_log10_p"] = -np.log10(out["pvalue"].clip(lower=1e-300))
        out = add_fdr(out, p_col="pvalue", out_col="fdr_bh")
        out["minus_log10_fdr"] = -np.log10(out["fdr_bh"].clip(lower=1e-300))
        out = out.sort_values(["pvalue", f"n_{positive_group}"], ascending=[True, False])
    return out


def fisher_enrichment_between_groups(
    delta_df: pd.DataFrame,
    tokens_df: pd.DataFrame,
    group_a: str,
    group_b: str,
    min_count: int = 2,
) -> pd.DataFrame:
    if tokens_df.empty:
        return pd.DataFrame(columns=["source", "field", "token"])
    a_set = set(delta_df.loc[delta_df["delta_group"] == group_a, "drug"].dropna())
    b_set = set(delta_df.loc[delta_df["delta_group"] == group_b, "drug"].dropna())
    rows = []
    for keys, g in tokens_df.groupby(["source", "field", "token"], sort=False):
        source, field, token = keys
        token_drugs = set(g["drug"])
        a = len(token_drugs & a_set)
        b = len(token_drugs & b_set)
        c = len(a_set - token_drugs)
        d = len(b_set - token_drugs)
        if (a + b) < min_count:
            continue
        try:
            odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        except Exception:
            odds, p = np.nan, np.nan
        rows.append({
            "source": source,
            "field": field,
            "token": token,
            f"n_{group_a}": a,
            f"n_{group_b}": b,
            f"n_total_{group_a}": len(a_set),
            f"n_total_{group_b}": len(b_set),
            "odds_ratio": safe_float(odds),
            "pvalue": safe_float(p),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["minus_log10_p"] = -np.log10(out["pvalue"].clip(lower=1e-300))
        out = add_fdr(out, p_col="pvalue", out_col="fdr_bh")
        out["minus_log10_fdr"] = -np.log10(out["fdr_bh"].clip(lower=1e-300))
        out = out.sort_values(["pvalue", f"n_{group_a}"], ascending=[True, False])
    return out


# ----------------------------
# plots
# ----------------------------

def plot_sorted_delta(df: pd.DataFrame, delta_col: str, out_png: Path, title: str) -> None:
    g = df.sort_values(delta_col, ascending=False).copy()
    plt.figure(figsize=(max(10, 0.35 * len(g)), 5.5))
    colors = ["tab:green" if x == "improve" else "tab:red" if x == "worsen" else "tab:gray" for x in g["delta_group"]]
    plt.bar(range(len(g)), g[delta_col].to_numpy(dtype=float), color=colors)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(range(len(g)), g["drug"], rotation=90)
    plt.ylabel(delta_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_top_features(assoc: pd.DataFrame, col: str, out_png: Path, title: str, topk: int = 15) -> None:
    if assoc.empty or col not in assoc.columns:
        return
    g = assoc.head(topk).iloc[::-1]
    plt.figure(figsize=(8, max(4, 0.35 * len(g))))
    plt.barh(g["feature"], g[col].to_numpy(dtype=float))
    plt.xlabel(col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_top_categories(df: pd.DataFrame, out_png: Path, title: str, topk: int = 12, score_col: str = "minus_log10_p") -> None:
    if df.empty or score_col not in df.columns:
        return
    g = df.head(topk).iloc[::-1]
    labels = [f"{f}: {t}" for f, t in zip(g["field"], g["token"])]
    vals = g[score_col].to_numpy(dtype=float)
    plt.figure(figsize=(10, max(4, 0.4 * len(g))))
    plt.barh(labels, vals)
    plt.xlabel(score_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


# ----------------------------
# main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before-summary", required=True, help="baseline ensemble_summary.csv")
    ap.add_argument("--after-summary", required=True, help="final ensemble_summary.csv")
    ap.add_argument("--drug-sample", required=True, help="drug_sample.csv")
    ap.add_argument("--metric", default="R2_ens", help="Metric column to compare. Examples: R2_ens, RMSE_ens, Spearman_ens")
    ap.add_argument("--improve-threshold", type=float, default=None, help="Delta threshold for improve. For R2 default=+0.01; for RMSE default=+0.01 on improvement scale")
    ap.add_argument("--worsen-threshold", type=float, default=None, help="Absolute threshold for worsen. Default mirrors improve-threshold")
    ap.add_argument("--before-weights", default=None)
    ap.add_argument("--after-weights", default=None)
    ap.add_argument("--before-coverage", default=None)
    ap.add_argument("--after-coverage", default=None)
    ap.add_argument("--drug-feature-summary", default=None)
    ap.add_argument("--mechanisms-long", default=None)
    ap.add_argument("--activities-long", default=None)
    ap.add_argument("--targets-long", default=None)
    ap.add_argument("--molecule-resolution", default=None)
    ap.add_argument("--molecules", default=None)
    ap.add_argument("--pubchem-by-drug", default=None)
    ap.add_argument("--cid-resolution", default=None)
    ap.add_argument("--analysis-profile", choices=["interpretable", "all"], default="interpretable")
    ap.add_argument("--outdir", default="analyze_ensemble_delta")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    plots_dir = outdir / "plots"
    ensure_dir(outdir)
    ensure_dir(plots_dir)

    feature_sources: dict[str, str] = {}
    all_tokens = []

    before, src_before = read_ensemble_summary(Path(args.before_summary), args.metric)
    after, src_after = read_ensemble_summary(Path(args.after_summary), args.metric)
    feature_sources.update({"metric_before": src_before["metric"], "metric_after": src_after["metric"]})

    agg, tokens, src = aggregate_drug_sample(Path(args.drug_sample))
    all_tokens.append(tokens)
    feature_sources.update(src)

    df = before.merge(after, on="drug", how="outer", suffixes=("_before", "_after"))

    metric_upper = args.metric.upper()
    higher_better = not (metric_upper.startswith("RMSE") or metric_upper.startswith("MSE"))
    if higher_better:
        df["delta"] = df["metric_after"] - df["metric_before"]
    else:
        df["delta"] = df["metric_before"] - df["metric_after"]

    improve_thr = args.improve_threshold if args.improve_threshold is not None else 0.01
    worsen_thr = args.worsen_threshold if args.worsen_threshold is not None else improve_thr
    df["delta_group"] = df["delta"].apply(lambda x: classify_delta(x, improve_thr, worsen_thr))

    df = df.merge(agg, on="drug", how="left")

    wb, src = read_weights(Path(args.before_weights) if args.before_weights else None, "_before", "weights_before")
    wa, src2 = read_weights(Path(args.after_weights) if args.after_weights else None, "_after", "weights_after")
    cb, src3 = read_coverage(Path(args.before_coverage) if args.before_coverage else None, "_before", "coverage_before")
    ca, src4 = read_coverage(Path(args.after_coverage) if args.after_coverage else None, "_after", "coverage_after")
    for tmp in [wb, wa, cb, ca]:
        if not tmp.empty:
            df = df.merge(tmp, on="drug", how="left")
    feature_sources.update(src)
    feature_sources.update(src2)
    feature_sources.update(src3)
    feature_sources.update(src4)

    external_aggs = []

    dfs_agg, dfs_tok, dfs_src = generic_per_drug_table(
        Path(args.drug_feature_summary) if args.drug_feature_summary else None,
        prefix="dfs",
        source_name="drug_feature_summary",
        token_cols=[
            "molecule_type", "mechanism_of_action_list", "action_type_list",
            "best_standard_type", "best_target_pref_name", "target_organisms_seen",
            "target_type_list", "indication_class"
        ],
        categorical_mode_cols=["molecule_type", "best_standard_type"],
        drop_cols=["canonical_smiles", "standard_inchi", "standard_inchi_key", "target_pref_name_list", "target_accessions_list"],
    )
    mol_agg, mol_tok, mol_src = generic_per_drug_table(
        Path(args.molecules) if args.molecules else None,
        prefix="mol",
        source_name="molecules",
        token_cols=["molecule_type", "indication_class"],
        categorical_mode_cols=["molecule_type", "indication_class"],
        drop_cols=["canonical_smiles", "standard_inchi", "standard_inchi_key", "helm_notation", "pref_name"],
    )
    chemblres_agg, chemblres_tok, chemblres_src = generic_per_drug_table(
        Path(args.molecule_resolution) if args.molecule_resolution else None,
        prefix="chemblres",
        source_name="molecule_resolution",
        numeric_keep=["match_score", "n_candidates_seen"],
        token_cols=["pref_name_search_result"],
        categorical_mode_cols=[],
        drop_cols=["query_used", "molecule_chembl_id"],
    )
    pubchem_agg, pubchem_tok, pubchem_src = generic_per_drug_table(
        Path(args.pubchem_by_drug) if args.pubchem_by_drug else None,
        prefix="pubchem",
        source_name="pubchem_by_drug",
        numeric_keep=[
            "MolecularWeight", "XLogP", "TPSA",
            "HBondDonorCount", "HBondAcceptorCount",
            "RotatableBondCount", "Complexity"
        ],
        token_cols=["status", "MolecularFormula"],
        categorical_mode_cols=["status", "MolecularFormula"],
        drop_cols=["CID", "selected_cid", "all_cids", "SMILES", "ConnectivitySMILES"],
    )
    cidres_agg, cidres_tok, cidres_src = generic_per_drug_table(
        Path(args.cid_resolution) if args.cid_resolution else None,
        prefix="cidres",
        source_name="cid_resolution_results",
        numeric_keep=[],
        token_cols=["status"],
        categorical_mode_cols=["status"],
        drop_cols=["selected_cid", "all_cids", "query_used"],
    )
    mech_agg, mech_tok, mech_src = aggregate_mechanisms_long(Path(args.mechanisms_long) if args.mechanisms_long else None)
    act_agg, act_tok, act_src = aggregate_activities_long(Path(args.activities_long) if args.activities_long else None)
    tgt_agg, tgt_tok, tgt_src = aggregate_targets_long(Path(args.targets_long) if args.targets_long else None)

    for agg_df, tok_df, src_map in [
        (dfs_agg, dfs_tok, dfs_src),
        (mol_agg, mol_tok, mol_src),
        (chemblres_agg, chemblres_tok, chemblres_src),
        (pubchem_agg, pubchem_tok, pubchem_src),
        (cidres_agg, cidres_tok, cidres_src),
        (mech_agg, mech_tok, mech_src),
        (act_agg, act_tok, act_src),
        (tgt_agg, tgt_tok, tgt_src),
    ]:
        if not agg_df.empty:
            external_aggs.append(agg_df)
            df = df.merge(agg_df, on="drug", how="left")
        if not tok_df.empty:
            all_tokens.append(tok_df)
        feature_sources.update(src_map)

    df = df.sort_values("delta", ascending=False)
    df.to_csv(outdir / "drug_delta_table.csv", index=False)
    agg.to_csv(outdir / "drug_sample_aggregated.csv", index=False)

    if external_aggs:
        ext = external_aggs[0].copy()
        for extra in external_aggs[1:]:
            ext = ext.merge(extra, on="drug", how="outer")
    else:
        ext = pd.DataFrame(columns=["drug"])
    ext.to_csv(outdir / "external_feature_aggregated.csv", index=False)

    if all_tokens:
        tokens_all = pd.concat([x for x in all_tokens if not x.empty], ignore_index=True)
        tokens_all = tokens_all.drop_duplicates()
    else:
        tokens_all = pd.DataFrame(columns=["drug", "source", "field", "token"])
    tokens_all.to_csv(outdir / "all_category_tokens.csv", index=False)

    interp_df, feat_audit = filter_numeric_table(df, feature_sources, args.analysis_profile)
    feat_audit.to_csv(outdir / "feature_filter_audit.csv", index=False)

    if not ext.empty:
        interp_ext_cols = [c for c in interp_df.columns if c in ext.columns or c == "drug"]
        ext_interp = ext[[c for c in ext.columns if c == "drug" or c in interp_ext_cols]].copy()
    else:
        ext_interp = ext.copy()
    ext_interp.to_csv(outdir / "external_feature_aggregated_interpretable.csv", index=False)

    tokens_interp, tok_audit = filter_tokens_df(tokens_all, args.analysis_profile)
    tok_audit.to_csv(outdir / "token_filter_audit.csv", index=False)
    tokens_interp.to_csv(outdir / "all_category_tokens_interpretable.csv", index=False)

    kept_feature_sources = {r["feature"]: r["source"] for _, r in feat_audit.loc[feat_audit["keep"]].iterrows()} if not feat_audit.empty else feature_sources

    assoc = calc_numeric_assoc(interp_df, "delta", kept_feature_sources)
    assoc.to_csv(outdir / "numeric_feature_associations.csv", index=False)

    group_cmp = calc_numeric_group_compare(interp_df, "delta_group", "improve", "worsen", kept_feature_sources)
    group_cmp.to_csv(outdir / "numeric_group_comparison_improve_vs_worsen.csv", index=False)

    enrich_improve = fisher_enrichment(df, tokens_interp, "improve", min_count=2)
    enrich_worsen = fisher_enrichment(df, tokens_interp, "worsen", min_count=2)
    enrich_iw = fisher_enrichment_between_groups(df, tokens_interp, "improve", "worsen", min_count=2)
    enrich_wi = fisher_enrichment_between_groups(df, tokens_interp, "worsen", "improve", min_count=2)

    enrich_improve.to_csv(outdir / "category_enrichment_improve.csv", index=False)
    enrich_worsen.to_csv(outdir / "category_enrichment_worsen.csv", index=False)
    enrich_iw.to_csv(outdir / "category_enrichment_improve_vs_worsen.csv", index=False)
    enrich_wi.to_csv(outdir / "category_enrichment_worsen_vs_improve.csv", index=False)

    summary_rows = [{
        "metric": args.metric,
        "analysis_profile": args.analysis_profile,
        "higher_better_raw_metric": higher_better,
        "delta_definition": "after-before" if higher_better else "before-after",
        "n_total_drugs": int(df["drug"].nunique()),
        "n_improve": int((df["delta_group"] == "improve").sum()),
        "n_worsen": int((df["delta_group"] == "worsen").sum()),
        "n_stable": int((df["delta_group"] == "stable").sum()),
        "median_delta": safe_float(df["delta"].median()),
        "mean_delta": safe_float(df["delta"].mean()),
        "improve_threshold": improve_thr,
        "worsen_threshold": worsen_thr,
        "n_token_rows_total_raw": int(len(tokens_all)),
        "n_token_rows_total_filtered": int(len(tokens_interp)),
        "n_numeric_features_raw": int(sum((c not in PROTECTED_NONFEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])) for c in df.columns)),
        "n_numeric_features_filtered": int(sum((c not in PROTECTED_NONFEATURE_COLS and pd.api.types.is_numeric_dtype(interp_df[c])) for c in interp_df.columns)),
    }]
    pd.DataFrame(summary_rows).to_csv(outdir / "analysis_summary.csv", index=False)

    plot_sorted_delta(df, "delta", plots_dir / "drug_delta_sorted.png", f"Drug-level delta ({args.metric})")
    plot_top_features(assoc, "abs_spearman", plots_dir / "top_numeric_features_by_abs_spearman.png", "Top numeric features associated with delta")
    plot_top_features(group_cmp, "abs_cohen_d", plots_dir / "top_numeric_features_improve_vs_worsen_abs_cohen_d.png", "Top numeric features: improve vs worsen")
    plot_top_categories(enrich_improve, plots_dir / "top_enriched_categories_improve.png", "Categories enriched among improved drugs")
    plot_top_categories(enrich_worsen, plots_dir / "top_enriched_categories_worsen.png", "Categories enriched among worsened drugs")
    plot_top_categories(enrich_iw, plots_dir / "top_enriched_categories_improve_vs_worsen.png", "Categories enriched: improve vs worsen")
    plot_top_categories(enrich_wi, plots_dir / "top_enriched_categories_worsen_vs_improve.png", "Categories enriched: worsen vs improve")

    txt = []
    txt.append(f"Metric = {args.metric}")
    txt.append(f"Analysis profile = {args.analysis_profile}")
    txt.append(f"Delta definition = {'after-before' if higher_better else 'before-after'}")
    txt.append(f"Improved / worsened / stable = {(df['delta_group'] == 'improve').sum()} / {(df['delta_group'] == 'worsen').sum()} / {(df['delta_group'] == 'stable').sum()}")
    txt.append(f"Filtered numeric features kept = {sum((c not in PROTECTED_NONFEATURE_COLS and pd.api.types.is_numeric_dtype(interp_df[c])) for c in interp_df.columns)}")
    txt.append(f"Filtered tokens kept = {len(tokens_interp)}")
    if not assoc.empty:
        txt.append("\nTop numeric associations (by |Spearman|):")
        for _, r in assoc.head(10).iterrows():
            txt.append(f"- {r['feature']} [{r['source']}]: spearman={r['spearman']:.3f}, pearson={r['pearson']:.3f}, n={int(r['n'])}")
    if not group_cmp.empty:
        txt.append("\nTop improve-vs-worsen numeric differences (by |Cohen d|):")
        for _, r in group_cmp.head(10).iterrows():
            txt.append(f"- {r['feature']} [{r['source']}]: cohen_d={r['cohen_d_a_minus_b']:.3f}, p={r['pvalue']:.3g}")
    if not enrich_iw.empty:
        txt.append("\nTop enriched categories: improve vs worsen:")
        for _, r in enrich_iw.head(10).iterrows():
            txt.append(f"- {r['field']} | {r['token']} [{r['source']}]: odds_ratio={r['odds_ratio']:.3g}, p={r['pvalue']:.3g}")
    (outdir / "README_results.txt").write_text("\n".join(txt), encoding="utf-8")

    print("[OK] wrote:")
    print(" -", outdir / "analysis_summary.csv")
    print(" -", outdir / "drug_delta_table.csv")
    print(" -", outdir / "drug_sample_aggregated.csv")
    print(" -", outdir / "external_feature_aggregated.csv")
    print(" -", outdir / "external_feature_aggregated_interpretable.csv")
    print(" -", outdir / "numeric_feature_associations.csv")
    print(" -", outdir / "numeric_group_comparison_improve_vs_worsen.csv")
    print(" -", outdir / "category_enrichment_improve.csv")
    print(" -", outdir / "category_enrichment_worsen.csv")
    print(" -", outdir / "category_enrichment_improve_vs_worsen.csv")
    print(" -", outdir / "category_enrichment_worsen_vs_improve.csv")
    print(" -", outdir / "all_category_tokens_interpretable.csv")
    print(" -", outdir / "feature_filter_audit.csv")
    print(" -", outdir / "token_filter_audit.csv")
    print(" -", plots_dir)


if __name__ == "__main__":
    main()
