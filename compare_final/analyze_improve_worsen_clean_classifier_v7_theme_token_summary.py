#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a CLEAN, interpretable feature matrix to classify improved vs worsened drugs,
while retaining the richer summary / association / enrichment outputs from the
older analysis script.

What is fixed compared with the previous clean classifier script:
1) Restores missing summary analyses:
   - analysis_summary.csv
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
   - all_category_tokens_theme_mapped.csv
   - README_results.txt
2) Keeps the clean canonicalized numeric matrix + theme matrix + classifier outputs.
3) Fixes noisy runtime warnings by sanitizing inf / -inf / absurdly large values
   before mean/std/median/iqr/quantile/classifier calculations.
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, mannwhitneyu, pearsonr, spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXTREME_ABS_VALUE = 1e150
BASE_NUMERIC_COLS = [
    "n_rows", "Y_mean", "Y_std", "Y_iqr", "Y_median",
    "AUC_mean", "AUC_std", "curve_RMSE_mean", "curve_RMSE_std",
    "Z_SCORE_mean", "Z_SCORE_std", "Z_SCORE_abs_mean", "TCGA_entropy",
]


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


def clean_numeric_series(s: pd.Series | Iterable) -> pd.Series:
    x = pd.to_numeric(pd.Series(s), errors="coerce").astype(float)
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.where(x.abs() <= EXTREME_ABS_VALUE, np.nan)
    return x


def finite_values(s: pd.Series | Iterable) -> np.ndarray:
    x = clean_numeric_series(s).to_numpy(dtype=float)
    return x[np.isfinite(x)]


def finite_mean(s: pd.Series | Iterable) -> float:
    x = finite_values(s)
    return float(x.mean()) if x.size else np.nan


def finite_std(s: pd.Series | Iterable, ddof: int = 1) -> float:
    x = finite_values(s)
    return float(x.std(ddof=ddof)) if x.size > ddof else np.nan


def finite_median(s: pd.Series | Iterable) -> float:
    x = finite_values(s)
    return float(np.median(x)) if x.size else np.nan


def finite_max(s: pd.Series | Iterable) -> float:
    x = finite_values(s)
    return float(x.max()) if x.size else np.nan


def finite_quantile(s: pd.Series | Iterable, q: float) -> float:
    x = finite_values(s)
    return float(np.quantile(x, q)) if x.size else np.nan


def shannon_entropy(values: pd.Series) -> float:
    s = values.dropna().astype(str)
    if s.empty:
        return float("nan")
    p = s.value_counts(normalize=True)
    return float(-(p * np.log2(p)).sum())


def iqr(values: pd.Series) -> float:
    x = finite_values(values)
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, 0.75) - np.quantile(x, 0.25))


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = finite_values(x)
    y = finite_values(y)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    sx = np.var(x, ddof=1)
    sy = np.var(y, ddof=1)
    pooled = ((len(x) - 1) * sx + (len(y) - 1) * sy) / (len(x) + len(y) - 2)
    if not np.isfinite(pooled) or pooled <= 0:
        return np.nan
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled))


def add_fdr(df: pd.DataFrame, p_col: str = "pvalue", out_col: str = "fdr_bh") -> pd.DataFrame:
    if df.empty or p_col not in df.columns:
        return df
    p = clean_numeric_series(df[p_col]).to_numpy(dtype=float)
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


def classify_delta(delta: float, improve_thr: float, worsen_thr: float) -> str:
    if pd.isna(delta):
        return "missing"
    if delta >= improve_thr:
        return "improve"
    if delta <= -abs(worsen_thr):
        return "worsen"
    return "stable"


def parse_threshold_grid(s: str | None, default_threshold: float) -> list[float]:
    if s is None or not str(s).strip():
        vals = [default_threshold]
    else:
        vals = []
        for part in str(s).split(','):
            part = part.strip()
            if not part:
                continue
            try:
                vals.append(abs(float(part)))
            except Exception:
                continue
        if not vals:
            vals = [default_threshold]
    vals = sorted(set(v for v in vals if np.isfinite(v) and v > 0))
    return vals or [default_threshold]


def effect_sign(x: float) -> int:
    if not np.isfinite(x):
        return 0
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def safe_log2_or(a: int, b: int, c: int, d: int) -> float:
    num = (a + 0.5) * (d + 0.5)
    den = (b + 0.5) * (c + 0.5)
    if den <= 0 or num <= 0:
        return np.nan
    return float(np.log2(num / den))


def build_delta_groups_from_threshold(df: pd.DataFrame, thr: float) -> pd.Series:
    return clean_numeric_series(df['delta']).apply(lambda x: classify_delta(x, thr, thr))


def summarize_annotation_burden(df: pd.DataFrame, tokens: pd.DataFrame, token_audit: pd.DataFrame, theme_mat: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    out = df[['drug', 'delta', 'delta_group']].drop_duplicates(subset=['drug']).copy()
    if numeric_cols:
        present = pd.DataFrame({'drug': out['drug']})
        for c in numeric_cols:
            if c in df.columns:
                present[c] = clean_numeric_series(out['drug'].map(df.drop_duplicates(subset=['drug']).set_index('drug')[c]))
        num_cols_here = [c for c in present.columns if c != 'drug']
        out['n_numeric_nonmissing'] = present[num_cols_here].notna().sum(axis=1).astype(int) if num_cols_here else 0
    else:
        out['n_numeric_nonmissing'] = 0
    raw_tok = tokens.groupby('drug').size().rename('n_raw_tokens').reset_index() if not tokens.empty else pd.DataFrame(columns=['drug', 'n_raw_tokens'])
    mapped_tok = token_audit.groupby('drug').size().rename('n_theme_mapped_tokens').reset_index() if not token_audit.empty else pd.DataFrame(columns=['drug', 'n_theme_mapped_tokens'])
    out = out.merge(raw_tok, on='drug', how='left')
    out = out.merge(mapped_tok, on='drug', how='left')
    out['n_raw_tokens'] = clean_numeric_series(out['n_raw_tokens']).fillna(0).astype(int)
    out['n_theme_mapped_tokens'] = clean_numeric_series(out['n_theme_mapped_tokens']).fillna(0).astype(int)
    theme_cols = [c for c in theme_mat.columns if c.startswith('theme_')] if not theme_mat.empty else []
    if theme_cols:
        tm = theme_mat[['drug'] + theme_cols].copy()
        for c in theme_cols:
            tm[c] = clean_numeric_series(tm[c]).fillna(0)
        tm['n_theme_features_present'] = tm[theme_cols].sum(axis=1)
        out = out.merge(tm[['drug', 'n_theme_features_present']], on='drug', how='left')
    else:
        out['n_theme_features_present'] = 0
    out['n_theme_features_present'] = clean_numeric_series(out['n_theme_features_present']).fillna(0).astype(int)
    out = out.sort_values(['delta_group', 'drug']).reset_index(drop=True)
    return out


def read_ensemble_summary(path: Path, metric: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    drug_col = first_existing(df, ["drug", "Drug"])
    if drug_col is None:
        raise ValueError(f"No drug column in {path}")
    metric_col = detect_metric_column(df, metric)
    out = df[[drug_col, metric_col]].copy()
    out.columns = ["drug", "metric"]
    out["drug"] = out["drug"].map(norm_drug)
    out["metric"] = clean_numeric_series(out["metric"])
    return out.dropna(subset=["drug"]).drop_duplicates("drug")


def aggregate_drug_sample(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    for c in num_map.values():
        if c is not None:
            df[c] = clean_numeric_series(df[c])

    tcga_col = first_existing(df, ["TCGA_DESC", "tcga_desc", "tissue", "Tissue"])
    target_col = first_existing(df, ["PUTATIVE_TARGET", "putative_target", "target", "Target"])
    pathway_col = first_existing(df, ["PATHWAY_NAME", "pathway_name", "pathway", "Pathway"])

    rows, tok_rows = [], []
    for drug, g in df.groupby("drug", sort=False):
        row = {"drug": drug, "n_rows": int(len(g))}
        if num_map["Y"] is not None:
            s = g[num_map["Y"]]
            row["Y_mean"] = finite_mean(s)
            row["Y_std"] = finite_std(s, ddof=1)
            row["Y_iqr"] = iqr(s)
            row["Y_median"] = finite_median(s)
        if num_map["AUC"] is not None:
            s = g[num_map["AUC"]]
            row["AUC_mean"] = finite_mean(s)
            row["AUC_std"] = finite_std(s, ddof=1)
        if num_map["RMSE_curve"] is not None:
            s = g[num_map["RMSE_curve"]]
            row["curve_RMSE_mean"] = finite_mean(s)
            row["curve_RMSE_std"] = finite_std(s, ddof=1)
        if num_map["Z_SCORE"] is not None:
            s = g[num_map["Z_SCORE"]]
            row["Z_SCORE_mean"] = finite_mean(s)
            row["Z_SCORE_std"] = finite_std(s, ddof=1)
            row["Z_SCORE_abs_mean"] = finite_mean(clean_numeric_series(s).abs())
        if tcga_col is not None:
            row["TCGA_entropy"] = shannon_entropy(g[tcga_col])
        rows.append(row)

        if target_col is not None:
            toks = set()
            for v in g[target_col].dropna().astype(str):
                toks.update(split_multi_value(v))
            for tok in sorted(t for t in toks if t):
                tok_rows.append({"drug": drug, "source": "drug_sample", "field": "PUTATIVE_TARGET", "token": tok})
        if pathway_col is not None:
            toks = set()
            for v in g[pathway_col].dropna().astype(str):
                toks.update(split_multi_value(v))
            for tok in sorted(t for t in toks if t):
                tok_rows.append({"drug": drug, "source": "drug_sample", "field": "PATHWAY_NAME", "token": tok})
    return pd.DataFrame(rows), pd.DataFrame(tok_rows)


CANONICAL_NUMERIC_SPECS = [
    ("max_phase", [("drug_feature_summary", "max_phase"), ("molecules", "max_phase")]),
    ("oral", [("drug_feature_summary", "oral"), ("molecules", "oral")]),
    ("black_box_warning", [("drug_feature_summary", "black_box_warning"), ("molecules", "black_box_warning")]),
    ("first_approval", [("drug_feature_summary", "first_approval"), ("molecules", "first_approval")]),
    ("full_mwt", [("drug_feature_summary", "full_mwt"), ("molecules", "full_mwt"), ("pubchem", "MolecularWeight")]),
    ("alogp_xlogp", [("drug_feature_summary", "alogp"), ("molecules", "alogp"), ("pubchem", "XLogP")]),
    ("psa_tpsa", [("drug_feature_summary", "psa"), ("molecules", "psa"), ("pubchem", "TPSA")]),
    ("hba", [("drug_feature_summary", "hba"), ("molecules", "hba"), ("pubchem", "HBondAcceptorCount")]),
    ("hbd", [("drug_feature_summary", "hbd"), ("molecules", "hbd"), ("pubchem", "HBondDonorCount")]),
    ("rtb", [("drug_feature_summary", "rtb"), ("molecules", "rtb"), ("pubchem", "RotatableBondCount")]),
    ("aromatic_rings", [("drug_feature_summary", "aromatic_rings"), ("molecules", "aromatic_rings")]),
    ("heavy_atoms", [("drug_feature_summary", "heavy_atoms"), ("molecules", "heavy_atoms")]),
    ("qed_weighted", [("drug_feature_summary", "qed_weighted"), ("molecules", "qed_weighted")]),
    ("best_pchembl", [("drug_feature_summary", "best_pchembl")]),
    ("median_pchembl", [("drug_feature_summary", "median_pchembl")]),
    ("best_standard_value", [("drug_feature_summary", "best_standard_value")]),
    ("ac50_best_pchembl", [("drug_feature_summary", "ac50_best_pchembl")]),
    ("ec50_best_pchembl", [("drug_feature_summary", "ec50_best_pchembl")]),
    ("ic50_best_pchembl", [("drug_feature_summary", "ic50_best_pchembl")]),
    ("kd_best_pchembl", [("drug_feature_summary", "kd_best_pchembl")]),
    ("ki_best_pchembl", [("drug_feature_summary", "ki_best_pchembl")]),
    ("act_pchembl_mean", [("activities", "pchembl_mean")]),
    ("act_pchembl_median", [("activities", "pchembl_median")]),
    ("act_pchembl_std", [("activities", "pchembl_std")]),
    ("mech_direct_interaction_mean", [("mechanisms", "direct_interaction_mean")]),
    ("mech_disease_efficacy_mean", [("mechanisms", "disease_efficacy_mean")]),
    ("mech_molecular_mechanism_mean", [("mechanisms", "molecular_mechanism_mean")]),
]

# Theme rules are intentionally ordered from MORE SPECIFIC to MORE GENERAL.
# This makes the theme layer feel like a structured convergence from raw tokens,
# instead of immediately collapsing many kinase / receptor tokens into one giant bucket.
THEME_RULES = [
    # --- receptor tyrosine kinase families / subfamilies ---
    ("ERBB_EGFR_FAMILY_RTK", [
        r"\bEGFR\b", r"\bERBB\b", r"\bERBB-?[1234]\b", r"\bHER2\b", r"\bHER3\b", r"\bHER4\b",
        r"\bEPIDERMAL GROWTH FACTOR RECEPTOR\b", r"\bRECEPTOR TYROSINE-PROTEIN KINASE ERBB-?2\b"
    ]),
    ("MET_HGF_RTK", [
        r"\bC-?MET\b", r"\b\bMET\b", r"\bHGFR\b", r"\bHEPATOCYTE GROWTH FACTOR RECEPTOR\b", r"\bHGF RECEPTOR\b"
    ]),
    ("EPH_RECEPTOR_SIGNALING", [
        r"\bEPHRIN\b", r"\bEPH[A-Z0-9-]*\b", r"\bEPHA[0-9]+\b", r"\bEPHB[0-9]+\b", r"\bEPHRIN TYPE-[AB] RECEPTOR\b"
    ]),
    ("VEGFR_FGFR_PDGFR_KIT_FLT_RTK", [
        r"\bVEGFR\b", r"\bFGFR\b", r"\bPDGFR\b", r"\bKIT\b", r"\bFLT1\b", r"\bFLT3\b", r"\bFLT4\b",
        r"\bRET\b", r"\bAXL\b", r"\bALK\b"
    ]),
    ("RTK_SIGNALING_BROAD", [
        r"\bRTK SIGNALING\b", r"\bRECEPTOR TYROSINE KINASE\b", r"\bTYROSINE[- ]PROTEIN KINASE RECEPTOR\b"
    ]),

    # --- MAPK and kinase signaling, refined from raw kinase tokens ---
    ("JNK_P38_STRESS_MAPK", [
        r"\bJNK\b", r"\bMAPK8\b", r"\bMAPK9\b", r"\bMAPK10\b", r"\bMITOGEN-ACTIVATED PROTEIN KINASE 10\b",
        r"\bP38\b", r"\bMAPK11\b", r"\bMAPK12\b", r"\bMAPK13\b", r"\bMAPK14\b", r"\bSTRESS[- ]ACTIVATED\b"
    ]),
    ("MAP2K_MEK_ERK_RAF_CASCADE", [
        r"\bMAP2K[0-9]+\b", r"\bMEK[1-9]?\b", r"\bERK[1-9]?\b", r"\bRAF\b", r"\bBRAF\b", r"\bCRAF\b",
        r"\bMEK5\b", r"\bMITOGEN-ACTIVATED PROTEIN KINASE KINASE\b", r"\bDUAL SPECIFICITY MITOGEN-ACTIVATED PROTEIN KINASE KINASE\b"
    ]),
    ("PI3K_AKT_MTOR", [
        r"\bPI3K\b", r"\bPI 3\b", r"\bPI3-KINASE\b", r"\bPIK3[A-Z0-9]*\b", r"\bAKT\b", r"\bAKT[123]\b",
        r"\bMTOR\b", r"\bPDPK1\b", r"\bRICTOR\b", r"\bRAPTOR\b"
    ]),
    ("SRC_LCK_NONRECEPTOR_TK", [
        r"\bLCK\b", r"\bSRC\b", r"\bFYN\b", r"\bLYN\b", r"\bYES1\b", r"\bHCK\b", r"\bFGR\b", r"\bBLK\b", r"\bFRK\b", r"\bSYK\b"
    ]),
    ("SER_THR_KINASE_SIGNALING", [
        r"\bGLYCOGEN SYNTHASE KINASE\b", r"\bGSK3\b", r"\bGSK-?3\b", r"\bCASEIN KINASE\b", r"\bCSNK1\b", r"\bCK1\b",
        r"\bHIPK\b", r"\bHOMEODOMAIN-INTERACTING PROTEIN KINASE\b", r"\bCYCLIN-G-ASSOCIATED KINASE\b", r"\bGAK\b", r"\bDYRK\b"
    ]),

    # --- cell-cycle / checkpoint / mitosis ---
    ("PARP_DNA_REPAIR_CHECKPOINT", [
        r"\bPARP\b", r"\bDNA REPAIR\b", r"\bDNA DAMAGE\b", r"\bDDR\b", r"\bATR\b", r"\bATM\b",
        r"\bCHK1\b", r"\bCHK2\b", r"\bCHEK1\b", r"\bCHEK2\b", r"\bWEE1\b"
    ]),
    ("CELL_CYCLE_MITOSIS_KINASE", [
        r"\bMITOSIS\b", r"\bCELL CYCLE\b", r"\bAURORA\b", r"\bPLK\b", r"\bCDK\b", r"\bCDC7\b",
        r"\bSPINDLE\b", r"\bTUBULIN\b", r"\bKIF11\b"
    ]),

    # --- epigenetics / apoptosis / genome integrity ---
    ("CHROMATIN_EPIGENETIC", [
        r"\bCHROMATIN\b", r"\bEPIGEN", r"\bBET\b", r"\bBRD[234]\b", r"\bHDAC\b", r"\bEZH2\b", r"\bDNMT\b", r"\bHISTONE\b",
        r"\bKDM[0-9A-Z]*\b", r"\bSMARCA[24]\b", r"\bBRM\b", r"\bBRG1\b"
    ]),
    ("APOPTOSIS_BCL2", [
        r"\bBCL2\b", r"\bBCL-2\b", r"\bMCL1\b", r"\bBCLXL\b", r"\bBCL-XL\b", r"\bBAX\b", r"\bBAK\b", r"\bAPOPTOSIS\b"
    ]),
    ("DNA_REPLICATION_GENOME_INTEGRITY", [
        r"\bDNA REPLICATION\b", r"\bGENOME INTEGRITY\b", r"\bREPLICATION\b", r"\bPOLA\b", r"\bPOLE\b", r"\bMCM[0-9]+\b"
    ]),

    # --- membrane proteins / transport ---
    ("ION_CHANNEL_TRANSPORTER", [
        r"\bCHANNEL\b", r"\bPOTASSIUM CHANNEL\b", r"\bSODIUM CHANNEL\b", r"\bCALCIUM CHANNEL\b",
        r"\bKCNH2\b", r"\bHERG\b", r"\bTRANSPORTER\b", r"\bEXPORT PUMP\b", r"\bPUMP\b", r"\bABCB11\b", r"\bBILE SALT EXPORT PUMP\b", r"\bSLC[0-9A-Z]+\b"
    ]),

    # --- broad fallback themes retained for backwards interpretability ---
    ("MAPK_ERK", [r"\bMAPK\b", r"\bERK\b", r"\bMEK\b", r"\bRAF\b", r"\bBRAF\b"]),
    ("EGFR_RTK", [r"\bEGFR\b", r"\bERBB\b", r"\bHER2\b", r"\bRTK SIGNALING\b", r"\bVEGFR\b", r"\bFGFR\b", r"\bPDGFR\b", r"\bKIT\b", r"\bFLT1\b", r"\bFLT4\b"]),
    ("PROTEOSTASIS", [r"\bPROTEASOME\b", r"\bDEGRADATION\b", r"\bPROTEIN STABILITY\b", r"\bUBIQUITIN\b", r"\bHSP90\b"]),
]
ALLOWED_TOKEN_FIELDS = {
    ("drug_sample", "PUTATIVE_TARGET"),
    ("drug_sample", "PATHWAY_NAME"),
    ("drug_feature_summary", "mechanism_of_action_list"),
    ("drug_feature_summary", "action_type_list"),
    ("mechanisms", "mechanism_of_action"),
    ("mechanisms", "action_type"),
    ("targets", "single_protein_pref_name"),
}
NOISY_TOKEN_PATTERNS = [r"^INHIBITOR$", r"^AGONIST$", r"^ANTAGONIST$", r"^OTHER$", r"^HOMO SAPIENS$"]


def read_optional(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)


def aggregate_drug_feature_summary(path: Optional[Path]) -> Optional[pd.DataFrame]:
    df = read_optional(path)
    if df is None:
        return None
    drug_col = first_existing(df, ["drug", "Drug", "original_drug"])
    if drug_col is None:
        return None
    df = df.copy()
    df["drug"] = df[drug_col].map(norm_drug)
    keep = [c for c in ["drug", "molecule_type", "mechanism_of_action_list", "action_type_list", "indication_class"] if c in df.columns] + [
        c for c in df.columns if c in {
            "max_phase", "oral", "black_box_warning", "first_approval", "full_mwt", "alogp", "psa", "hba", "hbd",
            "rtb", "aromatic_rings", "heavy_atoms", "qed_weighted", "best_pchembl", "median_pchembl",
            "best_standard_value", "ac50_best_pchembl", "ec50_best_pchembl", "ic50_best_pchembl", "kd_best_pchembl", "ki_best_pchembl"
        }
    ]
    out = df[keep].copy()
    for c in out.columns:
        if c != "drug" and c not in {"molecule_type", "mechanism_of_action_list", "action_type_list", "indication_class"}:
            out[c] = clean_numeric_series(out[c])
    return out


def aggregate_molecules(path: Optional[Path]) -> Optional[pd.DataFrame]:
    df = read_optional(path)
    if df is None:
        return None
    drug_col = first_existing(df, ["drug", "Drug", "original_drug"])
    if drug_col is None:
        return None
    df = df.copy()
    df["drug"] = df[drug_col].map(norm_drug)
    keep = [c for c in ["drug", "molecule_type", "indication_class"] if c in df.columns] + [
        c for c in df.columns if c in {
            "max_phase", "oral", "black_box_warning", "first_approval", "full_mwt", "alogp", "psa", "hba", "hbd",
            "rtb", "aromatic_rings", "heavy_atoms", "qed_weighted"
        }
    ]
    out = df[keep].copy()
    for c in out.columns:
        if c != "drug" and c not in {"molecule_type", "indication_class"}:
            out[c] = clean_numeric_series(out[c])
    return out


def aggregate_pubchem(path: Optional[Path]) -> Optional[pd.DataFrame]:
    df = read_optional(path)
    if df is None:
        return None
    drug_col = first_existing(df, ["original_drug", "drug", "Drug"])
    if drug_col is None:
        return None
    df = df.copy()
    df["drug"] = df[drug_col].map(norm_drug)
    keep = ["drug"] + [c for c in df.columns if c in {
        "MolecularWeight", "XLogP", "TPSA", "HBondAcceptorCount", "HBondDonorCount", "RotatableBondCount", "Complexity"
    }]
    out = df[keep].copy()
    for c in out.columns:
        if c != "drug":
            out[c] = clean_numeric_series(out[c])
    return out


def aggregate_mechanisms(path: Optional[Path]) -> tuple[Optional[pd.DataFrame], pd.DataFrame]:
    df = read_optional(path)
    if df is None:
        return None, pd.DataFrame(columns=["drug", "source", "field", "token"])
    drug_col = first_existing(df, ["drug", "Drug"])
    if drug_col is None:
        return None, pd.DataFrame(columns=["drug", "source", "field", "token"])
    df = df.copy()
    df["drug"] = df[drug_col].map(norm_drug)
    for c in ["direct_interaction", "disease_efficacy", "molecular_mechanism"]:
        if c in df.columns:
            df[c] = clean_numeric_series(df[c])

    rows, toks = [], []
    for drug, g in df.groupby("drug", sort=False):
        rows.append({
            "drug": drug,
            "direct_interaction_mean": finite_mean(g["direct_interaction"]) if "direct_interaction" in g.columns else np.nan,
            "disease_efficacy_mean": finite_mean(g["disease_efficacy"]) if "disease_efficacy" in g.columns else np.nan,
            "molecular_mechanism_mean": finite_mean(g["molecular_mechanism"]) if "molecular_mechanism" in g.columns else np.nan,
        })
        for field in ["mechanism_of_action", "action_type"]:
            if field in g.columns:
                vals = set(g[field].dropna().astype(str).str.strip())
                for tok in sorted(t for t in vals if t and t.lower() != "nan"):
                    toks.append({"drug": drug, "source": "mechanisms", "field": field, "token": tok})
    return pd.DataFrame(rows), pd.DataFrame(toks)


def aggregate_activities(path: Optional[Path]) -> tuple[Optional[pd.DataFrame], pd.DataFrame]:
    df = read_optional(path)
    if df is None:
        return None, pd.DataFrame(columns=["drug", "source", "field", "token"])
    drug_col = first_existing(df, ["drug", "Drug"])
    if drug_col is None:
        return None, pd.DataFrame(columns=["drug", "source", "field", "token"])
    df = df.copy()
    df["drug"] = df[drug_col].map(norm_drug)
    for c in ["pchembl_value", "standard_value"]:
        if c in df.columns:
            df[c] = clean_numeric_series(df[c])
    rows = []
    for drug, g in df.groupby("drug", sort=False):
        row = {"drug": drug}
        if "pchembl_value" in g.columns:
            s = g["pchembl_value"]
            row["pchembl_mean"] = finite_mean(s)
            row["pchembl_median"] = finite_median(s)
            row["pchembl_std"] = finite_std(s, ddof=1)
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(columns=["drug", "source", "field", "token"])


def aggregate_targets(path: Optional[Path]) -> tuple[Optional[pd.DataFrame], pd.DataFrame]:
    df = read_optional(path)
    if df is None:
        return None, pd.DataFrame(columns=["drug", "source", "field", "token"])
    drug_col = first_existing(df, ["drug", "Drug"])
    if drug_col is None:
        return None, pd.DataFrame(columns=["drug", "source", "field", "token"])
    df = df.copy()
    df["drug"] = df[drug_col].map(norm_drug)
    rows, toks = [], []
    for drug, g in df.groupby("drug", sort=False):
        row = {"drug": drug}
        if "target_chembl_id" in g.columns:
            row["targets_n_unique_target_chembl_id"] = int(g["target_chembl_id"].dropna().astype(str).nunique())
        if "pref_name" in g.columns:
            row["targets_n_unique_pref_name"] = int(g["pref_name"].dropna().astype(str).nunique())
        rows.append(row)
        if {"target_type", "pref_name"}.issubset(g.columns):
            gg = g[g["target_type"].astype(str).str.upper() == "SINGLE PROTEIN"]
            vals = set(gg["pref_name"].dropna().astype(str).str.strip())
            for tok in sorted(t for t in vals if t and t.lower() != "nan"):
                toks.append({"drug": drug, "source": "targets", "field": "single_protein_pref_name", "token": tok})
    return pd.DataFrame(rows), pd.DataFrame(toks)


def map_token_to_theme(tok: str) -> Optional[str]:
    u = re.sub(r"\s+", " ", str(tok)).strip().upper()
    if not u:
        return None
    if any(re.search(p, u) for p in NOISY_TOKEN_PATTERNS):
        return None
    for theme, patterns in THEME_RULES:
        for pat in patterns:
            try:
                if re.search(pat, u):
                    return theme
            except re.error:
                pass
    return None




def normalize_raw_token(tok: str) -> Optional[str]:
    u = re.sub(r"\s+", " ", str(tok)).strip().upper()
    if not u or u == "NAN":
        return None
    if any(re.search(p, u) for p in NOISY_TOKEN_PATTERNS):
        return None
    return u


def safe_token_slug(s: str, max_len: int = 72) -> str:
    slug = re.sub(r"[^A-Z0-9]+", "_", str(s).upper()).strip("_")
    if not slug:
        slug = "TOKEN"
    return slug[:max_len]


def make_unique_feature_map(tokens: list[str], prefix: str) -> dict[str, str]:
    mapping = {}
    used = set()
    for tok in tokens:
        base = f"{prefix}{safe_token_slug(tok)}"
        name = base
        i = 2
        while name in used:
            name = f"{base}__{i}"
            i += 1
        used.add(name)
        mapping[tok] = name
    return mapping


def build_theme_presence(tokens_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if tokens_df.empty:
        return pd.DataFrame(columns=["drug"]), pd.DataFrame(columns=["drug", "source", "field", "token", "theme"])
    keep = []
    for _, r in tokens_df.iterrows():
        key = (str(r["source"]), str(r["field"]))
        if key not in ALLOWED_TOKEN_FIELDS:
            continue
        theme = map_token_to_theme(str(r["token"]))
        if theme is None:
            continue
        keep.append({"drug": r["drug"], "source": r["source"], "field": r["field"], "token": r["token"], "theme": theme})
    mapped = pd.DataFrame(keep)
    if mapped.empty:
        return pd.DataFrame(columns=["drug"]), mapped
    pres = mapped[["drug", "theme"]].drop_duplicates()
    pres["value"] = 1
    mat = pres.pivot(index="drug", columns="theme", values="value").fillna(0).reset_index()
    mat.columns = ["drug"] + [f"theme_{c}" for c in mat.columns[1:]]
    return mat, mapped




def build_raw_token_presence(tokens_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if tokens_df.empty:
        empty_mat = pd.DataFrame(columns=["drug"])
        empty_audit = pd.DataFrame(columns=["drug", "source", "field", "token_original", "token_norm", "feature"])
        empty_catalog = pd.DataFrame(columns=["feature", "raw_token", "n_drugs_supporting_token", "n_rows", "sources", "fields", "example_token"])
        return empty_mat, empty_audit, empty_catalog

    rows = []
    for _, r in tokens_df.iterrows():
        key = (str(r["source"]), str(r["field"]))
        if key not in ALLOWED_TOKEN_FIELDS:
            continue
        tok_orig = re.sub(r"\s+", " ", str(r["token"])).strip()
        tok_norm = normalize_raw_token(tok_orig)
        if tok_norm is None:
            continue
        rows.append({
            "drug": r["drug"],
            "source": r["source"],
            "field": r["field"],
            "token_original": tok_orig,
            "token_norm": tok_norm,
        })

    audit = pd.DataFrame(rows)
    if audit.empty:
        empty_mat = pd.DataFrame(columns=["drug"])
        empty_catalog = pd.DataFrame(columns=["feature", "raw_token", "n_drugs_supporting_token", "n_rows", "sources", "fields", "example_token"])
        return empty_mat, audit, empty_catalog

    feature_map = make_unique_feature_map(sorted(audit["token_norm"].dropna().astype(str).unique().tolist()), prefix="rawtok_")
    audit["feature"] = audit["token_norm"].map(feature_map)

    pres = audit[["drug", "feature"]].drop_duplicates().copy()
    pres["value"] = 1
    mat = pres.pivot(index="drug", columns="feature", values="value").fillna(0).reset_index()
    mat.columns = ["drug"] + list(mat.columns[1:])

    catalog_rows = []
    for tok, g in audit.groupby("token_norm", sort=True):
        catalog_rows.append({
            "feature": feature_map.get(tok),
            "raw_token": tok,
            "n_drugs_supporting_token": int(g["drug"].nunique()),
            "n_rows": int(len(g)),
            "sources": " | ".join(sorted(g["source"].dropna().astype(str).unique().tolist())),
            "fields": " | ".join(sorted(g["field"].dropna().astype(str).unique().tolist())),
            "example_token": sorted(g["token_original"].dropna().astype(str).unique().tolist())[0] if len(g) else tok,
        })
    catalog = pd.DataFrame(catalog_rows).sort_values(["n_drugs_supporting_token", "raw_token"], ascending=[False, True])
    return mat, audit, catalog


def build_raw_token_theme_catalog(raw_token_catalog: pd.DataFrame) -> pd.DataFrame:
    if raw_token_catalog.empty:
        return pd.DataFrame(columns=["feature", "raw_token", "assigned_theme", "theme_mapping_status"])
    out = raw_token_catalog[["feature", "raw_token"]].copy()
    out["assigned_theme"] = out["raw_token"].astype(str).map(lambda x: map_token_to_theme(x) or "UNMAPPED")
    out["theme_mapping_status"] = np.where(out["assigned_theme"].eq("UNMAPPED"), "unmapped", "mapped")
    return out


def raw_token_univariate(df: pd.DataFrame, raw_token_cols: list[str], raw_token_catalog: pd.DataFrame, group_col: str = 'delta_group', group_a: str = 'improve', group_b: str = 'worsen') -> pd.DataFrame:
    rows = []
    sub = df[df[group_col].isin([group_a, group_b])].copy()
    group_a_set = set(sub.loc[sub[group_col] == group_a, 'drug'])
    group_b_set = set(sub.loc[sub[group_col] == group_b, 'drug'])
    eligible_drugs = group_a_set | group_b_set
    token_lookup = dict(zip(raw_token_catalog["feature"], raw_token_catalog["raw_token"])) if not raw_token_catalog.empty else {}
    for c in raw_token_cols:
        token_drugs = set(sub.loc[clean_numeric_series(sub[c]).fillna(0) > 0, 'drug'])
        a = len(token_drugs & group_a_set)
        b = len(token_drugs & group_b_set)
        c0 = len(group_a_set - token_drugs)
        d = len(group_b_set - token_drugs)
        if a + b < 2:
            continue
        try:
            odds, p = fisher_exact([[a, b], [c0, d]], alternative='two-sided')
        except Exception:
            odds, p = np.nan, np.nan
        log2_or = safe_log2_or(a, b, c0, d)
        rows.append({
            'feature': c,
            'raw_token': token_lookup.get(c, c.replace('rawtok_', '')),
            f'n_{group_a}_with_token': a,
            f'n_{group_b}_with_token': b,
            'n_drugs_supporting_token': a + b,
            'support_rate_total': safe_float((a + b) / len(eligible_drugs)) if eligible_drugs else np.nan,
            f'support_rate_{group_a}': safe_float(a / len(group_a_set)) if group_a_set else np.nan,
            f'support_rate_{group_b}': safe_float(b / len(group_b_set)) if group_b_set else np.nan,
            'odds_ratio': safe_float(odds),
            'log2_odds_ratio': log2_or,
            'effect_direction': 'enriched_in_' + group_a if np.isfinite(log2_or) and log2_or > 0 else ('enriched_in_' + group_b if np.isfinite(log2_or) and log2_or < 0 else 'neutral_or_missing'),
            'pvalue': safe_float(p),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_fdr(out)
        out['minus_log10_p'] = -np.log10(clean_numeric_series(out['pvalue']).clip(lower=1e-300))
        out = out.sort_values(['pvalue', 'n_drugs_supporting_token', 'feature'], ascending=[True, False, True])
    return out


def raw_token_enrichment(delta_df: pd.DataFrame, raw_token_mat: pd.DataFrame, raw_token_catalog: pd.DataFrame, positive_group: str, opposite_group: Optional[str] = None) -> pd.DataFrame:
    if raw_token_mat.empty:
        return pd.DataFrame(columns=["source", "field", "token"])
    if opposite_group is None:
        pos_set = set(delta_df.loc[delta_df["delta_group"] == positive_group, "drug"])
        all_set = set(delta_df["drug"])
    else:
        pos_set = set(delta_df.loc[delta_df["delta_group"] == positive_group, "drug"])
        all_set = pos_set | set(delta_df.loc[delta_df["delta_group"] == opposite_group, "drug"])
    lookup = dict(zip(raw_token_catalog["feature"], raw_token_catalog["raw_token"])) if not raw_token_catalog.empty else {}
    rows = []
    token_cols = [c for c in raw_token_mat.columns if c.startswith("rawtok_")]
    for c in token_cols:
        token = lookup.get(c, c.replace("rawtok_", ""))
        token_drugs = set(raw_token_mat.loc[clean_numeric_series(raw_token_mat[c]).fillna(0) > 0, "drug"]) & all_set
        if len(token_drugs) < 2:
            continue
        if opposite_group is None:
            a = len(token_drugs & pos_set)
            b = len(token_drugs - pos_set)
            c0 = len(pos_set - token_drugs)
            d = len((all_set - pos_set) - token_drugs)
        else:
            neg_set = set(delta_df.loc[delta_df["delta_group"] == opposite_group, "drug"])
            a = len(token_drugs & pos_set)
            b = len(token_drugs & neg_set)
            c0 = len(pos_set - token_drugs)
            d = len(neg_set - token_drugs)
        if (a + b) < 2:
            continue
        try:
            odds, p = fisher_exact([[a, b], [c0, d]], alternative="greater")
        except Exception:
            odds, p = np.nan, np.nan
        row = {
            "source": "raw_token",
            "field": "raw_token",
            "token": token,
            "feature": c,
            "odds_ratio": safe_float(odds),
            "pvalue": safe_float(p),
        }
        if opposite_group is None:
            row[f"n_{positive_group}"] = a
            row["n_with_token"] = a + b
            row["n_total"] = len(all_set)
        else:
            row[f"n_{positive_group}"] = a
            row[f"n_{opposite_group}"] = b
            row[f"n_total_{positive_group}"] = len(pos_set)
            row[f"n_total_{opposite_group}"] = len(set(delta_df.loc[delta_df["delta_group"] == opposite_group, "drug"]))
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_fdr(out)
        out["minus_log10_p"] = -np.log10(clean_numeric_series(out["pvalue"]).clip(lower=1e-300))
        out["minus_log10_fdr"] = -np.log10(clean_numeric_series(out["fdr_bh"]).clip(lower=1e-300))
        sort_col = f"n_{positive_group}" if f"n_{positive_group}" in out.columns else "pvalue"
        out = out.sort_values(["pvalue", sort_col], ascending=[True, False])
    return out


def lodo_raw_token_stability(df: pd.DataFrame, raw_token_cols: list[str], raw_token_catalog: pd.DataFrame) -> pd.DataFrame:
    sub = df[df['delta_group'].isin(['improve', 'worsen'])].copy()
    drugs = sub['drug'].dropna().astype(str).tolist()
    lookup = dict(zip(raw_token_catalog["feature"], raw_token_catalog["raw_token"])) if not raw_token_catalog.empty else {}
    rows = []
    for c in raw_token_cols:
        effects, pvals = [], []
        for drug in drugs:
            tmp = sub[sub['drug'] != drug]
            improve = set(tmp.loc[tmp['delta_group'] == 'improve', 'drug'])
            worsen = set(tmp.loc[tmp['delta_group'] == 'worsen', 'drug'])
            token_drugs = set(tmp.loc[clean_numeric_series(tmp[c]).fillna(0) > 0, 'drug'])
            a = len(token_drugs & improve)
            b = len(token_drugs & worsen)
            c0 = len(improve - token_drugs)
            d = len(worsen - token_drugs)
            if a + b < 2:
                continue
            effects.append(safe_log2_or(a, b, c0, d))
            try:
                pvals.append(float(fisher_exact([[a, b], [c0, d]], alternative='two-sided').pvalue))
            except Exception:
                try:
                    _, p = fisher_exact([[a, b], [c0, d]], alternative='two-sided')
                    pvals.append(float(p))
                except Exception:
                    pvals.append(np.nan)
        eff_arr = np.array([e for e in effects if np.isfinite(e)], dtype=float)
        p_arr = np.array([p for p in pvals if np.isfinite(p)], dtype=float)
        if eff_arr.size == 0:
            continue
        maj = effect_sign(np.nanmedian(eff_arr))
        sign_consistency = float(np.mean([effect_sign(e) == maj for e in eff_arr])) if maj != 0 else np.nan
        rows.append({
            'feature': c,
            'raw_token': lookup.get(c, c.replace('rawtok_', '')),
            'lodo_runs_evaluable': int(len(effects)),
            'lodo_median_log2_odds_ratio': float(np.nanmedian(eff_arr)),
            'lodo_min_log2_odds_ratio': float(np.nanmin(eff_arr)),
            'lodo_max_log2_odds_ratio': float(np.nanmax(eff_arr)),
            'lodo_majority_direction': 'enriched_in_improve' if maj > 0 else ('enriched_in_worsen' if maj < 0 else 'neutral_or_missing'),
            'lodo_sign_consistency': sign_consistency,
            'lodo_p_lt_0_05_rate': float(np.mean(p_arr < 0.05)) if p_arr.size else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(['lodo_sign_consistency', 'lodo_p_lt_0_05_rate'], ascending=[False, False])
    return out


def threshold_robustness_raw_token(df: pd.DataFrame, raw_token_cols: list[str], raw_token_catalog: pd.DataFrame, thresholds: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    for thr in thresholds:
        tmp = df.copy()
        tmp['delta_group_thr'] = build_delta_groups_from_threshold(tmp, thr)
        res = raw_token_univariate(tmp, raw_token_cols, raw_token_catalog, group_col='delta_group_thr', group_a='improve', group_b='worsen')
        if res.empty:
            continue
        res = res[['feature', 'raw_token', 'pvalue', 'fdr_bh', 'odds_ratio', 'log2_odds_ratio', 'effect_direction', 'n_drugs_supporting_token']].copy()
        res['threshold'] = thr
        detail_rows.append(res)
    detail = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame(columns=['feature', 'threshold'])
    summary_rows = []
    if not detail.empty:
        for feat, g in detail.groupby('feature', sort=False):
            eff = clean_numeric_series(g['log2_odds_ratio']).to_numpy(dtype=float)
            eff = eff[np.isfinite(eff)]
            p = clean_numeric_series(g['pvalue']).to_numpy(dtype=float)
            p = p[np.isfinite(p)]
            maj = effect_sign(np.nanmedian(eff)) if eff.size else 0
            sign_consistency = float(np.mean([effect_sign(e) == maj for e in eff])) if eff.size and maj != 0 else np.nan
            summary_rows.append({
                'feature': feat,
                'raw_token': g['raw_token'].iloc[0] if 'raw_token' in g.columns and len(g) else feat.replace('rawtok_', ''),
                'thresholds_evaluable': int(g['threshold'].nunique()),
                'threshold_median_log2_odds_ratio': float(np.nanmedian(eff)) if eff.size else np.nan,
                'threshold_sign_consistency': sign_consistency,
                'threshold_p_lt_0_05_rate': float(np.mean(p < 0.05)) if p.size else np.nan,
                'threshold_fdr_lt_0_10_rate': float(np.mean(clean_numeric_series(g['fdr_bh']) < 0.10)) if len(g) else np.nan,
                'threshold_majority_direction': 'enriched_in_improve' if maj > 0 else ('enriched_in_worsen' if maj < 0 else 'neutral_or_missing'),
            })
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(['threshold_sign_consistency', 'threshold_p_lt_0_05_rate'], ascending=[False, False])
    return detail, summary


def build_raw_token_consensus(raw_uni: pd.DataFrame, lodo: pd.DataFrame, thr: pd.DataFrame) -> pd.DataFrame:
    if raw_uni.empty:
        return pd.DataFrame()
    out = raw_uni.copy()
    if not lodo.empty:
        out = out.merge(lodo, on=['feature', 'raw_token'], how='left') if 'raw_token' in lodo.columns else out.merge(lodo, on='feature', how='left')
    if not thr.empty:
        out = out.merge(thr, on=['feature', 'raw_token'], how='left') if 'raw_token' in thr.columns else out.merge(thr, on='feature', how='left')
    out['consensus_score'] = (
        clean_numeric_series(out.get('minus_log10_p', np.nan)).fillna(0) * 0.35 +
        clean_numeric_series(out.get('lodo_sign_consistency', np.nan)).fillna(0) * 2.0 +
        clean_numeric_series(out.get('threshold_sign_consistency', np.nan)).fillna(0) * 2.0 +
        clean_numeric_series(out.get('support_rate_total', np.nan)).fillna(0) * 1.5 +
        clean_numeric_series(out.get('threshold_p_lt_0_05_rate', np.nan)).fillna(0) * 1.5
    )
    out = out.sort_values(['consensus_score', 'minus_log10_p', 'n_drugs_supporting_token'], ascending=[False, False, False])
    return out


def build_clean_numeric_matrix(
    drug_sample_agg: pd.DataFrame,
    dfs: Optional[pd.DataFrame],
    mol: Optional[pd.DataFrame],
    pubchem: Optional[pd.DataFrame],
    mech_num: Optional[pd.DataFrame],
    act_num: Optional[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = drug_sample_agg.copy()
    keep_ds = ["drug"] + [c for c in BASE_NUMERIC_COLS if c in drug_sample_agg.columns]
    out = out[keep_ds].copy()
    tables = {
        "drug_feature_summary": dfs,
        "molecules": mol,
        "pubchem": pubchem,
        "mechanisms": mech_num,
        "activities": act_num,
    }
    audit_rows = []
    for canonical, choices in CANONICAL_NUMERIC_SPECS:
        selected_src = None
        for src, raw in choices:
            tab = tables.get(src)
            if tab is not None and raw in tab.columns:
                tmp = tab[["drug", raw]].copy().rename(columns={raw: canonical})
                tmp[canonical] = clean_numeric_series(tmp[canonical])
                out = out.merge(tmp, on="drug", how="left")
                selected_src = src
                break
        audit_rows.append({
            "canonical_feature": canonical,
            "selected_source": selected_src,
            "status": "kept" if selected_src else "missing",
            "candidate_chain": " > ".join([f"{s}.{r}" for s, r in choices]),
        })
    audit = pd.DataFrame(audit_rows)
    for c in out.columns:
        if c != "drug":
            out[c] = clean_numeric_series(out[c])
    return out, audit


def calc_numeric_assoc(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    y = clean_numeric_series(df["delta"]).to_numpy(dtype=float)
    for c in feature_cols:
        x = clean_numeric_series(df[c]).to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 4:
            continue
        if np.nanstd(x[m]) == 0 or np.nanstd(y[m]) == 0:
            sp = np.nan
            pr = np.nan
        else:
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
            "feature": c,
            "n": int(m.sum()),
            "spearman": safe_float(sp),
            "abs_spearman": abs(safe_float(sp)) if np.isfinite(safe_float(sp)) else np.nan,
            "pearson": safe_float(pr),
            "abs_pearson": abs(safe_float(pr)) if np.isfinite(safe_float(pr)) else np.nan,
            "mean_feature": finite_mean(x[m]),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["abs_spearman", "abs_pearson"], ascending=False)
    return out


def numeric_group_compare(df: pd.DataFrame, feature_cols: list[str], group_col: str = 'delta_group', group_a: str = 'improve', group_b: str = 'worsen') -> pd.DataFrame:
    rows = []
    sub = df[df[group_col].isin([group_a, group_b])].copy()
    n_total = int(len(sub))
    for c in feature_cols:
        s_all = clean_numeric_series(sub[c])
        xa = finite_values(sub.loc[sub[group_col] == group_a, c])
        xb = finite_values(sub.loc[sub[group_col] == group_b, c])
        n_nonmissing = int(s_all.notna().sum())
        if len(xa) < 2 or len(xb) < 2:
            continue
        try:
            p = mannwhitneyu(xa, xb, alternative='two-sided').pvalue
        except Exception:
            p = np.nan
        d = cohen_d(xa, xb)
        try:
            auc = roc_auc_score(
                np.r_[np.ones(len(xa), dtype=int), np.zeros(len(xb), dtype=int)],
                np.r_[xa, xb],
            )
            auc = max(auc, 1 - auc)
        except Exception:
            auc = np.nan
        rows.append({
            'feature': c,
            f'n_{group_a}': int(len(xa)),
            f'n_{group_b}': int(len(xb)),
            'n_nonmissing_total': n_nonmissing,
            'coverage_rate_total': safe_float(n_nonmissing / n_total) if n_total else np.nan,
            f'mean_{group_a}': finite_mean(xa),
            f'mean_{group_b}': finite_mean(xb),
            f'median_{group_a}': finite_median(xa),
            f'median_{group_b}': finite_median(xb),
            f'mean_diff_{group_a}_minus_{group_b}': finite_mean(xa) - finite_mean(xb) if len(xa) and len(xb) else np.nan,
            f'median_diff_{group_a}_minus_{group_b}': finite_median(xa) - finite_median(xb) if len(xa) and len(xb) else np.nan,
            f'cohen_d_{group_a}_minus_{group_b}': d,
            'abs_cohen_d': abs(d) if np.isfinite(d) else np.nan,
            'effect_direction': 'higher_in_' + group_a if np.isfinite(d) and d > 0 else ('higher_in_' + group_b if np.isfinite(d) and d < 0 else 'neutral_or_missing'),
            'pvalue': safe_float(p),
            'single_feature_auc_abs': safe_float(auc),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_fdr(out)
        out['minus_log10_p'] = -np.log10(clean_numeric_series(out['pvalue']).clip(lower=1e-300))
        out = out.sort_values(['abs_cohen_d', 'pvalue'], ascending=[False, True])
    return out


def theme_univariate(df: pd.DataFrame, theme_cols: list[str], group_col: str = 'delta_group', group_a: str = 'improve', group_b: str = 'worsen') -> pd.DataFrame:
    rows = []
    sub = df[df[group_col].isin([group_a, group_b])].copy()
    group_a_set = set(sub.loc[sub[group_col] == group_a, 'drug'])
    group_b_set = set(sub.loc[sub[group_col] == group_b, 'drug'])
    eligible_drugs = group_a_set | group_b_set
    for c in theme_cols:
        token_drugs = set(sub.loc[clean_numeric_series(sub[c]).fillna(0) > 0, 'drug'])
        a = len(token_drugs & group_a_set)
        b = len(token_drugs & group_b_set)
        c0 = len(group_a_set - token_drugs)
        d = len(group_b_set - token_drugs)
        if a + b < 2:
            continue
        try:
            odds, p = fisher_exact([[a, b], [c0, d]], alternative='two-sided')
        except Exception:
            odds, p = np.nan, np.nan
        log2_or = safe_log2_or(a, b, c0, d)
        rows.append({
            'feature': c,
            'theme': c.replace('theme_', ''),
            f'n_{group_a}_with_theme': a,
            f'n_{group_b}_with_theme': b,
            'n_drugs_supporting_theme': a + b,
            'support_rate_total': safe_float((a + b) / len(eligible_drugs)) if eligible_drugs else np.nan,
            f'support_rate_{group_a}': safe_float(a / len(group_a_set)) if group_a_set else np.nan,
            f'support_rate_{group_b}': safe_float(b / len(group_b_set)) if group_b_set else np.nan,
            'odds_ratio': safe_float(odds),
            'log2_odds_ratio': log2_or,
            'effect_direction': 'enriched_in_' + group_a if np.isfinite(log2_or) and log2_or > 0 else ('enriched_in_' + group_b if np.isfinite(log2_or) and log2_or < 0 else 'neutral_or_missing'),
            'pvalue': safe_float(p),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_fdr(out)
        out['minus_log10_p'] = -np.log10(clean_numeric_series(out['pvalue']).clip(lower=1e-300))
        out = out.sort_values(['pvalue', 'n_drugs_supporting_theme', 'feature'], ascending=[True, False, True])
    return out


def theme_enrichment(delta_df: pd.DataFrame, theme_mat: pd.DataFrame, positive_group: str, opposite_group: Optional[str] = None) -> pd.DataFrame:
    if theme_mat.empty:
        return pd.DataFrame(columns=["source", "field", "token"])
    if opposite_group is None:
        pos_set = set(delta_df.loc[delta_df["delta_group"] == positive_group, "drug"])
        all_set = set(delta_df["drug"])
    else:
        pos_set = set(delta_df.loc[delta_df["delta_group"] == positive_group, "drug"])
        all_set = pos_set | set(delta_df.loc[delta_df["delta_group"] == opposite_group, "drug"])
    rows = []
    theme_cols = [c for c in theme_mat.columns if c.startswith("theme_")]
    for c in theme_cols:
        token = c.replace("theme_", "")
        token_drugs = set(theme_mat.loc[clean_numeric_series(theme_mat[c]).fillna(0) > 0, "drug"]) & all_set
        if len(token_drugs) < 2:
            continue
        if opposite_group is None:
            a = len(token_drugs & pos_set)
            b = len(token_drugs - pos_set)
            c0 = len(pos_set - token_drugs)
            d = len((all_set - pos_set) - token_drugs)
        else:
            neg_set = set(delta_df.loc[delta_df["delta_group"] == opposite_group, "drug"])
            a = len(token_drugs & pos_set)
            b = len(token_drugs & neg_set)
            c0 = len(pos_set - token_drugs)
            d = len(neg_set - token_drugs)
        if (a + b) < 2:
            continue
        try:
            odds, p = fisher_exact([[a, b], [c0, d]], alternative="greater")
        except Exception:
            odds, p = np.nan, np.nan
        row = {
            "source": "theme",
            "field": "theme",
            "token": token,
            "odds_ratio": safe_float(odds),
            "pvalue": safe_float(p),
        }
        if opposite_group is None:
            row[f"n_{positive_group}"] = a
            row["n_with_token"] = a + b
            row["n_total"] = len(all_set)
        else:
            row[f"n_{positive_group}"] = a
            row[f"n_{opposite_group}"] = b
            row[f"n_total_{positive_group}"] = len(pos_set)
            row[f"n_total_{opposite_group}"] = len(set(delta_df.loc[delta_df["delta_group"] == opposite_group, "drug"]))
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_fdr(out)
        out["minus_log10_p"] = -np.log10(clean_numeric_series(out["pvalue"]).clip(lower=1e-300))
        out["minus_log10_fdr"] = -np.log10(clean_numeric_series(out["fdr_bh"]).clip(lower=1e-300))
        sort_col = f"n_{positive_group}" if f"n_{positive_group}" in out.columns else "pvalue"
        out = out.sort_values(["pvalue", sort_col], ascending=[True, False])
    return out


def lodo_numeric_stability(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    sub = df[df['delta_group'].isin(['improve', 'worsen'])].copy()
    drugs = sub['drug'].dropna().astype(str).tolist()
    rows = []
    for c in feature_cols:
        effects, pvals = [], []
        for drug in drugs:
            tmp = sub[sub['drug'] != drug]
            xa = finite_values(tmp.loc[tmp['delta_group'] == 'improve', c])
            xb = finite_values(tmp.loc[tmp['delta_group'] == 'worsen', c])
            if len(xa) < 2 or len(xb) < 2:
                continue
            effects.append(cohen_d(xa, xb))
            try:
                pvals.append(float(mannwhitneyu(xa, xb, alternative='two-sided').pvalue))
            except Exception:
                pvals.append(np.nan)
        eff_arr = np.array([e for e in effects if np.isfinite(e)], dtype=float)
        p_arr = np.array([p for p in pvals if np.isfinite(p)], dtype=float)
        if eff_arr.size == 0:
            continue
        maj = effect_sign(np.nanmedian(eff_arr))
        sign_consistency = float(np.mean([effect_sign(e) == maj for e in eff_arr])) if maj != 0 else np.nan
        rows.append({
            'feature': c,
            'lodo_runs_evaluable': int(len(effects)),
            'lodo_median_effect': float(np.nanmedian(eff_arr)),
            'lodo_min_effect': float(np.nanmin(eff_arr)),
            'lodo_max_effect': float(np.nanmax(eff_arr)),
            'lodo_majority_direction': 'higher_in_improve' if maj > 0 else ('higher_in_worsen' if maj < 0 else 'neutral_or_missing'),
            'lodo_sign_consistency': sign_consistency,
            'lodo_p_lt_0_05_rate': float(np.mean(p_arr < 0.05)) if p_arr.size else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(['lodo_sign_consistency', 'lodo_p_lt_0_05_rate', 'lodo_median_effect'], ascending=[False, False, False])
    return out


def lodo_theme_stability(df: pd.DataFrame, theme_cols: list[str]) -> pd.DataFrame:
    sub = df[df['delta_group'].isin(['improve', 'worsen'])].copy()
    drugs = sub['drug'].dropna().astype(str).tolist()
    rows = []
    for c in theme_cols:
        effects, pvals = [], []
        for drug in drugs:
            tmp = sub[sub['drug'] != drug]
            improve = set(tmp.loc[tmp['delta_group'] == 'improve', 'drug'])
            worsen = set(tmp.loc[tmp['delta_group'] == 'worsen', 'drug'])
            token_drugs = set(tmp.loc[clean_numeric_series(tmp[c]).fillna(0) > 0, 'drug'])
            a = len(token_drugs & improve)
            b = len(token_drugs & worsen)
            c0 = len(improve - token_drugs)
            d = len(worsen - token_drugs)
            if a + b < 2:
                continue
            effects.append(safe_log2_or(a, b, c0, d))
            try:
                pvals.append(float(fisher_exact([[a, b], [c0, d]], alternative='two-sided').pvalue))
            except Exception:
                try:
                    _, p = fisher_exact([[a, b], [c0, d]], alternative='two-sided')
                    pvals.append(float(p))
                except Exception:
                    pvals.append(np.nan)
        eff_arr = np.array([e for e in effects if np.isfinite(e)], dtype=float)
        p_arr = np.array([p for p in pvals if np.isfinite(p)], dtype=float)
        if eff_arr.size == 0:
            continue
        maj = effect_sign(np.nanmedian(eff_arr))
        sign_consistency = float(np.mean([effect_sign(e) == maj for e in eff_arr])) if maj != 0 else np.nan
        rows.append({
            'feature': c,
            'theme': c.replace('theme_', ''),
            'lodo_runs_evaluable': int(len(effects)),
            'lodo_median_log2_odds_ratio': float(np.nanmedian(eff_arr)),
            'lodo_min_log2_odds_ratio': float(np.nanmin(eff_arr)),
            'lodo_max_log2_odds_ratio': float(np.nanmax(eff_arr)),
            'lodo_majority_direction': 'enriched_in_improve' if maj > 0 else ('enriched_in_worsen' if maj < 0 else 'neutral_or_missing'),
            'lodo_sign_consistency': sign_consistency,
            'lodo_p_lt_0_05_rate': float(np.mean(p_arr < 0.05)) if p_arr.size else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(['lodo_sign_consistency', 'lodo_p_lt_0_05_rate'], ascending=[False, False])
    return out


def threshold_robustness_numeric(df: pd.DataFrame, feature_cols: list[str], thresholds: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    for thr in thresholds:
        tmp = df.copy()
        tmp['delta_group_thr'] = build_delta_groups_from_threshold(tmp, thr)
        res = numeric_group_compare(tmp, feature_cols, group_col='delta_group_thr', group_a='improve', group_b='worsen')
        if res.empty:
            continue
        res = res[['feature', 'pvalue', 'fdr_bh', 'abs_cohen_d', 'single_feature_auc_abs', 'cohen_d_improve_minus_worsen', 'effect_direction']].copy()
        res['threshold'] = thr
        detail_rows.append(res)
    detail = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame(columns=['feature', 'threshold'])
    summary_rows = []
    if not detail.empty:
        for feat, g in detail.groupby('feature', sort=False):
            eff = clean_numeric_series(g['cohen_d_improve_minus_worsen']).to_numpy(dtype=float)
            eff = eff[np.isfinite(eff)]
            p = clean_numeric_series(g['pvalue']).to_numpy(dtype=float)
            p = p[np.isfinite(p)]
            maj = effect_sign(np.nanmedian(eff)) if eff.size else 0
            sign_consistency = float(np.mean([effect_sign(e) == maj for e in eff])) if eff.size and maj != 0 else np.nan
            summary_rows.append({
                'feature': feat,
                'thresholds_evaluable': int(g['threshold'].nunique()),
                'threshold_median_effect': float(np.nanmedian(eff)) if eff.size else np.nan,
                'threshold_sign_consistency': sign_consistency,
                'threshold_p_lt_0_05_rate': float(np.mean(p < 0.05)) if p.size else np.nan,
                'threshold_fdr_lt_0_10_rate': float(np.mean(clean_numeric_series(g['fdr_bh']) < 0.10)) if len(g) else np.nan,
                'threshold_majority_direction': 'higher_in_improve' if maj > 0 else ('higher_in_worsen' if maj < 0 else 'neutral_or_missing'),
            })
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(['threshold_sign_consistency', 'threshold_p_lt_0_05_rate', 'threshold_median_effect'], ascending=[False, False, False])
    return detail, summary


def threshold_robustness_theme(df: pd.DataFrame, theme_cols: list[str], thresholds: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    for thr in thresholds:
        tmp = df.copy()
        tmp['delta_group_thr'] = build_delta_groups_from_threshold(tmp, thr)
        res = theme_univariate(tmp, theme_cols, group_col='delta_group_thr', group_a='improve', group_b='worsen')
        if res.empty:
            continue
        res = res[['feature', 'theme', 'pvalue', 'fdr_bh', 'odds_ratio', 'log2_odds_ratio', 'effect_direction', 'n_drugs_supporting_theme']].copy()
        res['threshold'] = thr
        detail_rows.append(res)
    detail = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame(columns=['feature', 'threshold'])
    summary_rows = []
    if not detail.empty:
        for feat, g in detail.groupby('feature', sort=False):
            eff = clean_numeric_series(g['log2_odds_ratio']).to_numpy(dtype=float)
            eff = eff[np.isfinite(eff)]
            p = clean_numeric_series(g['pvalue']).to_numpy(dtype=float)
            p = p[np.isfinite(p)]
            maj = effect_sign(np.nanmedian(eff)) if eff.size else 0
            sign_consistency = float(np.mean([effect_sign(e) == maj for e in eff])) if eff.size and maj != 0 else np.nan
            summary_rows.append({
                'feature': feat,
                'theme': g['theme'].iloc[0] if 'theme' in g.columns and len(g) else feat.replace('theme_', ''),
                'thresholds_evaluable': int(g['threshold'].nunique()),
                'threshold_median_log2_odds_ratio': float(np.nanmedian(eff)) if eff.size else np.nan,
                'threshold_sign_consistency': sign_consistency,
                'threshold_p_lt_0_05_rate': float(np.mean(p < 0.05)) if p.size else np.nan,
                'threshold_fdr_lt_0_10_rate': float(np.mean(clean_numeric_series(g['fdr_bh']) < 0.10)) if len(g) else np.nan,
                'threshold_majority_direction': 'enriched_in_improve' if maj > 0 else ('enriched_in_worsen' if maj < 0 else 'neutral_or_missing'),
            })
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(['threshold_sign_consistency', 'threshold_p_lt_0_05_rate'], ascending=[False, False])
    return detail, summary


def build_theme_consensus(theme_uni: pd.DataFrame, lodo: pd.DataFrame, thr: pd.DataFrame) -> pd.DataFrame:
    if theme_uni.empty:
        return pd.DataFrame()
    out = theme_uni.copy()
    if not lodo.empty:
        out = out.merge(lodo, on=['feature', 'theme'], how='left') if 'theme' in lodo.columns else out.merge(lodo, on='feature', how='left')
    if not thr.empty:
        out = out.merge(thr, on=['feature', 'theme'], how='left') if 'theme' in thr.columns else out.merge(thr, on='feature', how='left')
    out['consensus_score'] = (
        clean_numeric_series(out.get('minus_log10_p', np.nan)).fillna(0) * 0.35 +
        clean_numeric_series(out.get('lodo_sign_consistency', np.nan)).fillna(0) * 2.0 +
        clean_numeric_series(out.get('threshold_sign_consistency', np.nan)).fillna(0) * 2.0 +
        clean_numeric_series(out.get('support_rate_total', np.nan)).fillna(0) * 1.5 +
        clean_numeric_series(out.get('threshold_p_lt_0_05_rate', np.nan)).fillna(0) * 1.5
    )
    out = out.sort_values(['consensus_score', 'minus_log10_p', 'n_drugs_supporting_theme'], ascending=[False, False, False])
    return out


def build_numeric_consensus(num_uni: pd.DataFrame, lodo: pd.DataFrame, thr: pd.DataFrame) -> pd.DataFrame:
    if num_uni.empty:
        return pd.DataFrame()
    out = num_uni.copy()
    if not lodo.empty:
        out = out.merge(lodo, on='feature', how='left')
    if not thr.empty:
        out = out.merge(thr, on='feature', how='left')
    out['consensus_score'] = (
        clean_numeric_series(out.get('abs_cohen_d', np.nan)).fillna(0) * 0.75 +
        clean_numeric_series(out.get('single_feature_auc_abs', np.nan)).fillna(0) * 0.75 +
        clean_numeric_series(out.get('coverage_rate_total', np.nan)).fillna(0) * 1.0 +
        clean_numeric_series(out.get('lodo_sign_consistency', np.nan)).fillna(0) * 2.0 +
        clean_numeric_series(out.get('threshold_sign_consistency', np.nan)).fillna(0) * 2.0
    )
    out = out.sort_values(['consensus_score', 'abs_cohen_d', 'single_feature_auc_abs'], ascending=[False, False, False])
    return out




def build_theme_token_effect_summary(raw_token_uni: pd.DataFrame, raw_token_theme_catalog: pd.DataFrame, theme_uni: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if raw_token_uni.empty or raw_token_theme_catalog.empty:
        return pd.DataFrame()
    mapped = raw_token_theme_catalog.copy()
    mapped = mapped[mapped["assigned_theme"].notna() & (mapped["assigned_theme"] != "UNMAPPED")].copy()
    if mapped.empty:
        return pd.DataFrame()
    cols = [c for c in [
        "feature", "raw_token", "log2_odds_ratio", "minus_log10_p", "pvalue",
        "n_drugs_supporting_token", "support_rate_total", "effect_direction"
    ] if c in raw_token_uni.columns]
    token_eff = raw_token_uni[cols].copy()
    df = mapped.merge(token_eff, on=["feature", "raw_token"], how="left")
    df = df.dropna(subset=["assigned_theme"])
    rows = []
    theme_lookup = {}
    if theme_uni is not None and not theme_uni.empty and "theme" in theme_uni.columns:
        theme_lookup = theme_uni.set_index("theme").to_dict(orient="index")
    for theme, g in df.groupby("assigned_theme", sort=True):
        eff = clean_numeric_series(g.get("log2_odds_ratio", np.nan)).to_numpy(dtype=float)
        eff = eff[np.isfinite(eff)]
        mlp = clean_numeric_series(g.get("minus_log10_p", np.nan)).to_numpy(dtype=float)
        mlp = mlp[np.isfinite(mlp)]
        support = clean_numeric_series(g.get("n_drugs_supporting_token", np.nan)).to_numpy(dtype=float)
        support = np.where(np.isfinite(support) & (support > 0), support, np.nan)
        support_rate = clean_numeric_series(g.get("support_rate_total", np.nan)).to_numpy(dtype=float)
        support_rate = np.where(np.isfinite(support_rate) & (support_rate > 0), support_rate, np.nan)

        if eff.size:
            maj = effect_sign(np.nanmedian(eff))
            sign_consistency = float(np.mean([effect_sign(e) == maj for e in eff])) if maj != 0 else np.nan
            pos_rate = float(np.mean(eff > 0))
            neg_rate = float(np.mean(eff < 0))
            med_eff = float(np.nanmedian(eff))
            mean_eff = float(np.nanmean(eff))
            mean_abs_eff = float(np.nanmean(np.abs(eff)))
            iqr_eff = float(np.nanquantile(eff, 0.75) - np.nanquantile(eff, 0.25)) if eff.size else np.nan
            top_idx = int(np.nanargmax(np.abs(eff)))
            top_row = g.iloc[top_idx]
            top_token = top_row.get("raw_token", np.nan)
            top_token_effect = safe_float(top_row.get("log2_odds_ratio", np.nan))
            top_token_support = safe_float(top_row.get("n_drugs_supporting_token", np.nan))
            order = np.argsort(-np.abs(eff))
            topk = eff[order[:min(3, len(order))]]
            top3_mean_abs = float(np.nanmean(np.abs(topk))) if len(topk) else np.nan
        else:
            maj = 0
            sign_consistency = np.nan
            pos_rate = np.nan
            neg_rate = np.nan
            med_eff = np.nan
            mean_eff = np.nan
            mean_abs_eff = np.nan
            iqr_eff = np.nan
            top_token = np.nan
            top_token_effect = np.nan
            top_token_support = np.nan
            top3_mean_abs = np.nan

        w_mean_support = np.nan
        w_mean_support_rate = np.nan
        if eff.size and support.size == len(g):
            m = np.isfinite(clean_numeric_series(g.get("log2_odds_ratio", np.nan)).to_numpy(dtype=float)) & np.isfinite(support)
            if m.any() and np.nansum(support[m]) > 0:
                vals = clean_numeric_series(g.get("log2_odds_ratio", np.nan)).to_numpy(dtype=float)[m]
                w_mean_support = float(np.average(vals, weights=support[m]))
        if eff.size and support_rate.size == len(g):
            m = np.isfinite(clean_numeric_series(g.get("log2_odds_ratio", np.nan)).to_numpy(dtype=float)) & np.isfinite(support_rate)
            if m.any() and np.nansum(support_rate[m]) > 0:
                vals = clean_numeric_series(g.get("log2_odds_ratio", np.nan)).to_numpy(dtype=float)[m]
                w_mean_support_rate = float(np.average(vals, weights=support_rate[m]))

        row = {
            "theme": theme,
            "n_tokens_mapped_to_theme": int(len(g)),
            "n_tokens_with_effect": int(np.isfinite(clean_numeric_series(g.get("log2_odds_ratio", np.nan))).sum()),
            "mean_token_log2_odds_ratio": mean_eff,
            "median_token_log2_odds_ratio": med_eff,
            "support_weighted_mean_token_log2_odds_ratio": w_mean_support,
            "support_rate_weighted_mean_token_log2_odds_ratio": w_mean_support_rate,
            "mean_abs_token_log2_odds_ratio": mean_abs_eff,
            "token_effect_iqr": iqr_eff,
            "mean_token_minus_log10_p": float(np.nanmean(mlp)) if mlp.size else np.nan,
            "median_token_minus_log10_p": float(np.nanmedian(mlp)) if mlp.size else np.nan,
            "token_sign_consistency": sign_consistency,
            "prop_tokens_positive": pos_rate,
            "prop_tokens_negative": neg_rate,
            "token_majority_direction": "enriched_in_improve" if maj > 0 else ("enriched_in_worsen" if maj < 0 else "neutral_or_missing"),
            "top_token": top_token,
            "top_token_log2_odds_ratio": top_token_effect,
            "top_token_support": top_token_support,
            "top3_mean_abs_token_log2_odds_ratio": top3_mean_abs,
        }
        theme_row = theme_lookup.get(theme)
        if theme_row is not None:
            for key in [
                "feature", "n_improve_with_theme", "n_worsen_with_theme", "n_drugs_supporting_theme",
                "support_rate_total", "support_rate_improve", "support_rate_worsen",
                "odds_ratio", "log2_odds_ratio", "effect_direction", "pvalue", "fdr_bh",
                "minus_log10_p", "consensus_score"
            ]:
                if key in theme_row:
                    row[f"theme_{key}"] = theme_row[key]
        row["theme_size_bias_gap"] = (
            row.get("theme_log2_odds_ratio", np.nan) - row.get("support_weighted_mean_token_log2_odds_ratio", np.nan)
            if np.isfinite(row.get("theme_log2_odds_ratio", np.nan)) and np.isfinite(row.get("support_weighted_mean_token_log2_odds_ratio", np.nan))
            else np.nan
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["support_rate_weighted_mean_token_log2_odds_ratio", "token_sign_consistency", "mean_token_minus_log10_p"],
            ascending=[False, False, False]
        )
    return out

def build_classifier(df: pd.DataFrame, numeric_cols: list[str], theme_cols: list[str], max_features: int = 12):
    sub = df[df["delta_group"].isin(["improve", "worsen"])].copy()
    sub["label"] = (sub["delta_group"] == "improve").astype(int)

    keep_num = []
    for c in numeric_cols:
        s = clean_numeric_series(sub[c])
        if s.notna().sum() >= 8 and s.nunique(dropna=True) >= 3:
            keep_num.append(c)

    keep_theme = []
    for c in theme_cols:
        s = clean_numeric_series(sub[c]).fillna(0)
        if s.sum() >= 2 and s.sum() <= len(s) - 2:
            keep_theme.append(c)

    num_uni = numeric_group_compare(sub, keep_num)
    theme_uni = theme_univariate(sub, keep_theme)

    selected = []
    if not num_uni.empty:
        selected += num_uni.head(max_features // 2)["feature"].tolist()
    if not theme_uni.empty:
        selected += theme_uni.head(max_features // 2)["feature"].tolist()
    if len(selected) < 4:
        selected = (keep_num[:max_features] + keep_theme[:max_features])[:max_features]

    seen = set()
    selected = [x for x in selected if not (x in seen or seen.add(x))]
    if not selected:
        raise ValueError("No usable clean numeric/theme features remain after filtering.")

    X = sub[selected].copy()
    for c in X.columns:
        X[c] = clean_numeric_series(X[c]) if c in keep_num else clean_numeric_series(X[c]).fillna(0)
    y = sub["label"].to_numpy()

    num_sel = [c for c in selected if c in keep_num]
    bin_sel = [c for c in selected if c in keep_theme]
    transformers = []
    if num_sel:
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_sel))
    if bin_sel:
        transformers.append(("bin", Pipeline([("imp", SimpleImputer(strategy="most_frequent"))]), bin_sel))
    pre = ColumnTransformer(transformers, remainder="drop")

    clf = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=5000, class_weight="balanced")
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    n_splits = min(5, int(sub["label"].value_counts().min()))
    if n_splits < 2:
        raise ValueError("Not enough improve/worsen drugs for cross-validation.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    preds = []
    coef_rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y), 1):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            pipe.fit(X.iloc[tr], y[tr])
            prob = pipe.predict_proba(X.iloc[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        for idx, p, pp in zip(X.iloc[te].index, pred, prob):
            preds.append({
                "drug": sub.loc[idx, "drug"],
                "true_label": int(sub.loc[idx, "label"]),
                "true_group": sub.loc[idx, "delta_group"],
                "pred_prob_improve": float(pp),
                "pred_label": int(p),
                "fold": fold,
            })
        coef = pipe.named_steps["clf"].coef_.ravel()
        names = num_sel + bin_sel
        for n, v in zip(names, coef):
            coef_rows.append({"fold": fold, "feature": n, "coef": float(v), "abs_coef": abs(float(v))})

    pred_df = pd.DataFrame(preds)
    coef_df = pd.DataFrame(coef_rows)
    if coef_df.empty:
        coef_sum = pd.DataFrame(columns=["feature", "mean_coef", "mean_abs_coef", "selection_freq"])
    else:
        coef_sum = coef_df.groupby("feature", as_index=False).agg(
            mean_coef=("coef", "mean"),
            mean_abs_coef=("abs_coef", "mean"),
            selection_freq=("coef", lambda s: float(np.mean(np.abs(s) > 1e-8))),
        ).sort_values(["mean_abs_coef", "selection_freq"], ascending=[False, False])

    auc = roc_auc_score(pred_df["true_label"], pred_df["pred_prob_improve"]) if (not pred_df.empty and pred_df["true_label"].nunique() == 2) else np.nan
    acc = accuracy_score(pred_df["true_label"], pred_df["pred_label"]) if not pred_df.empty else np.nan
    summary = pd.DataFrame([{
        "n_drugs_used": int(len(sub)),
        "n_improve": int((sub["label"] == 1).sum()),
        "n_worsen": int((sub["label"] == 0).sum()),
        "cv_folds": n_splits,
        "cv_auc": safe_float(auc),
        "cv_accuracy": safe_float(acc),
        "n_selected_features": int(len(selected)),
    }])
    return selected, pred_df, coef_sum, summary


def annotate_barh(ax, values):
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return
    xmin = min(0.0, float(finite.min()))
    xmax = max(0.0, float(finite.max()))
    span = xmax - xmin if xmax > xmin else 1.0
    ax.set_xlim(xmin - span * 0.03, xmax + span * 0.15)
    for patch, v in zip(ax.patches, vals):
        y = patch.get_y() + patch.get_height() / 2
        x = v + span * 0.02 if v >= 0 else v - span * 0.02
        ha = "left" if v >= 0 else "right"
        ax.text(x, y, f"{v:.3f}", va="center", ha=ha, fontsize=9)


def pretty_axis_label(value_col: str) -> str:
    mapping = {
        'minus_log10_p': r'$-\log_{10}(p)$',
        'minus_log10_fdr': r'$-\log_{10}(\mathrm{FDR})$',
        'abs_cohen_d': r'$|d|$',
        'abs_spearman': r'$|\rho|$',
        'single_feature_auc_abs': r'$\mathrm{AUC}$',
        'mean_abs_coef': r'$|\beta|$',
        'consensus_score': 'consensus score',
        'delta': r'$\Delta$',
    }
    return mapping.get(value_col, value_col)


def plot_barh(df, label_col, value_col, title, out_png, topk=12):
    if df.empty or value_col not in df.columns:
        return
    g = df.head(topk).iloc[::-1]
    vals = clean_numeric_series(g[value_col]).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(g))))
    ax.barh(g[label_col], vals)
    ax.set_xlabel(pretty_axis_label(value_col))
    ax.set_title(title)
    annotate_barh(ax, vals)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_sorted_delta(df: pd.DataFrame, out_png: Path, title: str) -> None:
    g = df.sort_values("delta", ascending=False).copy()
    plt.figure(figsize=(max(10, 0.35 * len(g)), 5.5))
    colors = ["tab:green" if x == "improve" else "tab:red" if x == "worsen" else "tab:gray" for x in g["delta_group"]]
    plt.bar(range(len(g)), clean_numeric_series(g["delta"]).to_numpy(dtype=float), color=colors)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(range(len(g)), g["drug"], rotation=90)
    plt.ylabel("delta")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before-summary", required=True)
    ap.add_argument("--after-summary", required=True)
    ap.add_argument("--drug-sample", required=True)
    ap.add_argument("--metric", default="R2_ens")
    ap.add_argument("--improve-threshold", type=float, default=0.01)
    ap.add_argument("--worsen-threshold", type=float, default=0.01)
    ap.add_argument("--drug-feature-summary", default=None)
    ap.add_argument("--mechanisms-long", default=None)
    ap.add_argument("--activities-long", default=None)
    ap.add_argument("--molecules", default=None)
    ap.add_argument("--pubchem-by-drug", default=None)
    ap.add_argument("--targets-long", default=None)
    ap.add_argument("--max-features", type=int, default=12)
    ap.add_argument("--threshold-grid", default="0.005,0.01,0.02")
    ap.add_argument("--outdir", default="analyze_improve_worsen_clean")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    plots_dir = outdir / "plots"
    ensure_dir(outdir)
    ensure_dir(plots_dir)

    before = read_ensemble_summary(Path(args.before_summary), args.metric)
    after = read_ensemble_summary(Path(args.after_summary), args.metric)
    ds_agg, ds_tokens = aggregate_drug_sample(Path(args.drug_sample))

    df = before.merge(after, on="drug", how="outer", suffixes=("_before", "_after"))
    metric_upper = args.metric.upper()
    higher_better = not (metric_upper.startswith("RMSE") or metric_upper.startswith("MSE"))
    if higher_better:
        df["delta"] = clean_numeric_series(df["metric_after"] - df["metric_before"])
    else:
        df["delta"] = clean_numeric_series(df["metric_before"] - df["metric_after"])
    df["delta_group"] = df["delta"].apply(lambda x: classify_delta(x, args.improve_threshold, args.worsen_threshold))
    df = df.merge(ds_agg, on="drug", how="left")

    dfs = aggregate_drug_feature_summary(Path(args.drug_feature_summary)) if args.drug_feature_summary else None
    mol = aggregate_molecules(Path(args.molecules)) if args.molecules else None
    pubchem = aggregate_pubchem(Path(args.pubchem_by_drug)) if args.pubchem_by_drug else None
    mech_num, mech_tok = aggregate_mechanisms(Path(args.mechanisms_long)) if args.mechanisms_long else (None, pd.DataFrame(columns=["drug", "source", "field", "token"]))
    act_num, act_tok = aggregate_activities(Path(args.activities_long)) if args.activities_long else (None, pd.DataFrame(columns=["drug", "source", "field", "token"]))
    tgt_num, tgt_tok = aggregate_targets(Path(args.targets_long)) if args.targets_long else (None, pd.DataFrame(columns=["drug", "source", "field", "token"]))

    external_aggs = []
    for ext in [dfs, mol, pubchem, mech_num, act_num, tgt_num]:
        if ext is not None and not ext.empty:
            external_aggs.append(ext)

    clean_num, audit_num = build_clean_numeric_matrix(ds_agg, dfs, mol, pubchem, mech_num, act_num)
    df = df.merge(clean_num, on="drug", how="left", suffixes=("", "_dup"))
    dup_cols = [c for c in df.columns if c.endswith("_dup")]
    if dup_cols:
        df = df.drop(columns=dup_cols)

    token_tables = [ds_tokens, mech_tok, act_tok, tgt_tok]
    if dfs is not None:
        rows = []
        use_cols = [c for c in ["drug", "mechanism_of_action_list", "action_type_list"] if c in dfs.columns]
        if set(use_cols) >= {"drug"}:
            for _, r in dfs[use_cols].fillna("").iterrows():
                for field in [c for c in ["mechanism_of_action_list", "action_type_list"] if c in use_cols]:
                    vals = set(split_multi_value(r[field]))
                    for tok in sorted(t for t in vals if t):
                        rows.append({"drug": r["drug"], "source": "drug_feature_summary", "field": field, "token": tok})
        token_tables.append(pd.DataFrame(rows))
    tokens = (
        pd.concat([t for t in token_tables if t is not None and not t.empty], ignore_index=True).drop_duplicates()
        if any(t is not None and not t.empty for t in token_tables)
        else pd.DataFrame(columns=["drug", "source", "field", "token"])
    )
    raw_token_mat, raw_token_audit, raw_token_catalog = build_raw_token_presence(tokens)
    raw_token_theme_catalog = build_raw_token_theme_catalog(raw_token_catalog)
    df = df.merge(raw_token_mat, on="drug", how="left")
    raw_token_cols = [c for c in df.columns if c.startswith("rawtok_")]
    for c in raw_token_cols:
        df[c] = clean_numeric_series(df[c]).fillna(0).astype(int)

    theme_mat, token_audit = build_theme_presence(tokens)
    df = df.merge(theme_mat, on="drug", how="left")
    theme_cols = [c for c in df.columns if c.startswith("theme_")]
    for c in theme_cols:
        df[c] = clean_numeric_series(df[c]).fillna(0).astype(int)

    numeric_cols = [c for c in clean_num.columns if c != "drug"]
    thresholds = parse_threshold_grid(args.threshold_grid, args.improve_threshold)

    # raw / richer outputs
    df.sort_values("delta", ascending=False).to_csv(outdir / "drug_delta_table.csv", index=False)
    ds_agg.to_csv(outdir / "drug_sample_aggregated.csv", index=False)

    if external_aggs:
        ext = external_aggs[0].copy()
        for extra in external_aggs[1:]:
            ext = ext.merge(extra, on="drug", how="outer")
    else:
        ext = pd.DataFrame(columns=["drug"])
    ext.to_csv(outdir / "external_feature_aggregated.csv", index=False)

    ext_interp_cols = [c for c in clean_num.columns if c not in {"drug", *BASE_NUMERIC_COLS}]
    ext_interp = clean_num[["drug"] + ext_interp_cols].copy() if ext_interp_cols else pd.DataFrame(columns=["drug"])
    ext_interp.to_csv(outdir / "external_feature_aggregated_interpretable.csv", index=False)

    tokens.to_csv(outdir / "all_category_tokens.csv", index=False)
    raw_token_audit.to_csv(outdir / "all_category_tokens_raw_fair.csv", index=False)
    raw_token_catalog.to_csv(outdir / "raw_token_catalog.csv", index=False)
    raw_token_theme_catalog.to_csv(outdir / "raw_token_to_theme_catalog.csv", index=False)
    token_audit.to_csv(outdir / "all_category_tokens_theme_mapped.csv", index=False)

    clean_num.to_csv(outdir / "clean_numeric_feature_matrix.csv", index=False)
    raw_token_mat.to_csv(outdir / "clean_raw_token_presence_matrix.csv", index=False)
    theme_mat.to_csv(outdir / "clean_theme_presence_matrix.csv", index=False)
    audit_num.to_csv(outdir / "feature_audit_numeric.csv", index=False)
    raw_token_catalog.to_csv(outdir / "token_audit_raw_fair.csv", index=False)
    token_audit.to_csv(outdir / "token_audit_theme_mapping.csv", index=False)

    annotation_burden = summarize_annotation_burden(df, tokens, token_audit, theme_mat, numeric_cols)
    annotation_burden.to_csv(outdir / "annotation_burden_by_drug.csv", index=False)

    numeric_assoc = calc_numeric_assoc(df, numeric_cols)
    numeric_assoc.to_csv(outdir / "numeric_feature_associations.csv", index=False)

    num_group = numeric_group_compare(df, numeric_cols)
    num_group.to_csv(outdir / "numeric_group_comparison_improve_vs_worsen.csv", index=False)
    num_group.to_csv(outdir / "numeric_univariate_improve_vs_worsen.csv", index=False)

    raw_token_uni = raw_token_univariate(df, raw_token_cols, raw_token_catalog)
    raw_token_uni.to_csv(outdir / "raw_token_univariate_improve_vs_worsen.csv", index=False)

    theme_uni = theme_univariate(df, theme_cols)
    theme_uni.to_csv(outdir / "theme_univariate_improve_vs_worsen.csv", index=False)

    numeric_lodo = lodo_numeric_stability(df, numeric_cols)
    numeric_lodo.to_csv(outdir / "numeric_lodo_stability.csv", index=False)
    raw_token_lodo = lodo_raw_token_stability(df, raw_token_cols, raw_token_catalog)
    raw_token_lodo.to_csv(outdir / "raw_token_lodo_stability.csv", index=False)
    theme_lodo = lodo_theme_stability(df, theme_cols)
    theme_lodo.to_csv(outdir / "theme_lodo_stability.csv", index=False)

    numeric_thr_detail, numeric_thr_summary = threshold_robustness_numeric(df, numeric_cols, thresholds)
    numeric_thr_detail.to_csv(outdir / "numeric_threshold_robustness_detail.csv", index=False)
    numeric_thr_summary.to_csv(outdir / "numeric_threshold_robustness_summary.csv", index=False)
    raw_token_thr_detail, raw_token_thr_summary = threshold_robustness_raw_token(df, raw_token_cols, raw_token_catalog, thresholds)
    raw_token_thr_detail.to_csv(outdir / "raw_token_threshold_robustness_detail.csv", index=False)
    raw_token_thr_summary.to_csv(outdir / "raw_token_threshold_robustness_summary.csv", index=False)
    theme_thr_detail, theme_thr_summary = threshold_robustness_theme(df, theme_cols, thresholds)
    theme_thr_detail.to_csv(outdir / "theme_threshold_robustness_detail.csv", index=False)
    theme_thr_summary.to_csv(outdir / "theme_threshold_robustness_summary.csv", index=False)

    numeric_consensus = build_numeric_consensus(num_group, numeric_lodo, numeric_thr_summary)
    numeric_consensus.to_csv(outdir / "numeric_feature_fairness_summary.csv", index=False)
    raw_token_consensus = build_raw_token_consensus(raw_token_uni, raw_token_lodo, raw_token_thr_summary)
    raw_token_consensus.to_csv(outdir / "raw_token_feature_fairness_summary.csv", index=False)
    theme_consensus = build_theme_consensus(theme_uni, theme_lodo, theme_thr_summary)
    theme_consensus.to_csv(outdir / "theme_feature_fairness_summary.csv", index=False)
    theme_token_effect_summary = build_theme_token_effect_summary(raw_token_uni, raw_token_theme_catalog, theme_consensus)
    theme_token_effect_summary.to_csv(outdir / "theme_token_effect_summary.csv", index=False)

    raw_enrich_improve = raw_token_enrichment(df, raw_token_mat, raw_token_catalog, "improve")
    raw_enrich_worsen = raw_token_enrichment(df, raw_token_mat, raw_token_catalog, "worsen")
    raw_enrich_iw = raw_token_enrichment(df[df["delta_group"].isin(["improve", "worsen"])], raw_token_mat, raw_token_catalog, "improve", "worsen")
    raw_enrich_wi = raw_token_enrichment(df[df["delta_group"].isin(["improve", "worsen"])], raw_token_mat, raw_token_catalog, "worsen", "improve")
    raw_enrich_improve.to_csv(outdir / "raw_token_enrichment_improve.csv", index=False)
    raw_enrich_worsen.to_csv(outdir / "raw_token_enrichment_worsen.csv", index=False)
    raw_enrich_iw.to_csv(outdir / "raw_token_enrichment_improve_vs_worsen.csv", index=False)
    raw_enrich_wi.to_csv(outdir / "raw_token_enrichment_worsen_vs_improve.csv", index=False)

    enrich_improve = theme_enrichment(df, theme_mat, "improve")
    enrich_worsen = theme_enrichment(df, theme_mat, "worsen")
    enrich_iw = theme_enrichment(df[df["delta_group"].isin(["improve", "worsen"])], theme_mat, "improve", "worsen")
    enrich_wi = theme_enrichment(df[df["delta_group"].isin(["improve", "worsen"])], theme_mat, "worsen", "improve")
    enrich_improve.to_csv(outdir / "category_enrichment_improve.csv", index=False)
    enrich_worsen.to_csv(outdir / "category_enrichment_worsen.csv", index=False)
    enrich_iw.to_csv(outdir / "category_enrichment_improve_vs_worsen.csv", index=False)
    enrich_wi.to_csv(outdir / "category_enrichment_worsen_vs_improve.csv", index=False)

    use_df = df[df["delta_group"].isin(["improve", "worsen"])].copy()
    use_df.to_csv(outdir / "clean_feature_matrix_improve_vs_worsen.csv", index=False)

    selected, pred_df, coef_df, summary_df = build_classifier(df, numeric_cols, theme_cols, max_features=args.max_features)
    pred_df.to_csv(outdir / "classifier_cv_predictions.csv", index=False)
    coef_df.to_csv(outdir / "classifier_feature_coefficients.csv", index=False)
    summary_df["selected_features"] = " | ".join(selected)
    summary_df.to_csv(outdir / "classifier_summary.csv", index=False)

    analysis_summary = pd.DataFrame([{
        "metric": args.metric,
        "higher_better_raw_metric": higher_better,
        "delta_definition": "after-before" if higher_better else "before-after",
        "n_total_drugs": int(df["drug"].nunique()),
        "n_improve": int((df["delta_group"] == "improve").sum()),
        "n_worsen": int((df["delta_group"] == "worsen").sum()),
        "n_stable": int((df["delta_group"] == "stable").sum()),
        "median_delta": finite_median(df["delta"]),
        "mean_delta": finite_mean(df["delta"]),
        "improve_threshold": args.improve_threshold,
        "worsen_threshold": args.worsen_threshold,
        "n_clean_numeric_features": int(len(numeric_cols)),
        "n_raw_token_features": int(len(raw_token_cols)),
        "n_theme_features": int(len(theme_cols)),
        "n_raw_token_rows": int(len(tokens)),
        "n_raw_token_audit_rows": int(len(raw_token_audit)),
        "n_mapped_theme_token_rows": int(len(token_audit)),
        "classifier_max_features": int(args.max_features),
        "threshold_grid": ' | '.join(str(x) for x in thresholds),
    }])
    analysis_summary.to_csv(outdir / "analysis_summary.csv", index=False)

    plot_sorted_delta(df, plots_dir / "drug_delta_sorted.png", f"Drug-level delta ({args.metric})")
    plot_barh(numeric_assoc, "feature", "abs_spearman", "Top numeric features associated with delta", plots_dir / "top_numeric_features_by_abs_spearman.png")
    plot_barh(num_group, "feature", "abs_cohen_d", "Top numeric features: improve vs worsen", plots_dir / "top_numeric_univariate_abs_cohen_d.png")
    plot_barh(raw_token_uni, "raw_token", "minus_log10_p", "Top raw tokens: improve vs worsen", plots_dir / "top_raw_token_univariate_minus_log10_p.png")
    plot_barh(theme_uni, "feature", "minus_log10_p", "Top themes: improve vs worsen", plots_dir / "top_theme_univariate_minus_log10_p.png")
    plot_barh(numeric_consensus, "feature", "consensus_score", "Top numeric features by consensus", plots_dir / "top_numeric_fairness_consensus.png")
    plot_barh(raw_token_consensus, "raw_token", "consensus_score", "Top raw tokens by consensus", plots_dir / "top_raw_token_fairness_consensus.png")
    plot_barh(theme_consensus, "theme", "consensus_score", "Top themes by consensus", plots_dir / "top_theme_fairness_consensus.png")
    plot_barh(theme_token_effect_summary, "theme", "support_weighted_mean_token_log2_odds_ratio", "Themes by support-weighted mean token effect", plots_dir / "top_theme_support_weighted_mean_token_effect.png")
    plot_barh(theme_token_effect_summary, "theme", "mean_token_log2_odds_ratio", "Themes by mean token effect", plots_dir / "top_theme_mean_token_effect.png")
    plot_barh(coef_df, "feature", "mean_abs_coef", "Top classifier features by |coef|", plots_dir / "top_classifier_abs_coef.png")
    plot_barh(raw_enrich_improve, "token", "minus_log10_p", "Raw tokens enriched among improved drugs", plots_dir / "top_raw_tokens_enriched_improve.png")
    plot_barh(raw_enrich_worsen, "token", "minus_log10_p", "Raw tokens enriched among worsened drugs", plots_dir / "top_raw_tokens_enriched_worsen.png")
    plot_barh(raw_enrich_iw, "token", "minus_log10_p", "Raw tokens enriched: improve vs worsen", plots_dir / "top_raw_tokens_enriched_improve_vs_worsen.png")
    plot_barh(raw_enrich_wi, "token", "minus_log10_p", "Raw tokens enriched: worsen vs improve", plots_dir / "top_raw_tokens_enriched_worsen_vs_improve.png")
    plot_barh(enrich_improve, "token", "minus_log10_p", "Themes enriched among improved drugs", plots_dir / "top_enriched_categories_improve.png")
    plot_barh(enrich_worsen, "token", "minus_log10_p", "Themes enriched among worsened drugs", plots_dir / "top_enriched_categories_worsen.png")
    plot_barh(enrich_iw, "token", "minus_log10_p", "Themes enriched: improve vs worsen", plots_dir / "top_enriched_categories_improve_vs_worsen.png")
    plot_barh(enrich_wi, "token", "minus_log10_p", "Themes enriched: worsen vs improve", plots_dir / "top_enriched_categories_worsen_vs_improve.png")

    readme = []
    readme.append("This fixed clean script keeps the clean canonical feature/theme classifier and restores the richer summary analyses.")
    readme.append(f"Metric = {args.metric}")
    readme.append(f"Delta definition = {'after-before' if higher_better else 'before-after'}")
    readme.append(f"Improved / worsened / stable = {(df['delta_group'] == 'improve').sum()} / {(df['delta_group'] == 'worsen').sum()} / {(df['delta_group'] == 'stable').sum()}")
    readme.append(f"Clean numeric features = {len(numeric_cols)}")
    readme.append(f"Raw token features = {len(raw_token_cols)}")
    readme.append(f"Theme features = {len(theme_cols)}")
    readme.append("")
    readme.append("Fairness / robustness additions:")
    readme.append("- Each numeric feature now reports non-missing coverage so sparse features do not look artificially strong.")
    readme.append("- Raw tokens are tested fairly at the drug level: each drug contributes at most one vote per normalized token, no matter how many times that token appears across rows.")
    readme.append("- Theme outputs are kept as a higher-level summary layer on top of the raw-token layer.")
    readme.append("- Each theme is counted at most once per drug after token-to-theme mapping, reducing duplicate counting of the same concept.")
    readme.append("- Leave-one-drug-out stability is exported to check whether a result is driven by only one drug.")
    readme.append("- Threshold robustness is exported across the threshold grid to see whether the same direction survives different improve/worsen cutoffs.")
    readme.append("")
    readme.append("Warning fix:")
    readme.append(f"- inf / -inf and |value| > {EXTREME_ABS_VALUE:.0e} are coerced to NaN before summary stats and modeling.")
    readme.append("- This removes the pandas/numpy overflow and mean-of-empty-slice spam that came from pathological numeric values.")
    if not numeric_assoc.empty:
        readme.append("")
        readme.append("Top numeric associations by |Spearman|:")
        for _, r in numeric_assoc.head(10).iterrows():
            readme.append(f"- {r['feature']}: spearman={r['spearman']:.3f}, pearson={r['pearson']:.3f}, n={int(r['n'])}")
    if not num_group.empty:
        readme.append("")
        readme.append("Top numeric improve-vs-worsen differences by |Cohen d|:")
        for _, r in num_group.head(10).iterrows():
            readme.append(f"- {r['feature']}: cohen_d={r['cohen_d_improve_minus_worsen']:.3f}, p={r['pvalue']:.3g}")
    if not raw_token_consensus.empty:
        readme.append("")
        readme.append("Top raw-token fairness consensus results:")
        for _, r in raw_token_consensus.head(10).iterrows():
            readme.append(f"- {r['raw_token']}: consensus={r['consensus_score']:.3f}, support={r['n_drugs_supporting_token']}, lodo_consistency={safe_float(r.get('lodo_sign_consistency')):.3f}, threshold_consistency={safe_float(r.get('threshold_sign_consistency')):.3f}")
    if not theme_consensus.empty:
        readme.append("")
        readme.append("Top theme consensus results:")
        for _, r in theme_consensus.head(10).iterrows():
            readme.append(f"- {r['theme']}: consensus={r['consensus_score']:.3f}, support={r['n_drugs_supporting_theme']}, lodo_consistency={safe_float(r.get('lodo_sign_consistency')):.3f}, threshold_consistency={safe_float(r.get('threshold_sign_consistency')):.3f}")
    if not theme_token_effect_summary.empty:
        readme.append("")
        readme.append("Theme-internal token effect summaries:")
        for _, r in theme_token_effect_summary.head(10).iterrows():
            readme.append(f"- {r['theme']}: theme_log2_or={safe_float(r.get('theme_log2_odds_ratio')):.3f}, token_mean={safe_float(r.get('mean_token_log2_odds_ratio')):.3f}, token_weighted_mean={safe_float(r.get('support_weighted_mean_token_log2_odds_ratio')):.3f}, sign_consistency={safe_float(r.get('token_sign_consistency')):.3f}")
    if not enrich_iw.empty:
        readme.append("")
        readme.append("Top theme enrichments: improve vs worsen:")
        for _, r in enrich_iw.head(10).iterrows():
            readme.append(f"- {r['token']}: odds_ratio={r['odds_ratio']:.3g}, p={r['pvalue']:.3g}")
    (outdir / "README_results.txt").write_text("\n".join(readme), encoding="utf-8")

    print("[OK] wrote:")
    for fn in [
        "analysis_summary.csv",
        "drug_delta_table.csv",
        "drug_sample_aggregated.csv",
        "external_feature_aggregated.csv",
        "external_feature_aggregated_interpretable.csv",
        "all_category_tokens.csv",
        "all_category_tokens_raw_fair.csv",
        "raw_token_catalog.csv",
        "all_category_tokens_theme_mapped.csv",
        "clean_numeric_feature_matrix.csv",
        "clean_raw_token_presence_matrix.csv",
        "clean_theme_presence_matrix.csv",
        "clean_feature_matrix_improve_vs_worsen.csv",
        "annotation_burden_by_drug.csv",
        "numeric_feature_associations.csv",
        "numeric_group_comparison_improve_vs_worsen.csv",
        "numeric_univariate_improve_vs_worsen.csv",
        "numeric_lodo_stability.csv",
        "numeric_threshold_robustness_detail.csv",
        "numeric_threshold_robustness_summary.csv",
        "numeric_feature_fairness_summary.csv",
        "raw_token_univariate_improve_vs_worsen.csv",
        "raw_token_lodo_stability.csv",
        "raw_token_threshold_robustness_detail.csv",
        "raw_token_threshold_robustness_summary.csv",
        "raw_token_feature_fairness_summary.csv",
        "raw_token_enrichment_improve.csv",
        "raw_token_enrichment_worsen.csv",
        "raw_token_enrichment_improve_vs_worsen.csv",
        "raw_token_enrichment_worsen_vs_improve.csv",
        "theme_univariate_improve_vs_worsen.csv",
        "theme_lodo_stability.csv",
        "theme_threshold_robustness_detail.csv",
        "theme_threshold_robustness_summary.csv",
        "theme_feature_fairness_summary.csv",
        "theme_token_effect_summary.csv",
        "category_enrichment_improve.csv",
        "category_enrichment_worsen.csv",
        "category_enrichment_improve_vs_worsen.csv",
        "category_enrichment_worsen_vs_improve.csv",
        "classifier_feature_coefficients.csv",
        "classifier_cv_predictions.csv",
        "classifier_summary.csv",
        "feature_audit_numeric.csv",
        "token_audit_theme_mapping.csv",
        "README_results.txt",
        "plots",
    ]:
        print(" -", outdir / fn)


if __name__ == "__main__":
    main()
