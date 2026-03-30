# -*- coding: utf-8 -*-
"""
CNV (DNA copy number) -> IC50 (Y) XGBoost CV pipeline (drug-wise)
Designed to be used later as ONE omics-model in a multi-omics weighted ensemble.

Key goals for future ensemble:
1) Per-drug OOF predictions saved (for learning weights without leakage)
2) Per-drug folds saved (so other omics can reuse exact folds)
3) Per-fold model + preproc saved (selected genes, scalers, calibrator, params)
4) Optional feature selection (top-N correlation on TRAIN only)
5) Train-only standardization (X and optional y scaling)
6) Optional fold-wise linear calibration (inner-OOF on outer-train)

Default input files (same folder):
- sample.csv                 : CNV wide table (rows=cell lines, cols=genes)
- cell_line_mapping.csv      : target cell lines list
- processed_drug_response.csv: drug response long table (Drug, CellLineName, Y)
- good_drugs_summary.csv     : list of drugs to train

Run example:
python cnv_train.py --data-dir . --cnv-file DNACopyNumber_final.csv --mapping-file cell_line_mapping.csv --drug-file processed_drug_response.csv --good-drugs-file good_drugs_summary.csv --top-n 1000 --calibrate

Compare FS vs No-FS:
python cnv_train.py ... --top-n 1000   (with feature selection)
python cnv_train.py ... --top-n 0      (no feature selection)
"""

import os
import re
import json
import shutil
import zipfile
from datetime import datetime
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from scipy.stats import spearmanr, pearsonr, rankdata

import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import xgboost as xgb


# ---------------------------
# Metrics helpers
# ---------------------------
def mse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_spearman(y_true, y_pred) -> float:
    try:
        c = spearmanr(y_true, y_pred).correlation
        return float(c) if np.isfinite(c) else float("nan")
    except Exception:
        return float("nan")


def safe_pearson(y_true, y_pred) -> float:
    try:
        c = pearsonr(y_true, y_pred).statistic  # scipy>=1.10
        return float(c) if np.isfinite(c) else float("nan")
    except Exception:
        try:
            c, _ = pearsonr(y_true, y_pred)
            return float(c) if np.isfinite(c) else float("nan")
        except Exception:
            return float("nan")


# ---------------------------
# Text / ID normalization (match your mRNA style)
# ---------------------------
def standardize_name(name: str) -> Optional[str]:
    """
    Match mRNA-side normalization more closely so fold_map cell lines align across omics.

    Examples:
      HCC827_LUNG -> HCC827
      OCIAML2_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE -> OCIAML2
      A-549 -> A549
    """
    if pd.isna(name):
        return None
    x = str(name).strip().upper()
    x = re.sub(r"_(LUNG|BLOOD|MYELOID|LYMPHOID|HAEMATOPOIETIC|HAEMATOPOIETIC_AND_LYMPHOID_TISSUE|[A-Z]+)$", "", x)
    x = re.sub(r"[^A-Z0-9]", "", x)
    return x if x else None


def standardize_drug(drug: str) -> Optional[str]:
    """Normalize drug names so case/whitespace variants are treated as the same drug."""
    if pd.isna(drug):
        return None
    x = str(drug).strip()
    x = re.sub(r"\s+", " ", x)
    x = x.upper()
    return x if x else None


# ============================
# Fold-map reuse (by cell_line)
# ============================
def _drug_safe_name(drug: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(drug))


def load_fold_map(folds_root: str, drug_name: str) -> dict:
    """
    Load fold_map.csv for a drug.
    Backward-compatible with older folders that may differ only by case/whitespace.
    returns dict: std_cell_line -> fold_id (int, 1..K)
    """
    candidates = []
    for nm in [
        drug_name,
        standardize_drug(drug_name),
        str(drug_name).strip(),
        str(drug_name).strip().lower(),
        str(drug_name).strip().upper(),
    ]:
        if nm is None:
            continue
        candidates.append(_drug_safe_name(nm))

    seen = set()
    for d in candidates:
        if not d or d in seen:
            continue
        seen.add(d)
        path = os.path.join(folds_root, d, "fold_map.csv")
        if os.path.exists(path):
            fm = pd.read_csv(path)
            need = {"cell_line", "fold"}
            missing = need - set(fm.columns)
            if missing:
                raise ValueError(f"fold_map.csv missing columns {missing}: {path}")
            return dict(zip(fm["cell_line"].astype(str), fm["fold"].astype(int)))

    target = _drug_safe_name(standardize_drug(drug_name) or str(drug_name).strip())
    try:
        for sub in os.listdir(folds_root):
            sub_path = os.path.join(folds_root, sub)
            if not os.path.isdir(sub_path):
                continue
            if str(sub).upper() == str(target).upper():
                path = os.path.join(sub_path, "fold_map.csv")
                if os.path.exists(path):
                    fm = pd.read_csv(path)
                    need = {"cell_line", "fold"}
                    missing = need - set(fm.columns)
                    if missing:
                        raise ValueError(f"fold_map.csv missing columns {missing}: {path}")
                    return dict(zip(fm["cell_line"].astype(str), fm["fold"].astype(int)))
    except FileNotFoundError:
        pass

    raise FileNotFoundError(
        f"fold_map.csv not found for drug={drug_name} (normalized={target}) under folds_root={folds_root}"
    )


def folds_from_foldmap(sub_agg: pd.DataFrame, fold_map: dict, n_splits: int = 5):
    """
    sub_agg must contain 'std' column.
    Returns: (filtered_sub_agg, folds[(tr_idx, te_idx)], K)
    """
    sub_agg = sub_agg[sub_agg["std"].astype(str).isin(fold_map)].copy()
    if len(sub_agg) == 0:
        return sub_agg, [], 0

    fold_id = sub_agg["std"].astype(str).map(fold_map).to_numpy(dtype=int)
    K = int(np.max(fold_id))
    folds = []
    for f in range(1, K + 1):
        te = np.where(fold_id == f)[0]
        tr = np.where(fold_id != f)[0]
        folds.append((tr, te))
    return sub_agg, folds, K


def pick_first_existing(df: pd.DataFrame, candidates: List[str], fallback_first_col=True) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    if fallback_first_col:
        return df.columns[0]
    raise ValueError(f"None of candidates exist: {candidates}")




def _parse_tissue_from_ccle(ccle_name: str) -> str:
    if pd.isna(ccle_name):
        return ""
    x = str(ccle_name).strip().upper()
    if "_" not in x:
        return ""
    return x.split("_", 1)[1]


def _simplify_tissue(tissue_raw: str, pat_a, pat_b) -> str:
    t = str(tissue_raw or "").upper()
    if pat_a.search(t):
        return "A"
    if pat_b.search(t):
        return "B"
    return "OTHER"


def build_tissue_marker_set(
    feature_df: pd.DataFrame,
    tissue_map_csv: str,
    marker_d: float = 1.0,
    group_a_pat: str = r"LUNG",
    group_b_pat: str = r"HAEMATOPOIETIC|LYMPHOID|BLOOD|MYELOID",
    cell_col: str = "標準化名稱",
    ccle_col: str = "CCLE原始名稱",
) -> Tuple[set, pd.DataFrame]:
    """Compute tissue markers by Cohen's d using feature_df + tissue labels from map (never touches y)."""
    tm = pd.read_csv(tissue_map_csv, low_memory=False)

    if cell_col not in tm.columns:
        for cand in ["標準化名稱", "StrippedCellLineName", "CellLine", "cellline", "cell_line", "MODEL", "CellLineName", "standardized", "Unnamed: 0"]:
            if cand in tm.columns:
                cell_col = cand
                break
    if ccle_col not in tm.columns:
        for cand in ["CCLE原始名稱", "CCLEName", "CCLE", "ccle", "ccle_name", "CCLE_NAME"]:
            if cand in tm.columns:
                ccle_col = cand
                break
    if cell_col not in tm.columns or ccle_col not in tm.columns:
        raise ValueError(f"tissue map csv must contain columns for cell line and CCLE name (got: {tm.columns.tolist()[:20]})")

    tm["std"] = tm[cell_col].apply(standardize_name)
    tm["tissue_raw"] = tm[ccle_col].apply(_parse_tissue_from_ccle)

    pat_a = re.compile(group_a_pat, re.IGNORECASE)
    pat_b = re.compile(group_b_pat, re.IGNORECASE)
    tm["group"] = tm["tissue_raw"].apply(lambda t: _simplify_tissue(t, pat_a, pat_b))

    tm = tm[tm["group"].isin(["A", "B"])].dropna(subset=["std"]).copy()
    tm = tm[tm["std"].isin(feature_df.index)].drop_duplicates("std").copy()

    if tm["group"].nunique() < 2:
        raise ValueError("tissue map does not contain both groups after alignment to feature_df")

    X = feature_df.loc[tm["std"]].to_numpy(dtype=np.float64)
    lab = tm["group"].to_numpy(dtype=str)
    maskA = lab == "A"
    maskB = lab == "B"
    nA, nB = int(maskA.sum()), int(maskB.sum())
    if nA < 5 or nB < 5:
        raise ValueError(f"Too few samples for marker computation: nA={nA}, nB={nB}")

    XA = X[maskA, :]
    XB = X[maskB, :]
    meanA = np.nanmean(XA, axis=0)
    meanB = np.nanmean(XB, axis=0)
    varA = np.nanvar(XA, axis=0, ddof=1)
    varB = np.nanvar(XB, axis=0, ddof=1)
    pooled = np.sqrt(((nA - 1) * varA + (nB - 1) * varB) / max((nA + nB - 2), 1))
    d = (meanA - meanB) / pooled
    d[~np.isfinite(d)] = np.nan

    marker_tbl = pd.DataFrame({
        "feature": feature_df.columns.astype(str),
        "mean_A": meanA,
        "mean_B": meanB,
        "mean_diff(A-B)": meanA - meanB,
        "cohen_d(A-B)": d,
    })
    marker_tbl["marker_direction"] = np.where(marker_tbl["cohen_d(A-B)"] > 0, "A_high", "B_high")
    marker_tbl["is_marker"] = marker_tbl["cohen_d(A-B)"].abs() >= float(marker_d)

    marker_set = set(marker_tbl.loc[marker_tbl["is_marker"], "feature"].astype(str).tolist())
    return marker_set, marker_tbl

# ---------------------------
# Calibration (linear)
# ---------------------------
def fit_linear_calibrator(yhat_tr: np.ndarray, y_tr: np.ndarray) -> Tuple[float, float]:
    """
    Fit y ≈ a + b*yhat using least squares.
    """
    yhat_tr = np.asarray(yhat_tr, dtype=np.float64).ravel()
    y_tr = np.asarray(y_tr, dtype=np.float64).ravel()

    var = np.var(yhat_tr)
    if not np.isfinite(var) or var < 1e-12:
        a = float(np.mean(y_tr))
        b = 0.0
        return a, b

    cov = float(np.mean((yhat_tr - yhat_tr.mean()) * (y_tr - y_tr.mean())))
    b = cov / var
    a = float(np.mean(y_tr) - b * np.mean(yhat_tr))
    return a, float(b)


def apply_linear_calibrator(yhat: np.ndarray, a: float, b: float) -> np.ndarray:
    yhat = np.asarray(yhat, dtype=np.float64).ravel()
    return (a + b * yhat).astype(np.float64)


# ---------------------------
# Feature selection (train-only): top-N by corr
# ---------------------------
def topn_by_corr(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    n: int,
    method: str = "pearson",
    use_abs: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return top-N feature indices based on correlation computed ONLY on train (no leakage).
    method: 'pearson' or 'spearman'
    """
    if n <= 0:
        # special case: no feature selection
        idx = np.arange(X_tr.shape[1], dtype=np.int32)
        scores = np.zeros_like(idx, dtype=np.float32)
        return idx, scores

    n = int(min(n, X_tr.shape[1]))
    y0 = y_tr.astype(np.float64)

    if method == "pearson":
        y_center = y0 - y0.mean()
        X_center = X_tr.astype(np.float64) - X_tr.mean(axis=0, keepdims=True)
        num = (X_center * y_center[:, None]).sum(axis=0)
        den = np.sqrt((X_center**2).sum(axis=0) * (y_center**2).sum())
        corr = np.divide(num, den, out=np.zeros_like(num), where=(den != 0))

    elif method == "spearman":
        y_rank = rankdata(y0)
        X_rank = pd.DataFrame(X_tr).rank(axis=0, method="average").to_numpy()
        y_center = y_rank - y_rank.mean()
        X_center = X_rank - X_rank.mean(axis=0, keepdims=True)
        num = (X_center * y_center[:, None]).sum(axis=0)
        den = np.sqrt((X_center**2).sum(axis=0) * (y_center**2).sum())
        corr = np.divide(num, den, out=np.zeros_like(num), where=(den != 0))
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")

    scores = np.abs(corr) if use_abs else corr
    idx = np.argpartition(scores, -n)[-n:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return idx.astype(np.int32), scores[idx]


# ---------------------------
# CV split (StratifiedGroupKFold)
# ---------------------------
def make_folds(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
):
    """
    StratifiedGroupKFold needs discrete y-bins.
    """
    y_ser = pd.Series(y)
    try:
        y_bins = pd.qcut(y_ser, q=min(5, len(y_ser)), labels=False, duplicates="drop")
        if y_bins.nunique() < 2:
            raise ValueError("qcut produced <2 bins")
    except Exception:
        ranks = y_ser.rank(method="first")
        q = min(5, max(2, len(y_ser)//10))
        y_bins = pd.qcut(ranks, q=q, labels=False, duplicates="drop")

    y_bins = np.asarray(y_bins, dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(sgkf.split(X, y_bins, groups=groups))


# ---------------------------
# XGBoost params + predict helper
# ---------------------------
def build_xgb_params(use_gpu: bool) -> Dict:
    params = dict(
        n_estimators=1200,
        max_depth=8,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=0.5,
        reg_alpha=0.1,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )
    if use_gpu:
        params.update(dict(tree_method="hist", device="cuda"))
    else:
        params.update(dict(tree_method="hist"))
    return params


def predict_xgb(model: xgb.XGBRegressor, X: np.ndarray, use_gpu: bool) -> np.ndarray:
    """
    Avoid mismatched devices warning by using Booster.predict(DMatrix) on GPU.
    """
    X = np.asarray(X, dtype=np.float32)
    if use_gpu:
        booster = model.get_booster()
        dmat = xgb.DMatrix(X)
        return booster.predict(dmat)
    return model.predict(X)


# ---------------------------
# Plot
# ---------------------------
def save_scatter(path_png: str, y_true: np.ndarray, y_pred: np.ndarray,
                 title: str, subtitle: str = "", stats=None):
    """Save y_true vs y_pred scatter with optional stats box."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    plt.figure()
    plt.scatter(y_true, y_pred, s=12, alpha=0.6)

    try:
        mn = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)]))
        mx = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)]))
        if np.isfinite(mn) and np.isfinite(mx):
            plt.plot([mn, mx], [mn, mx], "r--")
    except Exception:
        pass

    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title(title)

    if subtitle:
        plt.suptitle(subtitle, y=0.94, fontsize=10)

    if stats is not None:
        def _fmt(v, nd=3):
            try:
                v = float(v)
                return "nan" if (not np.isfinite(v)) else f"{v:.{nd}f}"
            except Exception:
                return "nan"

        text_lines = [
            f"mean IC50={_fmt(stats.get('mean_ic50'), nd=4)}",
            f"R2={_fmt(stats.get('r2'))}   RMSE={_fmt(stats.get('rmse'))}",
            f"Spearman={_fmt(stats.get('spearman'))}   Pearson={_fmt(stats.get('pearson'))}",
        ]

        ax = plt.gca()
        ax.text(
            0.02, 0.98, "\n".join(text_lines),
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.75,
                edgecolor="gray"
            )
        )

    plt.tight_layout()
    plt.savefig(path_png, dpi=160)
    plt.close()


# ---------------------------
# Inner-OOF calibration
# ---------------------------
def inner_oof_predictions(
    X_tr: np.ndarray,
    y_tr_s: np.ndarray,
    y_tr_raw: np.ndarray,
    groups_tr: np.ndarray,
    xgb_params: Dict,
    use_gpu: bool,
    n_splits_inner: int,
    y_scaler: Optional[StandardScaler],
    scale_y: bool,
) -> Tuple[np.ndarray, bool]:
    """
    Returns (oof_pred_in_raw_scale, ok)
    """
    n = len(y_tr_raw)
    if n_splits_inner < 2 or n < (n_splits_inner * 2):
        return np.full(n, np.nan, dtype=np.float64), False

    try:
        inner_folds = make_folds(
            X_tr, y_tr_raw, groups_tr,
            n_splits=n_splits_inner,
            random_state=123
        )
    except Exception:
        return np.full(n, np.nan, dtype=np.float64), False

    oof = np.full(n, np.nan, dtype=np.float64)

    for itr, ival in inner_folds:
        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X_tr[itr], y_tr_s[itr])

        pred_val_s = predict_xgb(m, X_tr[ival], use_gpu=use_gpu)

        if scale_y and (y_scaler is not None):
            pred_val = y_scaler.inverse_transform(pred_val_s.reshape(-1, 1)).ravel()
        else:
            pred_val = pred_val_s

        oof[ival] = pred_val

    ok = np.isfinite(oof).sum() >= max(10, int(0.6 * n))
    return oof, ok


# ---------------------------
# Data loading
# ---------------------------
def load_target_cells(mapping_csv: str) -> set[str]:
    df = pd.read_csv(mapping_csv)
    col = pick_first_existing(
        df,
        ["standardized", "CellLineName", "cell_line", "Original_Name", "CCLE_Name", "DepMap_ID"],
        True
    )
    keys = set(df[col].map(standardize_name).dropna().astype(str))
    keys = {k for k in keys if k and k != "None"}
    return keys


def load_good_drugs(good_csv: str) -> List[str]:
    df = pd.read_csv(good_csv)
    col = pick_first_existing(df, ["drug", "Drug", "DrugName", "compound", "Compound"], True)
    drugs = df[col].dropna().astype(str).tolist()

    seen = set()
    out = []
    for d in drugs:
        nd = standardize_drug(d)
        if nd is None or nd in seen:
            continue
        seen.add(nd)
        out.append(nd)
    return out


def load_drug_response(drug_csv: str) -> pd.DataFrame:
    df = pd.read_csv(drug_csv)
    required = {"Drug", "CellLineName", "Y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{os.path.basename(drug_csv)} missing columns: {missing}")

    df = df[["Drug", "CellLineName", "Y"]].copy()
    df["Drug"] = df["Drug"].apply(standardize_drug)
    df["std"] = df["CellLineName"].apply(standardize_name)
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["Drug", "std", "Y"])
    return df


def load_cnv_wide(
    cnv_csv: str,
    target_keys: set[str],
    cnv_id_col: Optional[str],
    chunksize: int = 0
) -> pd.DataFrame:
    """
    Load CNV wide table and keep only target cell lines to reduce memory.
    Output index: 'std', columns: genes
    """
    head = pd.read_csv(cnv_csv, nrows=5)
    id_col = cnv_id_col or pick_first_existing(
        head,
        ["Original_Name", "CellLineName", "CCLE_Name", "DepMap_ID"],
        True
    )

    def proc(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["std"] = df[id_col].apply(standardize_name)
        df = df.dropna(subset=["std"])
        df = df[df["std"].isin(target_keys)]
        return df

    if chunksize and chunksize > 0:
        parts = []
        for chunk in pd.read_csv(cnv_csv, chunksize=chunksize):
            chunk = proc(chunk)
            if len(chunk) > 0:
                parts.append(chunk)
        if not parts:
            raise ValueError("No CNV rows matched target cell lines. Check mapping & naming.")
        cnv = pd.concat(parts, ignore_index=True)
    else:
        cnv = pd.read_csv(cnv_csv)
        cnv = proc(cnv)
        if len(cnv) == 0:
            raise ValueError("No CNV rows matched target cell lines. Check mapping & naming.")

    gene_cols = [c for c in cnv.columns if c not in [id_col, "std"]]
    cnv[gene_cols] = cnv[gene_cols].apply(pd.to_numeric, errors="coerce")

    # duplicate std -> mean
    cnv = cnv.groupby("std", as_index=False)[gene_cols].mean(numeric_only=True)

    # fill NaN with 0
    cnv[gene_cols] = cnv[gene_cols].fillna(0.0)

    cnv = cnv.set_index("std")
    return cnv


# ---------------------------
# Per-drug CV
# ---------------------------
def run_cv_for_drug(
    drug_name: str,
    cnv_df: pd.DataFrame,
    drug_df: pd.DataFrame,
    out_dir: str,
    *,
    top_n: int = 1000,
    corr_method: str = "pearson",
    use_abs_corr: bool = True,
    min_cell_lines: int = 30,
    scale_x: bool = True,
    scale_y: bool = True,
    n_splits: int = 5,
    calibrate: bool = True,
    use_gpu: bool = False,
    best_by: str = "RMSE",  # "RMSE" or "R2"
    calib_inner_splits: int = 3,
    folds_mode: str = "create",  # "create" or "reuse"
    folds_root: Optional[str] = None,
    marker_set: Optional[set] = None,
    ablation_mode: str = "baseline",
):
    # ----- build dataset -----
    sub = drug_df.loc[drug_df["Drug"] == drug_name, ["std", "Y"]].copy()
    if len(sub) == 0:
        return None

    # per (drug, cell line) median
    sub_agg = (
        sub.groupby("std", as_index=False)
           .agg(Y=("Y", "median"), n_records=("Y", "size"))
    )

    # intersect with cnv rows
    sub_agg = sub_agg[sub_agg["std"].isin(cnv_df.index)].copy()
    if len(sub_agg) < min_cell_lines:
        print(f"[SKIP] {drug_name}: only {len(sub_agg)} cell lines (<{min_cell_lines}) after matching CNV.")
        return None

    X = cnv_df.loc[sub_agg["std"]].to_numpy(dtype=np.float32)
    y = sub_agg["Y"].to_numpy(dtype=np.float32)
    groups = sub_agg["std"].to_numpy()
    gene_names = np.array(cnv_df.columns, dtype=object)

    marker_set = set(marker_set or set())
    ablation_mode = str(ablation_mode or "baseline").lower().strip()
    if ablation_mode not in ["baseline", "drop_nofill", "drop_refill", "only_markers"]:
        raise ValueError(f"Unknown ablation_mode={ablation_mode}. Use baseline/drop_nofill/drop_refill/only_markers")

    # ----- folds (create or reuse) -----
    if folds_mode == "reuse":
        if not folds_root:
            raise ValueError("folds_mode='reuse' but --folds-root is missing.")
        fold_map = load_fold_map(folds_root, drug_name)
        sub_agg, folds, _K = folds_from_foldmap(sub_agg, fold_map, n_splits=n_splits)

        if len(sub_agg) < min_cell_lines:
            print(f"[SKIP] {drug_name}: only {len(sub_agg)} cell lines (<{min_cell_lines}) after fold_map filtering.")
            return None
        if len(folds) == 0:
            print(f"[SKIP] {drug_name}: no valid folds after fold_map filtering.")
            return None

        X = cnv_df.loc[sub_agg["std"]].to_numpy(dtype=np.float32)
        y = sub_agg["Y"].to_numpy(dtype=np.float32)
        groups = sub_agg["std"].to_numpy()
    else:
        folds = make_folds(X, y, groups, n_splits=n_splits, random_state=42)
        if len(folds) == 0:
            print(f"[SKIP] {drug_name}: no valid folds generated.")
            return None

    # filter invalid folds
    valid_folds = []
    for tr, te in folds:
        if len(tr) >= 2 and len(te) >= 1:
            valid_folds.append((tr, te))
    if len(valid_folds) == 0:
        print(f"[SKIP] {drug_name}: no valid folds after filtering.")
        return None
    folds = valid_folds

    # save folds for future ensemble
    os.makedirs(out_dir, exist_ok=True)
    folds_npz = os.path.join(out_dir, "folds.npz")
    save_dict = {"n_splits": len(folds)}
    for i, (tr, te) in enumerate(folds):
        save_dict[f"tr_{i}"] = np.asarray(tr, dtype=np.int32)
        save_dict[f"te_{i}"] = np.asarray(te, dtype=np.int32)
    np.savez(folds_npz, **save_dict)

    # save fold_map.csv too (so other omics can reuse exact folds)
    fold_map_csv = os.path.join(out_dir, "fold_map.csv")
    fold_rows = []
    for fold_id, (_, te) in enumerate(folds, start=1):
        for j in te:
            fold_rows.append({
                "cell_line": str(groups[j]),
                "fold": int(fold_id),
            })
    pd.DataFrame(fold_rows).sort_values(["fold", "cell_line"]).to_csv(fold_map_csv, index=False)

    # ----- feature selection per fold (train-only) with optional tissue-marker ablation -----
    if ablation_mode == "only_markers":
        if len(marker_set) == 0:
            print(f"[SKIP] {drug_name}: marker_set is empty, cannot run only_markers.")
            return None
        marker_mask = np.array([g in marker_set for g in gene_names], dtype=bool)
        if int(marker_mask.sum()) < 5:
            print(f"[SKIP] {drug_name}: too few marker features present in CNV ({int(marker_mask.sum())}).")
            return None
        X_for_fs = X[:, marker_mask]
        gene_names_for_fs = gene_names[marker_mask]
    else:
        X_for_fs = X
        gene_names_for_fs = gene_names

    fold_feature_idx = []
    for i, (tr, te) in enumerate(folds, 1):
        n_req = int(min(top_n, X_for_fs.shape[1])) if top_n > 0 else int(X_for_fs.shape[1])
        idx_pool, sc = topn_by_corr(X_for_fs[tr], y[tr], n_req, method=corr_method, use_abs=use_abs_corr)

        if ablation_mode == "drop_nofill":
            keep = np.array([gene_names_for_fs[j] not in marker_set for j in idx_pool], dtype=bool)
            idx = idx_pool[keep]
            if len(idx) == 0:
                big_n = int(min(max(top_n * 5, top_n + 2000), X_for_fs.shape[1])) if top_n > 0 else int(X_for_fs.shape[1])
                idx_big = topn_by_corr(X_for_fs[tr], y[tr], big_n, method=corr_method, use_abs=use_abs_corr)[0]
                for j in idx_big:
                    if gene_names_for_fs[j] not in marker_set:
                        idx = np.array([j], dtype=int)
                        break
        elif ablation_mode == "drop_refill":
            big_n = int(min(max(top_n * 5, top_n + 5000), X_for_fs.shape[1])) if top_n > 0 else int(X_for_fs.shape[1])
            idx_big = topn_by_corr(X_for_fs[tr], y[tr], big_n, method=corr_method, use_abs=use_abs_corr)[0]
            keep = np.array([gene_names_for_fs[j] not in marker_set for j in idx_big], dtype=bool)
            idx = idx_big[keep][:n_req]
            if len(idx) == 0:
                idx = idx_pool
        else:
            idx = idx_pool

        fold_feature_idx.append(idx)
        if top_n > 0:
            top1 = gene_names_for_fs[idx[0]] if len(idx) > 0 else "NA"
            print(f"[{drug_name}] Fold {i}: selected {len(idx)} CNV genes | ablation={ablation_mode} | top1={top1}")
        else:
            print(f"[{drug_name}] Fold {i}: using all {len(idx)} CNV genes | ablation={ablation_mode}")

    X_src = X_for_fs

    # ----- prepare dirs -----
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    metric_rows = []
    pred_rows = []

    xgb_params = build_xgb_params(use_gpu)
    gpu_failed_once = False

    for i, ((tr, te), idx) in enumerate(zip(folds, fold_feature_idx), 1):
        # select features
        X_tr = X_src[tr][:, idx]
        X_te = X_src[te][:, idx]

        # scale X (train-only)
        x_scaler = None
        if scale_x:
            x_scaler = StandardScaler()
            X_tr = x_scaler.fit_transform(X_tr).astype(np.float32, copy=False)
            X_te = x_scaler.transform(X_te).astype(np.float32, copy=False)
        else:
            X_tr = X_tr.astype(np.float32, copy=False)
            X_te = X_te.astype(np.float32, copy=False)

        y_tr = y[tr].astype(np.float32, copy=False)
        y_te = y[te].astype(np.float32, copy=False)
        groups_tr = groups[tr]

        # scale y (train-only)
        y_scaler = None
        if scale_y:
            y_scaler = StandardScaler()
            y_tr_s = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel().astype(np.float32, copy=False)
        else:
            y_tr_s = y_tr

        # train model (GPU -> fallback CPU)
        model = xgb.XGBRegressor(**xgb_params)
        try:
            model.fit(X_tr, y_tr_s)
        except xgb.core.XGBoostError as e:
            if use_gpu and not gpu_failed_once:
                print("[WARN] GPU training failed, falling back to CPU. Error:", str(e)[:200])
                gpu_failed_once = True
                use_gpu = False
                xgb_params = build_xgb_params(False)
                model = xgb.XGBRegressor(**xgb_params)
                model.fit(X_tr, y_tr_s)
            else:
                raise

        # predict test
        pred_te_s = predict_xgb(model, X_te, use_gpu=use_gpu)

        # inverse to original y
        if scale_y and y_scaler is not None:
            pred_te = y_scaler.inverse_transform(pred_te_s.reshape(-1, 1)).ravel()
        else:
            pred_te = pred_te_s

        # ---- calibration (INNER-OOF on outer train) ----
        cal_a, cal_b = (0.0, 1.0)
        fit_mode = "disabled"
        pred_te_cal = pred_te

        if calibrate:
            oof_pred, ok = inner_oof_predictions(
                X_tr=X_tr,
                y_tr_s=y_tr_s,
                y_tr_raw=y_tr,
                groups_tr=groups_tr,
                xgb_params=xgb_params,
                use_gpu=use_gpu,
                n_splits_inner=calib_inner_splits,
                y_scaler=y_scaler,
                scale_y=scale_y,
            )
            if ok:
                msk = np.isfinite(oof_pred)
                cal_a, cal_b = fit_linear_calibrator(oof_pred[msk], y_tr[msk])
                fit_mode = f"inner_oof_{calib_inner_splits}"
            else:
                pred_tr_s = predict_xgb(model, X_tr, use_gpu=use_gpu)
                if scale_y and y_scaler is not None:
                    pred_tr = y_scaler.inverse_transform(pred_tr_s.reshape(-1, 1)).ravel()
                else:
                    pred_tr = pred_tr_s
                cal_a, cal_b = fit_linear_calibrator(pred_tr, y_tr)
                fit_mode = "train_insample_fallback"

            pred_te_cal = apply_linear_calibrator(pred_te, cal_a, cal_b)

        # metrics (raw vs calibrated)
        mse_raw = mse(y_te, pred_te)
        rmse_raw = float(np.sqrt(mse_raw))
        r2_raw = float(r2_score(y_te, pred_te))
        sp_raw = safe_spearman(y_te, pred_te)
        pr_raw = safe_pearson(y_te, pred_te)

        mse_cal = mse(y_te, pred_te_cal)
        rmse_cal = float(np.sqrt(mse_cal))
        r2_cal = float(r2_score(y_te, pred_te_cal))
        sp_cal = safe_spearman(y_te, pred_te_cal)
        pr_cal = safe_pearson(y_te, pred_te_cal)

        metric_rows.append({
            "drug": drug_name,
            "fold": i,
            "n_features": int(len(idx)),
            "MSE": mse_cal,
            "RMSE": rmse_cal,
            "R2": r2_cal,
            "Spearman": sp_cal,
            "Pearson": pr_cal,
            "MSE_raw": mse_raw,
            "RMSE_raw": rmse_raw,
            "R2_raw": r2_raw,
            "Spearman_raw": sp_raw,
            "Pearson_raw": pr_raw,
            "cal_a": float(cal_a),
            "cal_b": float(cal_b),
            "cal_fit_mode": fit_mode,
            "n": int(len(te)),
        })

        # final pred for ensemble usage
        y_pred_final = pred_te_cal if calibrate else pred_te

        if scale_y and y_scaler is not None:
            y_te_z = y_scaler.transform(y_te.reshape(-1, 1)).ravel()
        else:
            y_te_z = y_te

        for k, j in enumerate(te):
            pred_rows.append({
                "drug": drug_name,
                "fold": i,
                "cell_line": str(groups[j]),
                "y_true": float(y_te[k]),
                "y_pred": float(y_pred_final[k]),       # calibrated if enabled
                "y_pred_raw": float(pred_te[k]),
                "y_true_z(train_scaler)": float(y_te_z[k]),
                "y_pred_z(train_scaler)": float(pred_te_s[k]),
            })

        # ----- save model + preproc -----
        model_path = os.path.join(models_dir, f"xgb_model_fold{i}.json")
        preproc_path = os.path.join(models_dir, f"preproc_fold{i}.joblib")

        model.save_model(model_path)

        preproc = dict(
            omics="CNV",
            drug=drug_name,
            fold=int(i),
            cnv_gene_names=[str(g) for g in gene_names.tolist()],
            selected_gene_idx=np.asarray(idx, dtype=np.int32),
            selected_genes=[str(g) for g in gene_names[idx].tolist()],
            scale_x=bool(scale_x),
            scale_y=bool(scale_y),
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            calibrator=dict(
                enabled=bool(calibrate),
                type="linear",
                a=float(cal_a),
                b=float(cal_b),
                fit_mode=fit_mode,
                inner_splits=int(calib_inner_splits),
            ),
            xgb_params=xgb_params,
            folds_file=os.path.basename(folds_npz),
            fold_map_file=os.path.basename(fold_map_csv),
        )
        joblib.dump(preproc, preproc_path)

    df_cv = pd.DataFrame(metric_rows)
    pred_df = pd.DataFrame(pred_rows)

    # ----- best fold export -----
    best_fold_dir = os.path.join(out_dir, "best_fold")
    os.makedirs(best_fold_dir, exist_ok=True)

    if best_by.upper() == "R2":
        best_row = df_cv.sort_values(["R2", "RMSE"], ascending=[False, True]).iloc[0]
    else:
        best_row = df_cv.sort_values(["RMSE", "R2"], ascending=[True, False]).iloc[0]
    best_fold = int(best_row["fold"])

    shutil.copy2(
        os.path.join(models_dir, f"xgb_model_fold{best_fold}.json"),
        os.path.join(best_fold_dir, "best_model.json")
    )
    shutil.copy2(
        os.path.join(models_dir, f"preproc_fold{best_fold}.joblib"),
        os.path.join(best_fold_dir, "best_preproc.joblib")
    )
    with open(os.path.join(best_fold_dir, "best_fold.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "drug": drug_name,
                "best_fold": best_fold,
                "best_by": best_by,
                "best_metrics": best_row.to_dict(),
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    # ----- scatter (with stats box) -----
    y_true_all = pred_df["y_true"].to_numpy(dtype=np.float64)
    y_pred_all = pred_df["y_pred"].to_numpy(dtype=np.float64)
    mask = np.isfinite(y_true_all) & np.isfinite(y_pred_all)

    if int(mask.sum()) >= 2:
        y_true_m = y_true_all[mask]
        y_pred_m = y_pred_all[mask]
        mean_ic50 = float(np.nanmean(y_true_m))
        r2_all = float(r2_score(y_true_m, y_pred_m))
        rmse_all = rmse(y_true_m, y_pred_m)
        sp_all = safe_spearman(y_true_m, y_pred_m)
        pr_all = safe_pearson(y_true_m, y_pred_m)
    else:
        mean_ic50 = float("nan")
        r2_all = float("nan")
        rmse_all = float("nan")
        sp_all = float("nan")
        pr_all = float("nan")

    stats = {
        "mean_ic50": mean_ic50,
        "r2": r2_all,
        "rmse": rmse_all,
        "spearman": sp_all,
        "pearson": pr_all,
    }

    title = f"{drug_name} (CNV OOF across folds)"
    subtitle = (
        f"Final: RMSE_mean={df_cv['RMSE'].mean():.4f}, R2_mean={df_cv['R2'].mean():.4f}, "
        f"Spearman_mean={df_cv['Spearman'].mean():.4f}, Pearson_mean={df_cv['Pearson'].mean():.4f}"
    )
    save_scatter(
        os.path.join(out_dir, "scatter_ytrue_vs_ypred.png"),
        y_true_all,
        y_pred_all,
        title=title,
        subtitle=subtitle,
        stats=stats,
    )

    y_pred_raw_all = pred_df["y_pred_raw"].to_numpy(dtype=np.float64)
    mask_raw = np.isfinite(y_true_all) & np.isfinite(y_pred_raw_all)

    if int(mask_raw.sum()) >= 2:
        y_true_raw = y_true_all[mask_raw]
        y_pred_raw = y_pred_raw_all[mask_raw]
        stats_raw = {
            "mean_ic50": float(np.nanmean(y_true_raw)),
            "r2": float(r2_score(y_true_raw, y_pred_raw)),
            "rmse": rmse(y_true_raw, y_pred_raw),
            "spearman": safe_spearman(y_true_raw, y_pred_raw),
            "pearson": safe_pearson(y_true_raw, y_pred_raw),
        }
    else:
        stats_raw = {
            "mean_ic50": float("nan"),
            "r2": float("nan"),
            "rmse": float("nan"),
            "spearman": float("nan"),
            "pearson": float("nan"),
        }

    subtitle_raw = (
        f"Raw: RMSE_mean={df_cv['RMSE_raw'].mean():.4f}, R2_mean={df_cv['R2_raw'].mean():.4f}, "
        f"Spearman_mean={df_cv['Spearman_raw'].mean():.4f}, Pearson_mean={df_cv['Pearson_raw'].mean():.4f}"
    )
    save_scatter(
        os.path.join(out_dir, "scatter_ytrue_vs_ypred_raw.png"),
        y_true_all,
        y_pred_raw_all,
        title=f"{drug_name} (CNV raw OOF across folds)",
        subtitle=subtitle_raw,
        stats=stats_raw,
    )

    return df_cv, pred_df, fold_feature_idx, gene_names, sub_agg


# ---------------------------
# Entry
# ---------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser()

    # share folds across omics (by cell_line)
    ap.add_argument(
        "--folds-mode",
        type=str,
        default="create",
        choices=["create", "reuse"],
        help="create: cut folds in this run; reuse: load fold_map.csv from --folds-root",
    )
    ap.add_argument(
        "--folds-root",
        type=str,
        default=None,
        help="Root folder containing <drug>/fold_map.csv from your shared folds pipeline",
    )

    ap.add_argument("--data-dir", default=".", help="Folder containing input csv files")
    ap.add_argument("--cnv-file", default="DNACopyNumber_final.csv")
    ap.add_argument("--mapping-file", default="lung_and_blood_Cline.csv")
    ap.add_argument("--drug-file", default="processed_drug_response_with_prism.csv")
    ap.add_argument("--good-drugs-file", default="good_drugs_summary.csv",
                    help="CSV file containing a drug list. Uses column drug/Drug if present.")
    ap.add_argument("--drug-list-csv", dest="drug_list_csv", type=str, default=None,
                    help="Alias of --good-drugs-file. If omitted together with --good-drugs-file, select top-k drugs by sample count.")
    ap.add_argument("--top-k-drugs", type=int, default=10,
                    help="When no explicit drug list CSV is provided, train the top-k drugs ranked by number of unique cell lines in drug response.")

    ap.add_argument("--top-n", type=int, default=1000, help="Top-N corr features per fold. 0 = no feature selection")
    ap.add_argument("--corr-method", type=str, default="pearson", choices=["pearson", "spearman"])
    ap.add_argument("--use-abs-corr", action="store_true", default=True)
    ap.add_argument("--min-cell-lines", type=int, default=30)

    ap.add_argument("--no-scale-x", action="store_true", help="Disable train-only z-score for X")
    ap.add_argument("--no-scale-y", action="store_true", help="Disable train-only z-score for y")

    ap.add_argument("--calibrate", action="store_true", default=True, help="Enable fold-wise linear calibrator")
    ap.add_argument("--no-calibrate", action="store_true", help="Disable calibrator")
    ap.add_argument("--calib-inner-splits", type=int, default=3)
    ap.add_argument("--use-gpu", action="store_true", default=False)

    ap.add_argument("--best-by", type=str, default="RMSE", choices=["RMSE", "R2"])
    ap.add_argument("--good-r2", type=float, default=0.30)
    ap.add_argument("--good-spearman", type=float, default=0.50)
    ap.add_argument("--good-topk-fallback", type=int, default=4)

    ap.add_argument("--cnv-id-col", type=str, default=None, help="Override CNV ID column name (e.g., Original_Name)")
    ap.add_argument("--cnv-chunksize", type=int, default=0, help="Chunk reading for huge CNV CSV. 0=off")
    ap.add_argument("--only-drug", type=str, default=None, help="Train only one specific drug name")

    ap.add_argument("--ablation-mode", type=str, default="baseline",
                    choices=["baseline", "drop_nofill", "drop_refill", "only_markers"],
                    help="How to apply tissue markers during feature selection")
    ap.add_argument("--tissue-map-csv", type=str, default=None,
                    help="CSV containing cell line + CCLE name columns for tissue-marker computation")
    ap.add_argument("--marker-d", type=float, default=1.0,
                    help="Cohen's d threshold used to define tissue markers")
    ap.add_argument("--group-a-pat", type=str, default=r"LUNG")
    ap.add_argument("--group-b-pat", type=str, default=r"HAEMATOPOIETIC|LYMPHOID|BLOOD|MYELOID")

    args = ap.parse_args()

    DATA_DIR = args.data_dir
    cnv_path = os.path.join(DATA_DIR, args.cnv_file)
    mapping_path = os.path.join(DATA_DIR, args.mapping_file)
    drug_path = os.path.join(DATA_DIR, args.drug_file)
    good_csv = args.drug_list_csv if args.drug_list_csv is not None else args.good_drugs_file
    good_path = os.path.join(DATA_DIR, good_csv) if good_csv else None

    print("DATA_DIR =", DATA_DIR)
    print("cnv_path =", cnv_path)
    print("mapping_path =", mapping_path)
    print("drug_path =", drug_path)
    print("good_path =", good_path)

    # ----- load -----
    target_keys = load_target_cells(mapping_path)
    drug_df = load_drug_response(drug_path)
    cnv_df = load_cnv_wide(
        cnv_path,
        target_keys,
        cnv_id_col=args.cnv_id_col,
        chunksize=args.cnv_chunksize
    )

    print("target_keys:", len(target_keys))
    print("drug_df:", drug_df.shape)
    print("cnv_df:", cnv_df.shape)

    marker_set = set()
    marker_tbl = None
    if args.ablation_mode != "baseline":
        tissue_map_csv = args.tissue_map_csv or mapping_path
        if not os.path.exists(tissue_map_csv):
            raise FileNotFoundError(f"tissue-map-csv not found: {tissue_map_csv}")
        print(f"[INFO] Building tissue marker set from: {tissue_map_csv} | marker_d={args.marker_d}")
        marker_set, marker_tbl = build_tissue_marker_set(
            cnv_df,
            tissue_map_csv=tissue_map_csv,
            marker_d=args.marker_d,
            group_a_pat=args.group_a_pat,
            group_b_pat=args.group_b_pat,
        )
        print(f"[INFO] Tissue marker features: {len(marker_set)} / {cnv_df.shape[1]}")

    if args.only_drug:
        drugs = [standardize_drug(args.only_drug)]
    elif good_path:
        drugs = load_good_drugs(good_path)
        avail = set(drug_df["Drug"].dropna().astype(str).unique())
        drugs = [d for d in drugs if d in avail]
        if not drugs:
            raise RuntimeError("No drugs from the provided drug list matched drug response.")
    else:
        drugs = (
            drug_df.groupby("Drug")["std"]
            .nunique()
            .sort_values(ascending=False)
            .head(args.top_k_drugs)
            .index.tolist()
        )

    if not drugs:
        raise RuntimeError("No drugs selected for training.")

    # ----- run root -----
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    calibrate = (args.calibrate and not args.no_calibrate)
    scale_x = (not args.no_scale_x)
    scale_y = (not args.no_scale_y)

    RESULTS_ROOT = os.path.join(
        DATA_DIR,
        "results_cnv",
        f"CNV__top{args.top_n}__{args.corr_method}__abs{int(args.use_abs_corr)}__abl{args.ablation_mode}__d{str(args.marker_d).replace('.','p')}__cal{int(calibrate)}__gpu{int(args.use_gpu)}__{run_id}",
    )
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    print("RESULTS_ROOT =", RESULTS_ROOT)

    if marker_tbl is not None:
        marker_tbl.to_csv(os.path.join(RESULTS_ROOT, "tissue_marker_table.csv"), index=False)
        with open(os.path.join(RESULTS_ROOT, "tissue_marker_meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "ablation_mode": args.ablation_mode,
                "marker_d": float(args.marker_d),
                "group_a_pat": args.group_a_pat,
                "group_b_pat": args.group_b_pat,
                "n_markers": int(len(marker_set)),
            }, f, indent=2, ensure_ascii=False)

    all_cv = []
    all_pred = []
    skipped = []
    audit_rows = []

    pdf_path = os.path.join(RESULTS_ROOT, "scatter_all_drugs.pdf")
    pdf = PdfPages(pdf_path)

    for d in drugs:
        print("\n" + "=" * 80)
        print("Running drug:", d)

        d_dir = os.path.join(RESULTS_ROOT, d.replace("/", "_"))
        os.makedirs(d_dir, exist_ok=True)

        out = run_cv_for_drug(
            d,
            cnv_df,
            drug_df,
            d_dir,
            top_n=args.top_n,
            corr_method=args.corr_method,
            use_abs_corr=args.use_abs_corr,
            min_cell_lines=args.min_cell_lines,
            scale_x=scale_x,
            scale_y=scale_y,
            n_splits=5,
            calibrate=calibrate,
            use_gpu=args.use_gpu,
            best_by=args.best_by,
            calib_inner_splits=int(args.calib_inner_splits) if args.calib_inner_splits else 0,
            folds_mode=args.folds_mode,
            folds_root=args.folds_root,
            marker_set=marker_set,
            ablation_mode=args.ablation_mode,
        )
        if out is None:
            skipped.append(d)
            audit_rows.append({"drug": d, "status": "skipped"})
            continue

        df_cv, pred_df, fold_feature_idx, gene_names, meta = out
        audit_rows.append({
            "drug": d,
            "status": "ok",
            "n_matched_cell_lines": int(len(meta)),
            "n_folds": int(df_cv["fold"].nunique()) if "fold" in df_cv.columns else np.nan,
            "mean_R2": float(df_cv["R2"].mean()) if "R2" in df_cv.columns and len(df_cv) else np.nan,
            "mean_RMSE": float(df_cv["RMSE"].mean()) if "RMSE" in df_cv.columns and len(df_cv) else np.nan,
        })
        all_cv.append(df_cv)
        all_pred.append(pred_df)

        df_cv.to_csv(os.path.join(d_dir, "cv_metrics_folds.csv"), index=False)
        pred_df.to_csv(os.path.join(d_dir, "cv_predictions.csv"), index=False)
        meta.to_csv(os.path.join(d_dir, "matched_celllines_meta.csv"), index=False)

        for i, idx in enumerate(fold_feature_idx, 1):
            genes = gene_names[idx].astype(str).tolist()
            with open(os.path.join(d_dir, f"selected_genes_fold{i}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(genes))

        config = {
            "OMICS": "CNV",
            "DRUG": d,
            "TOP_N": int(args.top_n),
            "CORR_METHOD": args.corr_method,
            "USE_ABS_CORR": bool(args.use_abs_corr),
            "MIN_CELL_LINES": int(args.min_cell_lines),
            "SCALE_X": bool(scale_x),
            "SCALE_Y": bool(scale_y),
            "CALIBRATE": bool(calibrate),
            "CALIB_INNER_SPLITS": int(args.calib_inner_splits),
            "USE_GPU": bool(args.use_gpu),
            "BEST_BY": args.best_by,
            "ABLATION_MODE": args.ablation_mode,
            "MARKER_D": float(args.marker_d),
        }
        with open(os.path.join(d_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        scatter_png = os.path.join(d_dir, "scatter_ytrue_vs_ypred.png")
        if os.path.exists(scatter_png):
            img = plt.imread(scatter_png)
            plt.figure()
            plt.imshow(img)
            plt.axis("off")
            plt.title(d)
            pdf.savefig()
            plt.close()

    pdf.close()

    audit_df = pd.DataFrame(audit_rows)
    audit_path = os.path.join(RESULTS_ROOT, "drug_run_audit.csv")
    audit_df.to_csv(audit_path, index=False)

    if len(all_cv) == 0:
        raise RuntimeError("All drugs were skipped. Check MIN_CELL_LINES, mapping, or CNV naming alignment.")

    cv_all = pd.concat(all_cv, ignore_index=True)
    pred_all = pd.concat(all_pred, ignore_index=True)

    summary = (
        cv_all.groupby("drug")[
            ["MSE", "RMSE", "R2", "Spearman", "Pearson",
             "MSE_raw", "RMSE_raw", "R2_raw", "Spearman_raw", "Pearson_raw", "n"]
        ]
        .agg(["mean", "std", "sum"])
    )
    summary.columns = ["_".join([a, b]) for a, b in summary.columns]
    summary = summary.reset_index()

    summary_path = os.path.join(RESULTS_ROOT, "summary.csv")
    pred_path = os.path.join(RESULTS_ROOT, "all_predictions.csv")
    cv_path = os.path.join(RESULTS_ROOT, "all_cv_folds.csv")

    summary.to_csv(summary_path, index=False)
    pred_all.to_csv(pred_path, index=False)
    cv_all.to_csv(cv_path, index=False)

    print("\nSaved combined outputs:")
    print(" -", summary_path)
    print(" -", pred_path)
    print(" -", cv_path)
    print(" -", pdf_path)
    print("\nSkipped drugs:", skipped)

    # ---------------------------
    # Export "good" models
    # ---------------------------
    good = summary[
        (summary["R2_mean"] >= args.good_r2) &
        (summary["Spearman_mean"] >= args.good_spearman)
    ].copy()
    used_fallback = False
    if len(good) == 0:
        good = summary.sort_values("RMSE_mean", ascending=True).head(args.good_topk_fallback).copy()
        used_fallback = True

    good_drugs_csv = good[
        ["drug", "RMSE_mean", "RMSE_std", "R2_mean", "R2_std",
         "Spearman_mean", "Spearman_std", "Pearson_mean", "Pearson_std", "n_sum"]
    ].copy()
    good_drugs_csv = good_drugs_csv.rename(columns={"n_sum": "total_samples"})
    good_drugs_csv = good_drugs_csv.sort_values("RMSE_mean", ascending=True)
    good_drugs_path = os.path.join(RESULTS_ROOT, "good_drugs_summary.csv")
    good_drugs_csv.to_csv(good_drugs_path, index=False)

    print(f"\n{'='*50}")
    print(f"CNV 預測較好的藥物 (共 {len(good)} 種):")
    if used_fallback:
        print(
            f"  (沒有藥物達到 R2>={args.good_r2} & Spearman>={args.good_spearman}，"
            f"改用 RMSE 最低的 {args.good_topk_fallback} 種)"
        )
    else:
        print(f"  (篩選條件: R2>={args.good_r2} & Spearman>={args.good_spearman})")
    print("-" * 50)
    for _, r in good_drugs_csv.iterrows():
        print(
            f"  {r['drug']:30s}  RMSE={r['RMSE_mean']:.4f}  "
            f"R2={r['R2_mean']:.4f}  Spearman={r['Spearman_mean']:.4f}"
        )
    print(f"{'='*50}")
    print(f"已儲存至: {good_drugs_path}")

    best_models_dir = os.path.join(RESULTS_ROOT, "best_models")
    os.makedirs(best_models_dir, exist_ok=True)

    manifest_rows = []
    for _, row in good.iterrows():
        drug = row["drug"]
        src_best = os.path.join(RESULTS_ROOT, drug.replace("/", "_"), "best_fold")
        if not os.path.exists(src_best):
            continue

        dst = os.path.join(best_models_dir, drug.replace("/", "_"))
        os.makedirs(dst, exist_ok=True)

        shutil.copy2(os.path.join(src_best, "best_model.json"), os.path.join(dst, "best_model.json"))
        shutil.copy2(os.path.join(src_best, "best_preproc.joblib"), os.path.join(dst, "best_preproc.joblib"))
        shutil.copy2(os.path.join(src_best, "best_fold.json"), os.path.join(dst, "best_fold.json"))

        manifest_rows.append({
            "drug": drug,
            "RMSE_mean": float(row["RMSE_mean"]),
            "R2_mean": float(row["R2_mean"]),
            "Spearman_mean": float(row["Spearman_mean"]),
            "Pearson_mean": float(row["Pearson_mean"]),
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(best_models_dir, "best_models_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    zip_path = os.path.join(RESULTS_ROOT, "best_models.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(best_models_dir):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, RESULTS_ROOT)
                zf.write(full, rel)

    print("\nExported best models:")
    print(" -", best_models_dir)
    print(" -", manifest_path)
    print(" -", zip_path)


if __name__ == "__main__":
    main()