# -*- coding: utf-8 -*-
"""
miRNA -> IC50 (Y) XGBoost CV pipeline (drug-wise)

This script mirrors your methylation_train_foldmap_fixed.py but for miRNA data (wide matrix):
- Same cell-line standardization (remove all non-alphanumeric)
- Same per-drug 5-fold CV outputs (cv_metrics_folds.csv, cv_predictions.csv, folds.npz, models/)
- Supports folds_mode="reuse" using per-drug fold_map.csv (cell_line(std) -> fold) so ALL omics share EXACT folds
- Train-only mean imputation -> train-only correlation FS (top-N) -> train-only scaling -> XGB -> optional inner-OOF linear calibration

Expected inputs in --data-dir:
- miRNA_final.csv              : wide table (rows=cell lines, cols=miRNAs). First col is usually "Unnamed: 0"
- cell_line_mapping.csv        : list of target cell lines (533)
- processed_drug_response.csv  : long table with columns [Drug, CellLineName, Y]
- good_drugs_summary.csv       : list of drugs to train (column 'drug')

Typical usage (reuse folds exported from mRNA run into folds_root/<drug>/fold_map.csv):
python mirna_train_foldmap_fixed.py --data-dir . --mirna-file miRNA_final.csv --mapping-file cell_line_mapping.csv ^
  --drug-file processed_drug_response.csv --good-drugs-file good_drugs_summary.csv ^
  --folds-mode reuse --folds-root folds_root --top-n 300 --corr-method pearson --use-gpu --calibrate
"""

import os
import re
import json
import shutil
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
# Text / ID normalization
# ---------------------------
def standardize_name(name: str) -> Optional[str]:
    """Strip/upper; if '_' exists keep left part; remove all non-alphanumeric."""
    if pd.isna(name):
        return None
    name = str(name).strip().upper()
    if "_" in name:
        name = name.split("_")[0]
    name = re.sub(r"[^A-Z0-9]", "", name)
    return name if name else None



def standardize_drug(drug: str) -> Optional[str]:
    """Normalize drug names so case/whitespace variants are treated as the same drug."""
    if pd.isna(drug):
        return None
    x = str(drug).strip()
    x = re.sub(r"\s+", " ", x)
    x = x.upper()
    return x if x else None


def drug_safe_name(drug: str) -> str:
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
        candidates.append(drug_safe_name(nm))

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

    target = drug_safe_name(standardize_drug(drug_name) or str(drug_name).strip())
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


def folds_from_foldmap(sub_agg: pd.DataFrame, fold_map: dict):
    """
    sub_agg must contain 'std' and 'Y'. Returns (filtered_sub_agg, folds, K).
    Folds are built so that each std cell line belongs to exactly one fold.
    """
    sub_agg = sub_agg[sub_agg["std"].astype(str).isin(fold_map)].copy()
    sub_agg = sub_agg.sort_values("std").reset_index(drop=True)

    fold_id = sub_agg["std"].astype(str).map(fold_map).to_numpy(dtype=int)
    K = int(fold_id.max())
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


# ---------------------------
# Calibration (linear)
# ---------------------------
def fit_linear_calibrator(yhat_tr: np.ndarray, y_tr: np.ndarray) -> Tuple[float, float]:
    """Fit y ≈ a + b*yhat using least squares."""
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
# Train-only mean imputation
# ---------------------------
def impute_train_mean(X_tr: np.ndarray, X_te: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replace NaN using TRAIN column means (no leakage).
    If a column is all-NaN in train, mean becomes 0.
    """
    X_tr = np.asarray(X_tr, dtype=np.float32)
    X_te = np.asarray(X_te, dtype=np.float32)

    col_mean = np.nanmean(X_tr, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0).astype(np.float32)

    tr_nan = np.isnan(X_tr)
    if tr_nan.any():
        X_tr = X_tr.copy()
        X_tr[tr_nan] = np.take(col_mean, np.where(tr_nan)[1])

    te_nan = np.isnan(X_te)
    if te_nan.any():
        X_te = X_te.copy()
        X_te[te_nan] = np.take(col_mean, np.where(te_nan)[1])

    return X_tr, X_te


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
def make_folds(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5, random_state: int = 42):
    """StratifiedGroupKFold needs discrete y-bins."""
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
    """Avoid mismatched devices warning by using Booster.predict(DMatrix) on GPU."""
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
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", alpha=0.75, edgecolor="gray")
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
        inner_folds = make_folds(X_tr, y_tr_raw, groups_tr, n_splits=n_splits_inner, random_state=123)
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
    col = pick_first_existing(df, ["standardized", "CellLineName", "cell_line", "Original_Name", "CCLE_Name", "DepMap_ID"], True)
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


def load_mirna_wide(mirna_csv: str, target_keys: set[str], id_col: Optional[str]) -> pd.DataFrame:
    """
    Load miRNA wide table and keep only target cell lines.
    Output index: 'std', columns: miRNAs
    """
    head = pd.read_csv(mirna_csv, nrows=5)
    id_col = id_col or pick_first_existing(
        head,
        ["Unnamed: 0", "CellLineName", "CCLENameStripped", "StrippedName", "Original_Name"],
        True
    )

    df = pd.read_csv(mirna_csv)
    df = df.copy()
    df["std"] = df[id_col].apply(standardize_name)
    df = df.dropna(subset=["std"])
    df = df[df["std"].isin(target_keys)]
    if len(df) == 0:
        raise ValueError("No miRNA rows matched target cell lines. Check mapping & naming.")

    mirna_cols = [c for c in df.columns if c not in [id_col, "std"]]
    df[mirna_cols] = df[mirna_cols].apply(pd.to_numeric, errors="coerce")

    # duplicate std -> mean
    df = df.copy()
    df = df.groupby("std", as_index=False)[mirna_cols].mean(numeric_only=True)

    df = df.set_index("std")
    return df


# ---------------------------
# Per-drug CV
# ---------------------------

def run_cv_for_drug(
    drug_name: str,
    mirna_df: pd.DataFrame,
    drug_df: pd.DataFrame,
    out_dir: str,
    *,
    top_n: int = 300,
    corr_method: str = "pearson",
    use_abs_corr: bool = True,
    min_cell_lines: int = 30,
    scale_x: bool = True,
    scale_y: bool = True,
    n_splits: int = 5,
    calibrate: bool = True,
    use_gpu: bool = False,
    best_by: str = "RMSE",
    calib_inner_splits: int = 3,
    folds_mode: str = "create",
    folds_root: Optional[str] = None,
):
    sub = drug_df.loc[drug_df["Drug"] == drug_name, ["std", "Y"]].copy()
    if len(sub) == 0:
        return None

    sub_agg = (
        sub.groupby("std", as_index=False)
           .agg(Y=("Y", "median"), n_records=("Y", "size"))
    )
    sub_agg = sub_agg[sub_agg["std"].isin(mirna_df.index)].copy()
    if len(sub_agg) < min_cell_lines:
        print(f"[SKIP] {drug_name}: only {len(sub_agg)} cell lines (<{min_cell_lines}) after matching miRNA.")
        return None

    X = mirna_df.loc[sub_agg["std"]].to_numpy(dtype=np.float32)
    y = sub_agg["Y"].to_numpy(dtype=np.float32)
    groups = sub_agg["std"].to_numpy()
    feat_names = np.array(mirna_df.columns, dtype=object)

    if folds_mode == "reuse":
        if not folds_root:
            raise ValueError("folds_mode='reuse' but --folds-root is missing.")
        fold_map = load_fold_map(folds_root, drug_name)
        sub_agg, folds, _K = folds_from_foldmap(sub_agg, fold_map)

        if len(sub_agg) < min_cell_lines:
            print(f"[SKIP] {drug_name}: only {len(sub_agg)} cell lines (<{min_cell_lines}) after fold_map filtering.")
            return None
        if len(folds) == 0:
            print(f"[SKIP] {drug_name}: no valid folds after fold_map filtering.")
            return None

        X = mirna_df.loc[sub_agg["std"]].to_numpy(dtype=np.float32)
        y = sub_agg["Y"].to_numpy(dtype=np.float32)
        groups = sub_agg["std"].to_numpy()
    else:
        folds = make_folds(X, y, groups, n_splits=n_splits, random_state=42)
        if len(folds) == 0:
            print(f"[SKIP] {drug_name}: no valid folds generated.")
            return None

    valid_folds = []
    for tr, te in folds:
        if len(tr) >= 2 and len(te) >= 1:
            valid_folds.append((tr, te))
    if len(valid_folds) == 0:
        print(f"[SKIP] {drug_name}: no valid folds after filtering.")
        return None
    folds = valid_folds

    os.makedirs(out_dir, exist_ok=True)

    folds_npz = os.path.join(out_dir, "folds.npz")
    save_dict = {"n_splits": len(folds)}
    for i, (tr, te) in enumerate(folds):
        save_dict[f"tr_{i}"] = np.asarray(tr, dtype=np.int32)
        save_dict[f"te_{i}"] = np.asarray(te, dtype=np.int32)
    np.savez(folds_npz, **save_dict)

    fold_map_csv = os.path.join(out_dir, "fold_map.csv")
    fold_rows = []
    for fold_id, (_, te) in enumerate(folds, start=1):
        for j in te:
            fold_rows.append({
                "cell_line": str(groups[j]),
                "fold": int(fold_id),
            })
    pd.DataFrame(fold_rows).sort_values(["fold", "cell_line"]).to_csv(fold_map_csv, index=False)

    fold_feature_idx = []
    for i, (tr, te) in enumerate(folds, 1):
        X_tr_imp, _ = impute_train_mean(X[tr], X[te])
        idx, sc = topn_by_corr(X_tr_imp, y[tr], top_n, method=corr_method, use_abs=use_abs_corr)
        fold_feature_idx.append(idx)
        if top_n > 0:
            print(f"[{drug_name}] Fold {i}: selected {len(idx)} miRNAs, top1={feat_names}[idx[0]], score={sc[0]:.4f}")
        else:
            print(f"[{drug_name}] Fold {i}: NO feature selection, using all {len(idx)} miRNAs")

    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    metric_rows = []
    pred_rows = []

    xgb_params = build_xgb_params(use_gpu)
    gpu_failed_once = False

    for i, ((tr, te), idx) in enumerate(zip(folds, fold_feature_idx), 1):
        X_tr = X[tr][:, idx]
        X_te = X[te][:, idx]

        X_tr, X_te = impute_train_mean(X_tr, X_te)

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

        y_scaler = None
        if scale_y:
            y_scaler = StandardScaler()
            y_tr_s = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel().astype(np.float32, copy=False)
        else:
            y_tr_s = y_tr

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

        pred_te_s = predict_xgb(model, X_te, use_gpu=use_gpu)
        if scale_y and y_scaler is not None:
            pred_te = y_scaler.inverse_transform(pred_te_s.reshape(-1, 1)).ravel()
        else:
            pred_te = pred_te_s

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

        y_pred_final = pred_te_cal if calibrate else pred_te

        for k, j in enumerate(te):
            pred_rows.append({
                "drug": drug_name,
                "fold": i,
                "cell_line": str(groups[j]),
                "y_true": float(y_te[k]),
                "y_pred": float(y_pred_final[k]),
                "y_pred_raw": float(pred_te[k]),
            })

        model_path = os.path.join(models_dir, f"xgb_model_fold{i}.json")
        preproc_path = os.path.join(models_dir, f"preproc_fold{i}.joblib")

        model.save_model(model_path)

        preproc = dict(
            omics="miRNA",
            drug=drug_name,
            fold=int(i),
            feature_names=[str(g) for g in feat_names.tolist()],
            selected_feature_idx=np.asarray(idx, dtype=np.int32),
            selected_features=[str(g) for g in feat_names[idx].tolist()],
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

    best_fold_dir = os.path.join(out_dir, "best_fold")
    os.makedirs(best_fold_dir, exist_ok=True)

    if best_by.upper() == "R2":
        best_row = df_cv.sort_values(["R2", "RMSE"], ascending=[False, True]).iloc[0]
    else:
        best_row = df_cv.sort_values(["RMSE", "R2"], ascending=[True, False]).iloc[0]
    best_fold = int(best_row["fold"])

    shutil.copy2(os.path.join(models_dir, f"xgb_model_fold{best_fold}.json"),
                 os.path.join(best_fold_dir, "best_model.json"))
    shutil.copy2(os.path.join(models_dir, f"preproc_fold{best_fold}.joblib"),
                 os.path.join(best_fold_dir, "best_preproc.joblib"))
    with open(os.path.join(best_fold_dir, "best_fold.json"), "w", encoding="utf-8") as f:
        json.dump({"drug": drug_name, "best_fold": best_fold, "best_by": best_by, "best_metrics": best_row.to_dict()},
                  f, indent=2, ensure_ascii=False)

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

    title = f"{drug_name} (miRNA OOF across folds)"
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
        title=f"{drug_name} (miRNA raw OOF across folds)",
        subtitle=subtitle_raw,
        stats=stats_raw,
    )

    return df_cv, pred_df, fold_feature_idx, feat_names, sub_agg


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".", help="Folder containing input csv files")
    ap.add_argument("--mirna-file", default="miRNA_final.csv")
    ap.add_argument("--mapping-file", default="cell_line_mapping.csv")
    ap.add_argument("--drug-file", default="processed_drug_response.csv")
    ap.add_argument("--good-drugs-file", default="good_drugs_summary.csv")

    ap.add_argument("--top-n", type=int, default=300, help="Top-N corr features per fold. 0 = no FS")
    ap.add_argument("--corr-method", type=str, default="pearson", choices=["pearson", "spearman"])
    ap.add_argument("--use-abs-corr", action="store_true", default=True)
    ap.add_argument("--min-cell-lines", type=int, default=30)

    ap.add_argument("--no-scale-x", action="store_true", help="Disable train-only z-score for X")
    ap.add_argument("--no-scale-y", action="store_true", help="Disable train-only z-score for y")

    ap.add_argument("--calibrate", action="store_true", default=True)
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--calib-inner-splits", type=int, default=3)
    ap.add_argument("--use-gpu", action="store_true", default=False)

    ap.add_argument("--best-by", type=str, default="RMSE", choices=["RMSE", "R2"])
    ap.add_argument("--mirna-id-col", type=str, default=None, help="Override miRNA ID column name")
    ap.add_argument("--only-drug", type=str, default=None, help="Train only one specific drug name")

    # share folds across omics
    ap.add_argument("--folds-mode", type=str, default="create", choices=["create", "reuse"])
    ap.add_argument("--folds-root", type=str, default=None,
                    help="When folds-mode=reuse: root dir that contains <drug>/fold_map.csv")
    args = ap.parse_args()

    DATA_DIR = args.data_dir
    mirna_path = os.path.join(DATA_DIR, args.mirna_file)
    mapping_path = os.path.join(DATA_DIR, args.mapping_file)
    drug_path = os.path.join(DATA_DIR, args.drug_file)
    good_path = os.path.join(DATA_DIR, args.good_drugs_file)

    print("DATA_DIR =", DATA_DIR)
    print("mirna_path =", mirna_path)
    print("mapping_path =", mapping_path)
    print("drug_path =", drug_path)
    print("good_path =", good_path)

    target_keys = load_target_cells(mapping_path)
    drug_df = load_drug_response(drug_path)
    mirna_df = load_mirna_wide(mirna_path, target_keys, id_col=args.mirna_id_col)

    print("target_keys:", len(target_keys))
    print("drug_df:", drug_df.shape)
    print("mirna_df:", mirna_df.shape)

    drugs = load_good_drugs(good_path)
    if args.only_drug:
        drugs = [standardize_drug(args.only_drug)]
    if not drugs:
        raise RuntimeError("No drugs found in good-drugs file.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    calibrate = (args.calibrate and not args.no_calibrate)
    scale_x = (not args.no_scale_x)
    scale_y = (not args.no_scale_y)

    RESULTS_ROOT = os.path.join(
        DATA_DIR,
        "results_mirna",
        f"MIRNA__top{args.top_n}__{args.corr_method}__abs{int(args.use_abs_corr)}__cal{int(calibrate)}__gpu{int(args.use_gpu)}__{run_id}",
    )
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    print("RESULTS_ROOT =", RESULTS_ROOT)

    all_cv = []
    all_pred = []
    skipped = []

    pdf_path = os.path.join(RESULTS_ROOT, "scatter_all_drugs.pdf")
    pdf = PdfPages(pdf_path)

    for d in drugs:
        print("\n" + "=" * 80)
        print("Running drug:", d)

        d_dir = os.path.join(RESULTS_ROOT, d.replace("/", "_"))
        os.makedirs(d_dir, exist_ok=True)

        out = run_cv_for_drug(
            d,
            mirna_df,
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
        )
        if out is None:
            skipped.append(d)
            continue

        df_cv, pred_df, fold_feature_idx, feat_names, meta = out
        all_cv.append(df_cv)
        all_pred.append(pred_df)

        df_cv.to_csv(os.path.join(d_dir, "cv_metrics_folds.csv"), index=False)
        pred_df.to_csv(os.path.join(d_dir, "cv_predictions.csv"), index=False)
        meta.to_csv(os.path.join(d_dir, "matched_celllines_meta.csv"), index=False)

        # write selected miRNAs per fold (like your mRNA run)
        for i, idx in enumerate(fold_feature_idx, 1):
            feats = feat_names[idx].astype(str).tolist()
            with open(os.path.join(d_dir, f"selected_features_fold{i}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(feats))

        config = {
            "OMICS": "miRNA",
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
            "FOLDS_MODE": args.folds_mode,
            "FOLDS_ROOT": args.folds_root,
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

    if len(all_cv) == 0:
        raise RuntimeError("All drugs were skipped. Check MIN_CELL_LINES or mapping.")

    cv_all = pd.concat(all_cv, ignore_index=True)
    pred_all = pd.concat(all_pred, ignore_index=True)

    summary = (
        cv_all.groupby("drug")[["MSE","RMSE","R2","Spearman","Pearson","n"]]
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


if __name__ == "__main__":
    main()
