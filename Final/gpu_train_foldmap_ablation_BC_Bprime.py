# -*- coding: utf-8 -*-
"""
GPU-ready XGBoost CV pipeline (drug-wise) - 優化版
植入核心邏輯：
1) 樣本加權 (Sample Weights) 以拉開預測分佈
2) MAE 損失函數取代 MSE
3) 保留原始所有檔案路徑與輸出自動化邏輯
"""
import re
import os
import json
import shutil
import zipfile
from datetime import datetime
from typing import Optional, Tuple, Dict

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
# Metrics helpers (保持不變)
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
        c = pearsonr(y_true, y_pred).statistic
        return float(c) if np.isfinite(c) else float("nan")
    except Exception:
        try:
            c, _ = pearsonr(y_true, y_pred)
            return float(c) if np.isfinite(c) else float("nan")
        except Exception:
            return float("nan")

# ---------------------------
# Text / ID normalization
# 與 feature_qc.py / feature_explainability.py 統一：
#   - strip + upper
#   - 移除 CCLE 常見 tissue suffix（*_LUNG, *_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE, 及其他全大寫 suffix）
#   - 移除所有非英數字元（例如 NCI-H1648 -> NCIH1648）
# ---------------------------
def standardize_name(name: str) -> Optional[str]:
    if pd.isna(name):
        return None
    x = str(name).strip().upper()
    # drop common CCLE-style tissue suffixes so that
    #   DMS273_LUNG, DMS-273  ->  DMS273
    #   LP1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE -> LP1
    x = re.sub(r"_(LUNG|HAEMATOPOIETIC_AND_LYMPHOID_TISSUE|[A-Z]+)$", "", x)
    # remove non-alphanumeric characters (hyphen, spaces, etc.)
    x = re.sub(r"[^A-Z0-9]", "", x)
    return x if x else None

def standardize_drug(drug: str) -> Optional[str]:
    """Normalize drug names so case/whitespace variants are treated as the same drug.
    - strip
    - collapse whitespace
    - upper
    """
    if pd.isna(drug):
        return None
    x = str(drug).strip()
    x = re.sub(r"\s+", " ", x)
    x = x.upper()
    return x if x else None

def _drug_safe_name(drug: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(drug))


# ---------------------------
# Tissue marker helpers (for ablation experiments)
# ---------------------------
def _parse_tissue_from_ccle(ccle_name: str) -> str:
    """CCLE原始名稱通常像 CELL_TISSUE；取 '_' 後面的全部當 tissue_raw"""
    if pd.isna(ccle_name):
        return ""
    parts = str(ccle_name).strip().upper().split("_")
    if len(parts) <= 1:
        return ""
    return "_".join(parts[1:]).upper()

def _simplify_tissue(tissue_raw: str, lung_pat: re.Pattern, blood_pat: re.Pattern) -> str:
    t = (tissue_raw or "").upper()
    if lung_pat.search(t):
        return "LUNG"
    if blood_pat.search(t):
        return "BLOOD"
    return ""

def build_tissue_marker_set(expression_df: pd.DataFrame,
                            tissue_map_csv: str,
                            marker_d: float = 1.0,
                            lung_key: str = r"LUNG",
                            blood_key: str = r"HAEMATOPOIETIC|LYMPHOID|BLOOD",
                            cell_col: str = "標準化名稱",
                            ccle_col: str = "CCLE原始名稱") -> Tuple[set, pd.DataFrame]:
    """Compute tissue markers by Cohen's d on mRNA (LUNG vs BLOOD).
    Returns: (marker_gene_set, marker_table_df)
    NOTE: this uses ONLY X (expression) + tissue labels from map, never touches y(IC50).
    """
    tm = pd.read_csv(tissue_map_csv, low_memory=False)

    # try to auto-detect columns if the default names don't exist
    if cell_col not in tm.columns:
        for cand in ["標準化名稱", "CellLine", "cellline", "cell_line", "MODEL", "CellLineName", "Unnamed: 0"]:
            if cand in tm.columns:
                cell_col = cand
                break
    if ccle_col not in tm.columns:
        for cand in ["CCLE原始名稱", "CCLE", "ccle", "ccle_name", "CCLE_NAME"]:
            if cand in tm.columns:
                ccle_col = cand
                break
    if cell_col not in tm.columns or ccle_col not in tm.columns:
        raise ValueError(f"tissue map csv must contain columns for cell line and CCLE name (got: {tm.columns.tolist()[:20]})")

    tm["std"] = tm[cell_col].apply(standardize_name)
    tm["tissue_raw"] = tm[ccle_col].apply(_parse_tissue_from_ccle)

    lung_pat = re.compile(lung_key, re.IGNORECASE)
    blood_pat = re.compile(blood_key, re.IGNORECASE)
    tm["tissue"] = tm["tissue_raw"].apply(lambda t: _simplify_tissue(t, lung_pat, blood_pat))

    # keep only lung/blood and those present in expression_df
    tm = tm[tm["tissue"].isin(["LUNG", "BLOOD"])].dropna(subset=["std"]).copy()
    tm = tm[tm["std"].isin(expression_df.index)].copy()

    tm = tm.drop_duplicates("std")
    if tm["tissue"].nunique() < 2:
        raise ValueError("tissue map does not contain both LUNG and BLOOD after alignment to expression_df")

    X = expression_df.loc[tm["std"]].to_numpy(dtype=np.float64)
    tissue = tm["tissue"].to_numpy(dtype=str)

    maskA = tissue == "LUNG"
    maskB = tissue == "BLOOD"
    nA, nB = int(maskA.sum()), int(maskB.sum())
    if nA < 5 or nB < 5:
        raise ValueError(f"Too few samples for marker computation: nLUNG={nA}, nBLOOD={nB}")

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
        "gene": expression_df.columns,
        "mean_LUNG": meanA,
        "mean_BLOOD": meanB,
        "mean_diff(LUNG-BLOOD)": meanA - meanB,
        "cohen_d(LUNG-BLOOD)": d,
    })
    marker_tbl["marker_direction"] = np.where(marker_tbl["cohen_d(LUNG-BLOOD)"] > 0, "LUNG_marker", "BLOOD_marker")
    marker_tbl["is_marker"] = marker_tbl["cohen_d(LUNG-BLOOD)"].abs() >= float(marker_d)

    marker_set = set(marker_tbl.loc[marker_tbl["is_marker"], "gene"].astype(str).tolist())
    return marker_set, marker_tbl

# ---------------------------
# Fold-map reuse (保持不變)
# ---------------------------
def load_fold_map(folds_root: str, drug_name: str) -> dict:
    """Load fold_map.csv for a drug.
    Backward-compatible with older folders that may differ only by case/whitespace.
    """
    # try a few direct candidates first
    candidates = []
    for nm in [drug_name, standardize_drug(drug_name), str(drug_name).strip(), str(drug_name).strip().lower(), str(drug_name).strip().upper()]:
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
            return dict(zip(fm["cell_line"].astype(str), fm["fold"].astype(int)))

    # fallback: scan subfolders and match case-insensitively on the safe folder name
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
                    return dict(zip(fm["cell_line"].astype(str), fm["fold"].astype(int)))
    except FileNotFoundError:
        pass

    raise FileNotFoundError(
        f"fold_map.csv not found for drug={drug_name} (normalized={target}) under folds_root={folds_root}"
    )

def folds_from_foldmap(sub_agg: pd.DataFrame, fold_map: dict, n_splits: int = 5):
    sub_agg = sub_agg[sub_agg["std"].astype(str).isin(fold_map)].copy()
    fold_id = sub_agg["std"].astype(str).map(fold_map).to_numpy(dtype=int)
    K = int(np.max(fold_id))
    folds = []
    for f in range(1, K + 1):
        te = np.where(fold_id == f)[0]
        tr = np.where(fold_id != f)[0]
        folds.append((tr, te))
    return sub_agg, folds, K

# ---------------------------
# Calibration (保持不變)
# ---------------------------
def fit_linear_calibrator(yhat_tr: np.ndarray, y_tr: np.ndarray) -> Tuple[float, float]:
    yhat_tr = np.asarray(yhat_tr, dtype=np.float64).ravel()
    y_tr = np.asarray(y_tr, dtype=np.float64).ravel()
    var = np.var(yhat_tr)
    if not np.isfinite(var) or var < 1e-12:
        return float(np.mean(y_tr)), 0.0
    cov = float(np.mean((yhat_tr - yhat_tr.mean()) * (y_tr - y_tr.mean())))
    b = cov / var
    a = float(np.mean(y_tr) - b * np.mean(yhat_tr))
    return a, float(b)

def apply_linear_calibrator(yhat: np.ndarray, a: float, b: float) -> np.ndarray:
    return (a + b * np.asarray(yhat, dtype=np.float64).ravel()).astype(np.float64)

# ---------------------------
# Feature selection (保持不變)
# ---------------------------
def topn_by_corr(X_tr, y_tr, n, method="pearson", use_abs=True):
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
    scores = np.abs(corr) if use_abs else corr
    idx = np.argpartition(scores, -n)[-n:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return idx, scores[idx]

def make_folds(X, y, groups, n_splits=5, random_state=42):
    y_ser = pd.Series(y)
    try:
        y_bins = pd.qcut(y_ser, q=min(5, len(y_ser)), labels=False, duplicates="drop")
    except Exception:
        ranks = y_ser.rank(method="first")
        y_bins = pd.qcut(ranks, q=min(5, max(2, len(y_ser)//10)), labels=False, duplicates="drop")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(sgkf.split(X, np.asarray(y_bins, dtype=int), groups=groups))

# ---------------------------
# XGBoost params (已修改核心參數)
# ---------------------------
def build_xgb_params(use_gpu: bool) -> Dict:
    params = dict(
        n_estimators=1200,
        max_depth=8,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=0.1,         # 降低 L2，允許模型有較大輸出
        reg_alpha=0.1,
        min_child_weight=1,     # 允許細分極端樣本
        gamma=0.1,
        random_state=42,
        n_jobs=-1,
        objective='reg:absoluteerror' # 改用 MAE 以穩定極端值預測
    )
    if use_gpu:
        params.update(dict(tree_method="hist", device="cuda"))
    else:
        params.update(dict(tree_method="hist"))
    return params

def predict_xgb(model, X, use_gpu) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if use_gpu:
        return model.get_booster().predict(xgb.DMatrix(X))
    return model.predict(X)

def save_scatter(path_png, y_true, y_pred, title, subtitle="", stats=None):
    """Save y_true vs y_pred scatter with optional per-drug stats box."""
    plt.figure()
    plt.scatter(y_true, y_pred, s=12, alpha=0.6)
    # 45-degree reference line
    try:
        mn = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)]))
        mx = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)]))
        if np.isfinite(mn) and np.isfinite(mx):
            plt.plot([mn, mx], [mn, mx], 'r--')
    except Exception:
        pass
    plt.xlabel("y_true"); plt.ylabel("y_pred"); plt.title(title)
    if subtitle:
        plt.suptitle(subtitle, y=0.94, fontsize=10)
    if stats is not None:
        # stats keys: mean_ic50, r2, rmse, spearman, pearson
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
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75, edgecolor="gray")
        )
    plt.tight_layout()
    plt.savefig(path_png, dpi=160)
    plt.close()

# ---------------------------
# Inner-OOF calibration (已修改：加入樣本權重)
# ---------------------------
def inner_oof_predictions(X_tr, y_tr_s, y_tr_raw, groups_tr, xgb_params, use_gpu, n_splits_inner, y_scaler, scale_y):
    n = len(y_tr_raw)
    if n_splits_inner < 2 or n < (n_splits_inner * 2):
        return np.full(n, np.nan), False
    try:
        inner_folds = make_folds(X_tr, y_tr_raw, groups_tr, n_splits=n_splits_inner, random_state=123)
    except Exception: return np.full(n, np.nan), False

    oof = np.full(n, np.nan)
    y_mean = np.mean(y_tr_raw)

    for itr, ival in inner_folds:
        # 計算 Inner Fold 權重：強化極端值學習
        inner_weights = np.abs(y_tr_raw[itr] - y_mean) + 1.0
        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X_tr[itr], y_tr_s[itr], sample_weight=inner_weights)

        pred_val_s = predict_xgb(m, X_tr[ival], use_gpu=use_gpu)
        if scale_y and (y_scaler is not None):
            pred_val = y_scaler.inverse_transform(pred_val_s.reshape(-1, 1)).ravel()
        else: pred_val = pred_val_s
        oof[ival] = pred_val
    return oof, (np.isfinite(oof).sum() >= 10)

# ---------------------------
# Main per-drug CV (核心邏輯修改)
# ---------------------------
def run_cv_for_drug(drug_name, expression_df, drug_df, std_to_expr, out_dir, marker_set=None, ablation_mode='baseline', **kwargs):
    # 參數提取 (保留原始結構)
    top_n = kwargs.get('top_n', 1000)
    corr_method = kwargs.get('corr_method', 'pearson')
    use_abs_corr = kwargs.get('use_abs_corr', True)
    min_cell_lines = kwargs.get('min_cell_lines', 10)
    scale_x = kwargs.get('scale_x', True)
    scale_y = kwargs.get('scale_y', True)
    n_splits = kwargs.get('n_splits', 5)
    calibrate = kwargs.get('calibrate', True)
    use_gpu = kwargs.get('use_gpu', False)
    best_by = kwargs.get('best_by', 'RMSE')
    calib_inner_splits = kwargs.get('calib_inner_splits', 3)
    folds_mode = kwargs.get('folds_mode', 'create')
    folds_root = kwargs.get('folds_root', None)

    # Ablation settings (baseline / drop_nofill / drop_refill / only_markers)
    marker_set = marker_set or set()
    ablation_mode = str(ablation_mode or 'baseline').lower().strip()
    if ablation_mode not in ['baseline', 'drop_nofill', 'drop_refill', 'only_markers']:
        raise ValueError(f"Unknown ablation_mode={ablation_mode}. Use baseline/drop_nofill/drop_refill/only_markers")

    # 數據集構建 (維持與原程式碼路徑與邏輯一致)
    sub = drug_df.loc[drug_df["Drug"] == drug_name, ["CellLineName", "Y"]].dropna().copy()
    if len(sub) == 0: return None
    sub["std"] = sub["CellLineName"].apply(standardize_name)
    sub["expr_name"] = sub["std"].map(std_to_expr)
    sub = sub.dropna(subset=["expr_name"])
    sub_agg = sub.groupby(["std", "expr_name"], as_index=False).agg(Y=("Y", "median"), n_records=("Y", "size"))
    sub_agg = sub_agg[sub_agg["std"].isin(expression_df.index)].copy()
    if len(sub_agg) < min_cell_lines: return None

    X = expression_df.loc[sub_agg["std"]].to_numpy(dtype=np.float32)
    y = sub_agg["Y"].to_numpy(dtype=np.float32)
    groups = sub_agg["std"].to_numpy()
    gene_names = np.array(expression_df.columns)

    if folds_mode == "reuse":
        fold_map = load_fold_map(folds_root, drug_name)
        sub_agg, folds, _K = folds_from_foldmap(sub_agg, fold_map, n_splits=n_splits)
        # 依照過濾後的 sub_agg 重新對齊
        if len(sub_agg) < min_cell_lines:
            print(f"[SKIP] {drug_name}: only {len(sub_agg)} cell lines (<{min_cell_lines}) after fold_map filtering.")
            return None
        if len(folds) == 0:
            print(f"[SKIP] {drug_name}: no valid folds after fold_map filtering.")
            return None
        X = expression_df.loc[sub_agg["std"]].to_numpy(dtype=np.float32)
        y = sub_agg["Y"].to_numpy(dtype=np.float32)
        groups = sub_agg["std"].to_numpy()
    else:
        folds = make_folds(X, y, groups, n_splits=n_splits, random_state=42)
        if len(folds) == 0:
            print(f"[SKIP] {drug_name}: no valid folds generated.")
            return None

    # 檢查每個 fold 是否有足夠的訓練和測試樣本
    valid_folds = []
    for tr, te in folds:
        if len(tr) >= 2 and len(te) >= 1:  # 至少需要 2 個訓練樣本和 1 個測試樣本
            valid_folds.append((tr, te))
    if len(valid_folds) == 0:
        print(f"[SKIP] {drug_name}: no valid folds after filtering (need at least 2 train + 1 test samples per fold).")
        return None
    folds = valid_folds


    # ---------------------------
    # Feature selection (保持原本 topn_by_corr)，加入 ablation 模式：
    #  - baseline: 原本 top_n
    #  - drop_nofill (Exp B): 先選 top_n，再把 tissue marker 去掉（不補回 top_n）
    #  - only_markers (Exp C): 只在 marker 基因集合內做 corr 排名取 top_n（不足就取全部）
    # ---------------------------
    if ablation_mode == 'only_markers':
        if len(marker_set) == 0:
            print(f"[SKIP] {drug_name}: marker_set is empty, cannot run only_markers.")
            return None
        marker_mask = np.array([g in marker_set for g in gene_names], dtype=bool)
        if int(marker_mask.sum()) < 5:
            print(f"[SKIP] {drug_name}: too few marker genes present in expression ({int(marker_mask.sum())}).")
            return None
        X_for_fs = X[:, marker_mask]
        gene_names_for_fs = gene_names[marker_mask]
    else:
        X_for_fs = X
        gene_names_for_fs = gene_names

    fold_feature_idx = []
    for (tr, te) in folds:
        n_req = int(min(top_n, X_for_fs.shape[1]))
        idx_pool = topn_by_corr(X_for_fs[tr], y[tr], n_req, method=corr_method, use_abs=use_abs_corr)[0]

        if ablation_mode == 'drop_nofill':
            keep = np.array([gene_names_for_fs[j] not in marker_set for j in idx_pool], dtype=bool)
            idx = idx_pool[keep]
            if len(idx) == 0:
                # fallback: 找到至少 1 個非 marker 基因，避免後續完全無特徵
                big_n = int(min(max(top_n * 5, top_n + 2000), X_for_fs.shape[1]))
                idx_big = topn_by_corr(X_for_fs[tr], y[tr], big_n, method=corr_method, use_abs=use_abs_corr)[0]
                for j in idx_big:
                    if gene_names_for_fs[j] not in marker_set:
                        idx = np.array([j], dtype=int)
                        break
        elif ablation_mode == 'drop_refill':
            # Exp B': 從更大的 corr 排名池中，挑前 top_n 個「非 marker」基因（維持特徵數接近 top_n）
            big_n = int(min(max(top_n * 5, top_n + 5000), X_for_fs.shape[1]))
            idx_big = topn_by_corr(X_for_fs[tr], y[tr], big_n, method=corr_method, use_abs=use_abs_corr)[0]
            keep = np.array([gene_names_for_fs[j] not in marker_set for j in idx_big], dtype=bool)
            idx = idx_big[keep][:n_req]
            if len(idx) == 0:
                idx = idx_pool
        else:
            idx = idx_pool

        fold_feature_idx.append(idx)

    X_src = X_for_fs
    
    models_dir = os.path.join(out_dir, "models"); os.makedirs(models_dir, exist_ok=True)
    metric_rows, pred_rows = [], []
    xgb_params = build_xgb_params(use_gpu)
    gpu_failed_once = False

    for i, ((tr, te), idx) in enumerate(zip(folds, fold_feature_idx), 1):
        X_tr, X_te = X_src[tr][:, idx], X_src[te][:, idx]
        y_tr, y_te = y[tr], y[te]

        if len(idx) == 0:
            y_fin = np.full(len(te), float(np.mean(y_tr)), dtype=np.float64)
            metric_rows.append({
                "drug": drug_name, "fold": i, "n_features": 0,
                "RMSE": rmse(y_te, y_fin), "R2": r2_score(y_te, y_fin), "Spearman": safe_spearman(y_te, y_fin), "Pearson": safe_pearson(y_te, y_fin),
                "RMSE_raw": rmse(y_te, y_fin), "R2_raw": r2_score(y_te, y_fin), "cal_a": 0.0, "cal_b": 1.0, "n": len(te)
            })
            for k, j in enumerate(te):
                pred_rows.append({"drug": drug_name, "fold": i, "cell_line": str(groups[j]), "y_true": float(y_te[k]), "y_pred": float(y_fin[k]), "y_pred_raw": float(y_fin[k])})
            continue

        if scale_x:
            x_scaler = StandardScaler()
            X_tr = x_scaler.fit_transform(X_tr)
            X_te = x_scaler.transform(X_te)
        else: x_scaler = None

        y_scaler = StandardScaler().fit(y_tr.reshape(-1, 1)) if scale_y else None
        y_tr_s = y_scaler.transform(y_tr.reshape(-1, 1)).ravel() if scale_y else y_tr

        # 【核心修改】：計算樣本權重，強制模型去抓極端樣本
        y_tr_mean = np.mean(y_tr)
        weights = np.abs(y_tr - y_tr_mean) + 1.0

        model = xgb.XGBRegressor(**xgb_params)
        try:
            model.fit(X_tr, y_tr_s, sample_weight=weights) # 傳入權重
        except Exception as e:
            if use_gpu and not gpu_failed_once:
                gpu_failed_once = True; use_gpu = False
                xgb_params = build_xgb_params(False)
                model = xgb.XGBRegressor(**xgb_params)
                model.fit(X_tr, y_tr_s, sample_weight=weights)
            else: raise e

        pred_te_s = predict_xgb(model, X_te, use_gpu)
        pred_te = y_scaler.inverse_transform(pred_te_s.reshape(-1, 1)).ravel() if scale_y else pred_te_s

        cal_a, cal_b, fit_mode = 0.0, 1.0, "disabled"
        pred_te_cal = pred_te

        if calibrate:
            oof_pred, ok = inner_oof_predictions(X_tr, y_tr_s, y_tr, groups[tr], xgb_params, use_gpu, calib_inner_splits, y_scaler, scale_y)
            if ok:
                cal_a, cal_b = fit_linear_calibrator(oof_pred[np.isfinite(oof_pred)], y_tr[np.isfinite(oof_pred)])
                fit_mode = f"inner_oof_{calib_inner_splits}"
            else:
                p_tr_s = predict_xgb(model, X_tr, use_gpu)
                p_tr = y_scaler.inverse_transform(p_tr_s.reshape(-1, 1)).ravel() if scale_y else p_tr_s
                cal_a, cal_b = fit_linear_calibrator(p_tr, y_tr)
                fit_mode = "train_insample_fallback"
            pred_te_cal = apply_linear_calibrator(pred_te, cal_a, cal_b)

        # 儲存指標 (保留原始指標名稱以相容後續輸出)
        y_fin = pred_te_cal if calibrate else pred_te
        metric_rows.append({
            "drug": drug_name, "fold": i, "n_features": len(idx),
            "RMSE": rmse(y_te, y_fin), "R2": r2_score(y_te, y_fin), "Spearman": safe_spearman(y_te, y_fin), "Pearson": safe_pearson(y_te, y_fin),
            "RMSE_raw": rmse(y_te, pred_te), "R2_raw": r2_score(y_te, pred_te), "cal_a": cal_a, "cal_b": cal_b, "n": len(te)
        })
        
        for k, j in enumerate(te):
            pred_rows.append({"drug": drug_name, "fold": i, "cell_line": str(groups[j]), "y_true": float(y_te[k]), "y_pred": float(y_fin[k]), "y_pred_raw": float(pred_te[k])})

        # 儲存模型與前處理 (維持 joblib 結構)
        model.save_model(os.path.join(models_dir, f"xgb_model_fold{i}.json"))
        preproc = {"drug": drug_name, "fold": i, "selected_gene_idx": idx, "x_scaler": x_scaler, "y_scaler": y_scaler, "calibrator": {"a": cal_a, "b": cal_b}, "xgb_params": xgb_params}
        joblib.dump(preproc, os.path.join(models_dir, f"preproc_fold{i}.joblib"))

    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows), fold_feature_idx, gene_names, sub_agg

# ---------------------------
# Main 執行區 (保留原始檔案讀取路徑)
# ---------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds-mode", type=str, default="create")
    ap.add_argument("--folds-root", type=str, default=None)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--top-k-drugs", type=int, default=10)
    ap.add_argument("--top-n", type=int, default=1000)
    ap.add_argument("--ablation-mode", type=str, default="baseline", choices=["baseline","drop_nofill","drop_refill","only_markers"],
                   help="Ablation mode: baseline / drop_nofill (Exp B) / only_markers (Exp C)")
    ap.add_argument("--tissue-map-csv", type=str, default=None,
                   help="CellLine->CCLE map CSV to derive LUNG/BLOOD (e.g., lung_and_blood_Cline.csv)")
    ap.add_argument("--marker-d", type=float, default=1.0,
                   help="abs(Cohen's d) threshold for tissue markers (computed on mRNA LUNG vs BLOOD)")
    ap.add_argument("--lung-key", type=str, default="LUNG")
    ap.add_argument("--blood-key", type=str, default="HAEMATOPOIETIC|LYMPHOID|BLOOD")
    ap.add_argument("--use-gpu", action="store_true", default=False)
    ap.add_argument("--calibrate", action="store_true", default=True)
    ap.add_argument("--no-scale-y", action="store_true")
    ap.add_argument("--drug-list-csv", "--good-drugs-csv", dest="drug_list_csv", type=str, default=None,
                   help="CSV file containing a list of drugs to train. Uses column `drug`/`Drug` if present; otherwise the first column.")
    args = ap.parse_args()

    # 嚴格對齊原始路徑名稱
    expr_path = os.path.join(args.data_dir, "processed_expression_unnormalized_lung.csv")
    # if not os.path.exists(expr_path):
    #     expr_path = os.path.join(args.data_dir, "processed_expression.csv")

    drug_path = os.path.join(args.data_dir, "processed_drug_response_with_prism.csv")
    map_path  = os.path.join(args.data_dir, "expression_mapping_lung_and_lymphoid_only.csv")



    expression_df = pd.read_csv(expr_path, index_col=0)
    drug_df = pd.read_csv(drug_path, low_memory=False)
    # Tissue marker set for ablation (computed once)
    marker_set = set()
    marker_tbl = None
    if args.ablation_mode != "baseline":
        tissue_map_csv = args.tissue_map_csv or os.path.join(args.data_dir, "lung_and_blood_Cline.csv")
        if not os.path.exists(tissue_map_csv):
            raise FileNotFoundError(f"tissue-map-csv not found: {tissue_map_csv}")
        print(f"[INFO] Building tissue marker set from: {tissue_map_csv} | marker_d={args.marker_d}")
        marker_set, marker_tbl = build_tissue_marker_set(expression_df, tissue_map_csv, marker_d=args.marker_d,
                                                       lung_key=args.lung_key, blood_key=args.blood_key)
        print(f"[INFO] Tissue marker genes: {len(marker_set)} / {expression_df.shape[1]}")
    # Normalize drug names so case-variants (e.g., "navitoclax"/"Navitoclax") are treated as the same drug
    if "Drug" in drug_df.columns:
        drug_df["Drug_raw"] = drug_df["Drug"]
        drug_df["Drug"] = drug_df["Drug"].apply(standardize_drug)
    map_df = pd.read_csv(map_path)
    std_to_expr = dict(zip(map_df["standardized"], map_df["expression_name"]))
    # Decide which drugs to train
    if args.drug_list_csv:
        good_df = pd.read_csv(args.drug_list_csv, low_memory=False)
        # pick a sensible column
        cols_lower = {c.lower(): c for c in good_df.columns}
        if "drug" in cols_lower:
            col = cols_lower["drug"]
        else:
            col = good_df.columns[0]
        raw_list = good_df[col].dropna().astype(str).tolist()

        # normalize + de-duplicate while preserving order
        seen = set()
        desired = []
        for x in raw_list:
            nx = standardize_drug(x)
            if (nx is None) or (nx in seen):
                continue
            seen.add(nx)
            desired.append(nx)

        avail = set(drug_df["Drug"].dropna().astype(str).unique())
        top_drugs = [d for d in desired if d in avail]
        missing = [d for d in desired if d not in avail]
        print(f"[INFO] Using drug list from CSV: {args.drug_list_csv}")
        print(f"[INFO] Drugs in list: {len(desired)} | matched in drug_response: {len(top_drugs)} | missing: {len(missing)}")
        if missing:
            print("[WARN] Missing (show up to 20):", missing[:20])
        if len(top_drugs) == 0:
            raise ValueError("No drugs from --drug-list-csv matched processed_drug_response_with_prism.csv after normalization.")
    else:
        top_drugs = (drug_df.groupby("Drug")["CellLineName"].nunique()
                     .sort_values(ascending=False).head(args.top_k_drugs).index.tolist())


    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_ROOT = os.path.join(args.data_dir, "results", f"TRAIN_{run_id}")
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    if marker_tbl is not None:
        try:
            marker_tbl.to_csv(os.path.join(RESULTS_ROOT, "tissue_marker_table.csv"), index=False)
        except Exception:
            pass

    all_cv, all_pred = [], []
    pdf = PdfPages(os.path.join(RESULTS_ROOT, "all_scatters.pdf"))

    for d in top_drugs:
        print(f"Processing: {d} | ablation={args.ablation_mode}")
        d_dir = os.path.join(RESULTS_ROOT, _drug_safe_name(d)); os.makedirs(d_dir, exist_ok=True)
        out = run_cv_for_drug(d, expression_df, drug_df, std_to_expr, d_dir,
                             marker_set=marker_set, ablation_mode=args.ablation_mode,
                             top_n=args.top_n, use_gpu=args.use_gpu, calibrate=args.calibrate,
                             scale_y=(not args.no_scale_y), folds_mode=args.folds_mode, folds_root=args.folds_root)
        
        if out:
            df_cv, pred_df, _, _, meta = out
            all_cv.append(df_cv); all_pred.append(pred_df)
            df_cv.to_csv(os.path.join(d_dir, "cv_metrics_folds.csv"), index=False)
            pred_df.to_csv(os.path.join(d_dir, "cv_predictions.csv"), index=False)
            
            # 生成散佈圖
            
            # Per-drug overall metrics on out-of-fold predictions (all folds concatenated)
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
            stats = {"mean_ic50": mean_ic50, "r2": r2_all, "rmse": rmse_all, "spearman": sp_all, "pearson": pr_all}
            save_scatter(
                os.path.join(d_dir, "scatter_ytrue_vs_ypred.png"),
                y_true_all, y_pred_all,
                title=f"{d} (Calibrated)" if args.calibrate else str(d),
                stats=stats
            )
            img = plt.imread(os.path.join(d_dir, "scatter_ytrue_vs_ypred.png"))
            plt.figure(); plt.imshow(img); plt.axis("off"); plt.title(d); pdf.savefig(); plt.close()

    pdf.close()
    
    if all_cv:
        pd.concat(all_cv).to_csv(os.path.join(RESULTS_ROOT, "all_cv_metrics.csv"), index=False)
        if all_pred:
            pd.concat(all_pred).to_csv(os.path.join(RESULTS_ROOT, "all_cv_predictions.csv"), index=False)
        print(f"Done. Results saved to: {RESULTS_ROOT}")
        print(f"Successfully processed {len(all_cv)} drugs.")
    else:
        print(f"Warning: No valid results generated for any drug.")
        print(f"Possible reasons:")
        print(f"  - Cell line filtering (lung/lymphoid only) removed too many samples")
        print(f"  - Fold map filtering (if using --folds-mode reuse) removed too many samples")
        print(f"  - Minimum cell line requirement (default: 10) not met")
        print(f"Results directory created at: {RESULTS_ROOT}")

if __name__ == "__main__":
    main()