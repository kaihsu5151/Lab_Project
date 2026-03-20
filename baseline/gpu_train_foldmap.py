# -*- coding: utf-8 -*-
"""
GPU-ready XGBoost CV pipeline (drug-wise) - integrated mRNA version
保留這支 mRNA 主腳本原本的兩個核心設計：
1) sample weights：加強極端樣本學習
2) objective='reg:absoluteerror'：使用 MAE 損失

並補齊與其他 omics 腳本一致的功能：
- 同一個 drug x cell line 的 Y 用 median 聚合
- 可重用其他組學已切好的 fold_map.csv
- 每個藥物額外輸出自己的 fold_map.csv
- 散點圖同時輸出 calibrated / raw 版本
- 散點圖標註 mean IC50 / R2 / RMSE / Spearman / Pearson
"""

import re
import os
import json
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
# ---------------------------
def standardize_name(name: str) -> Optional[str]:
    if pd.isna(name):
        return None
    x = str(name).strip().upper()
    x = re.sub(r"_(LUNG|HAEMATOPOIETIC_AND_LYMPHOID_TISSUE|[A-Z]+)$", "", x)
    x = re.sub(r"[^A-Z0-9]", "", x)
    return x if x else None


def standardize_drug(drug: str) -> Optional[str]:
    if pd.isna(drug):
        return None
    x = str(drug).strip()
    x = re.sub(r"\s+", " ", x)
    x = x.upper()
    return x if x else None


def _drug_safe_name(drug: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(drug))


# ---------------------------
# Fold-map reuse
# ---------------------------
def load_fold_map(folds_root: str, drug_name: str) -> dict:
    """Load fold_map.csv for a drug."""
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


def folds_from_foldmap(sub_agg: pd.DataFrame, fold_map: dict):
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


# ---------------------------
# Calibration
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
# Feature selection
# ---------------------------
def topn_by_corr(X_tr, y_tr, n, method="pearson", use_abs=True):
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


def make_folds(X, y, groups, n_splits=5, random_state=42):
    y_ser = pd.Series(y)
    try:
        y_bins = pd.qcut(y_ser, q=min(5, len(y_ser)), labels=False, duplicates="drop")
        if pd.Series(y_bins).nunique() < 2:
            raise ValueError("qcut produced <2 bins")
    except Exception:
        ranks = y_ser.rank(method="first")
        q = min(5, max(2, len(y_ser) // 10))
        y_bins = pd.qcut(ranks, q=q, labels=False, duplicates="drop")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(sgkf.split(X, np.asarray(y_bins, dtype=int), groups=groups))


# ---------------------------
# XGBoost params
# ---------------------------
def build_xgb_params(use_gpu: bool) -> Dict:
    params = dict(
        n_estimators=1200,
        max_depth=8,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=0.1,
        reg_alpha=0.1,
        min_child_weight=1,
        gamma=0.1,
        random_state=42,
        n_jobs=-1,
        objective="reg:absoluteerror",
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


# ---------------------------
# Plot
# ---------------------------
def save_scatter(path_png, y_true, y_pred, title, subtitle="", stats=None):
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
            0.02,
            0.98,
            "\n".join(text_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75, edgecolor="gray"),
        )

    plt.tight_layout()
    plt.savefig(path_png, dpi=160)
    plt.close()


# ---------------------------
# Inner-OOF calibration
# 保留 sample weights
# ---------------------------
def inner_oof_predictions(X_tr, y_tr_s, y_tr_raw, groups_tr, xgb_params, use_gpu, n_splits_inner, y_scaler, scale_y):
    n = len(y_tr_raw)
    if n_splits_inner < 2 or n < (n_splits_inner * 2):
        return np.full(n, np.nan), False

    try:
        inner_folds = make_folds(X_tr, y_tr_raw, groups_tr, n_splits=n_splits_inner, random_state=123)
    except Exception:
        return np.full(n, np.nan), False

    oof = np.full(n, np.nan)
    y_mean = np.mean(y_tr_raw)

    for itr, ival in inner_folds:
        inner_weights = np.abs(y_tr_raw[itr] - y_mean) + 1.0
        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X_tr[itr], y_tr_s[itr], sample_weight=inner_weights)

        pred_val_s = predict_xgb(m, X_tr[ival], use_gpu=use_gpu)
        if scale_y and (y_scaler is not None):
            pred_val = y_scaler.inverse_transform(pred_val_s.reshape(-1, 1)).ravel()
        else:
            pred_val = pred_val_s
        oof[ival] = pred_val

    return oof, (np.isfinite(oof).sum() >= 10)


# ---------------------------
# Data helpers
# ---------------------------
def load_good_drugs_from_csv(csv_path: str) -> List[str]:
    good_df = pd.read_csv(csv_path, low_memory=False)
    cols_lower = {c.lower(): c for c in good_df.columns}
    col = cols_lower["drug"] if "drug" in cols_lower else good_df.columns[0]
    raw_list = good_df[col].dropna().astype(str).tolist()

    seen = set()
    desired = []
    for x in raw_list:
        nx = standardize_drug(x)
        if (nx is None) or (nx in seen):
            continue
        seen.add(nx)
        desired.append(nx)
    return desired


def compute_overall_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 2:
        return {
            "mean_ic50": float("nan"),
            "r2": float("nan"),
            "rmse": float("nan"),
            "spearman": float("nan"),
            "pearson": float("nan"),
        }
    yt = y_true[mask]
    yp = y_pred[mask]
    return {
        "mean_ic50": float(np.nanmean(yt)),
        "r2": float(r2_score(yt, yp)),
        "rmse": rmse(yt, yp),
        "spearman": safe_spearman(yt, yp),
        "pearson": safe_pearson(yt, yp),
    }


# ---------------------------
# Main per-drug CV
# ---------------------------
def run_cv_for_drug(drug_name, expression_df, drug_df, std_to_expr, out_dir, **kwargs):
    top_n = kwargs.get("top_n", 1000)
    corr_method = kwargs.get("corr_method", "pearson")
    use_abs_corr = kwargs.get("use_abs_corr", True)
    min_cell_lines = kwargs.get("min_cell_lines", 10)
    scale_x = kwargs.get("scale_x", True)
    scale_y = kwargs.get("scale_y", True)
    n_splits = kwargs.get("n_splits", 5)
    calibrate = kwargs.get("calibrate", True)
    use_gpu = kwargs.get("use_gpu", False)
    calib_inner_splits = kwargs.get("calib_inner_splits", 3)
    folds_mode = kwargs.get("folds_mode", "create")
    folds_root = kwargs.get("folds_root", None)

    sub = drug_df.loc[drug_df["Drug"] == drug_name, ["CellLineName", "Y"]].dropna().copy()
    if len(sub) == 0:
        return None

    sub["std"] = sub["CellLineName"].apply(standardize_name)
    sub["expr_name"] = sub["std"].map(std_to_expr)
    sub = sub.dropna(subset=["std", "expr_name"])

    # 改成 median
    sub_agg = (
        sub.groupby(["std", "expr_name"], as_index=False)
        .agg(Y=("Y", "median"), n_records=("Y", "size"))
    )
    sub_agg = sub_agg[sub_agg["std"].isin(expression_df.index)].copy()
    if len(sub_agg) < min_cell_lines:
        print(f"[SKIP] {drug_name}: only {len(sub_agg)} cell lines (<{min_cell_lines}) after matching expression.")
        return None

    X = expression_df.loc[sub_agg["std"]].to_numpy(dtype=np.float32)
    y = sub_agg["Y"].to_numpy(dtype=np.float32)
    groups = sub_agg["std"].to_numpy()
    gene_names = np.array(expression_df.columns, dtype=object)

    # folds
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

        X = expression_df.loc[sub_agg["std"]].to_numpy(dtype=np.float32)
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

    # save folds.npz
    folds_npz = os.path.join(out_dir, "folds.npz")
    save_dict = {"n_splits": len(folds)}
    for i, (tr, te) in enumerate(folds):
        save_dict[f"tr_{i}"] = np.asarray(tr, dtype=np.int32)
        save_dict[f"te_{i}"] = np.asarray(te, dtype=np.int32)
    np.savez(folds_npz, **save_dict)

    # save fold_map.csv
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
        idx, sc = topn_by_corr(X[tr], y[tr], top_n, method=corr_method, use_abs=use_abs_corr)
        fold_feature_idx.append(idx)
        if top_n > 0:
            print(f"[{drug_name}] Fold {i}: selected {len(idx)} genes, top1={gene_names[idx[0]]}, score={sc[0]:.4f}")
        else:
            print(f"[{drug_name}] Fold {i}: NO feature selection, using all {len(idx)} genes")

    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    metric_rows = []
    pred_rows = []
    xgb_params = build_xgb_params(use_gpu)
    gpu_failed_once = False

    for i, ((tr, te), idx) in enumerate(zip(folds, fold_feature_idx), 1):
        X_tr = X[tr][:, idx]
        X_te = X[te][:, idx]
        y_tr = y[tr]
        y_te = y[te]

        if scale_x:
            x_scaler = StandardScaler()
            X_tr = x_scaler.fit_transform(X_tr).astype(np.float32, copy=False)
            X_te = x_scaler.transform(X_te).astype(np.float32, copy=False)
        else:
            x_scaler = None
            X_tr = X_tr.astype(np.float32, copy=False)
            X_te = X_te.astype(np.float32, copy=False)

        y_scaler = StandardScaler().fit(y_tr.reshape(-1, 1)) if scale_y else None
        y_tr_s = y_scaler.transform(y_tr.reshape(-1, 1)).ravel().astype(np.float32, copy=False) if scale_y else y_tr

        # 保留這支 mRNA 版原本的 sample weights
        y_tr_mean = float(np.mean(y_tr))
        weights = np.abs(y_tr - y_tr_mean) + 1.0

        model = xgb.XGBRegressor(**xgb_params)
        try:
            model.fit(X_tr, y_tr_s, sample_weight=weights)
        except Exception as e:
            if use_gpu and not gpu_failed_once:
                print("[WARN] GPU training failed, falling back to CPU. Error:", str(e)[:200])
                gpu_failed_once = True
                use_gpu = False
                xgb_params = build_xgb_params(False)
                model = xgb.XGBRegressor(**xgb_params)
                model.fit(X_tr, y_tr_s, sample_weight=weights)
            else:
                raise e

        pred_te_s = predict_xgb(model, X_te, use_gpu)
        pred_te = y_scaler.inverse_transform(pred_te_s.reshape(-1, 1)).ravel() if scale_y else pred_te_s

        cal_a, cal_b, fit_mode = 0.0, 1.0, "disabled"
        pred_te_cal = pred_te

        if calibrate:
            oof_pred, ok = inner_oof_predictions(
                X_tr, y_tr_s, y_tr, groups[tr], xgb_params, use_gpu, calib_inner_splits, y_scaler, scale_y
            )
            if ok:
                mask = np.isfinite(oof_pred)
                cal_a, cal_b = fit_linear_calibrator(oof_pred[mask], y_tr[mask])
                fit_mode = f"inner_oof_{calib_inner_splits}"
            else:
                p_tr_s = predict_xgb(model, X_tr, use_gpu)
                p_tr = y_scaler.inverse_transform(p_tr_s.reshape(-1, 1)).ravel() if scale_y else p_tr_s
                cal_a, cal_b = fit_linear_calibrator(p_tr, y_tr)
                fit_mode = "train_insample_fallback"
            pred_te_cal = apply_linear_calibrator(pred_te, cal_a, cal_b)

        y_fin = pred_te_cal if calibrate else pred_te

        metric_rows.append({
            "drug": drug_name,
            "fold": i,
            "n_features": int(len(idx)),
            "MSE": mse(y_te, y_fin),
            "RMSE": rmse(y_te, y_fin),
            "R2": float(r2_score(y_te, y_fin)),
            "Spearman": safe_spearman(y_te, y_fin),
            "Pearson": safe_pearson(y_te, y_fin),
            "MSE_raw": mse(y_te, pred_te),
            "RMSE_raw": rmse(y_te, pred_te),
            "R2_raw": float(r2_score(y_te, pred_te)),
            "Spearman_raw": safe_spearman(y_te, pred_te),
            "Pearson_raw": safe_pearson(y_te, pred_te),
            "cal_a": float(cal_a),
            "cal_b": float(cal_b),
            "cal_fit_mode": fit_mode,
            "n": int(len(te)),
        })

        for k, j in enumerate(te):
            pred_rows.append({
                "drug": drug_name,
                "fold": i,
                "cell_line": str(groups[j]),
                "y_true": float(y_te[k]),
                "y_pred": float(y_fin[k]),
                "y_pred_raw": float(pred_te[k]),
            })

        model.save_model(os.path.join(models_dir, f"xgb_model_fold{i}.json"))
        preproc = {
            "omics": "mRNA",
            "drug": drug_name,
            "fold": i,
            "selected_gene_idx": np.asarray(idx, dtype=np.int32),
            "selected_genes": [str(g) for g in gene_names[idx].tolist()],
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "calibrator": {
                "enabled": bool(calibrate),
                "a": float(cal_a),
                "b": float(cal_b),
                "fit_mode": fit_mode,
            },
            "sample_weight_mode": "abs(y - mean(y)) + 1.0",
            "xgb_params": xgb_params,
            "folds_file": os.path.basename(folds_npz),
            "fold_map_file": os.path.basename(fold_map_csv),
        }
        joblib.dump(preproc, os.path.join(models_dir, f"preproc_fold{i}.joblib"))

    df_cv = pd.DataFrame(metric_rows)
    pred_df = pd.DataFrame(pred_rows)
    return df_cv, pred_df, fold_feature_idx, gene_names, sub_agg


# ---------------------------
# Main
# ---------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--folds-mode", type=str, default="create", choices=["create", "reuse"])
    ap.add_argument("--folds-root", type=str, default=None)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--top-k-drugs", type=int, default=10)
    ap.add_argument("--top-n", type=int, default=1000)
    ap.add_argument("--use-gpu", action="store_true", default=False)
    ap.add_argument("--calibrate", action="store_true", default=True)
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--no-scale-y", action="store_true")
    ap.add_argument("--no-scale-x", action="store_true")
    ap.add_argument("--corr-method", type=str, default="pearson", choices=["pearson", "spearman"])
    ap.add_argument("--min-cell-lines", type=int, default=10)
    ap.add_argument(
        "--drug-list-csv",
        "--good-drugs-csv",
        dest="drug_list_csv",
        type=str,
        default=None,
        help="CSV file containing a list of drugs to train. Uses column `drug`/`Drug` if present; otherwise the first column.",
    )
    args = ap.parse_args()

    expr_path = os.path.join(args.data_dir, "processed_expression_unnormalized_lung.csv")
    drug_path = os.path.join(args.data_dir, "processed_drug_response_with_prism.csv")
    map_path = os.path.join(args.data_dir, "expression_mapping_lung_and_lymphoid_only.csv")

    expression_df = pd.read_csv(expr_path, index_col=0)
    expression_df.index = expression_df.index.map(standardize_name)
    expression_df = expression_df[~pd.isna(expression_df.index)].copy()
    expression_df = expression_df[~pd.Index(expression_df.index).duplicated(keep="first")].copy()

    drug_df = pd.read_csv(drug_path, low_memory=False)
    if "Drug" in drug_df.columns:
        drug_df["Drug_raw"] = drug_df["Drug"]
        drug_df["Drug"] = drug_df["Drug"].apply(standardize_drug)
    if "Y" in drug_df.columns:
        drug_df["Y"] = pd.to_numeric(drug_df["Y"], errors="coerce")
        drug_df = drug_df.dropna(subset=["Y"])

    map_df = pd.read_csv(map_path)
    std_col = "standardized" if "standardized" in map_df.columns else map_df.columns[0]
    expr_col = "expression_name" if "expression_name" in map_df.columns else map_df.columns[1]
    map_df[std_col] = map_df[std_col].apply(standardize_name)
    std_to_expr = dict(zip(map_df[std_col], map_df[expr_col]))

    if args.drug_list_csv:
        desired = load_good_drugs_from_csv(args.drug_list_csv)
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
        top_drugs = (
            drug_df.groupby("Drug")["CellLineName"]
            .nunique()
            .sort_values(ascending=False)
            .head(args.top_k_drugs)
            .index.tolist()
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_ROOT = os.path.join(args.data_dir, "results", f"TRAIN_{run_id}")
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    calibrate = bool(args.calibrate and not args.no_calibrate)
    scale_y = not args.no_scale_y
    scale_x = not args.no_scale_x

    all_cv = []
    all_pred = []
    pdf = PdfPages(os.path.join(RESULTS_ROOT, "all_scatters.pdf"))

    for d in top_drugs:
        print(f"Processing: {d}")
        d_dir = os.path.join(RESULTS_ROOT, _drug_safe_name(d))
        os.makedirs(d_dir, exist_ok=True)

        out = run_cv_for_drug(
            d,
            expression_df,
            drug_df,
            std_to_expr,
            d_dir,
            top_n=args.top_n,
            corr_method=args.corr_method,
            use_abs_corr=True,
            min_cell_lines=args.min_cell_lines,
            scale_x=scale_x,
            scale_y=scale_y,
            n_splits=5,
            calibrate=calibrate,
            use_gpu=args.use_gpu,
            calib_inner_splits=3,
            folds_mode=args.folds_mode,
            folds_root=args.folds_root,
        )

        if out is None:
            continue

        df_cv, pred_df, fold_feature_idx, gene_names, meta = out
        all_cv.append(df_cv)
        all_pred.append(pred_df)

        df_cv.to_csv(os.path.join(d_dir, "cv_metrics_folds.csv"), index=False)
        pred_df.to_csv(os.path.join(d_dir, "cv_predictions.csv"), index=False)
        meta.to_csv(os.path.join(d_dir, "matched_celllines_meta.csv"), index=False)

        for i, idx in enumerate(fold_feature_idx, 1):
            genes = gene_names[idx].astype(str).tolist()
            with open(os.path.join(d_dir, f"selected_genes_fold{i}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(genes))

        y_true_all = pred_df["y_true"].to_numpy(dtype=np.float64)
        y_pred_all = pred_df["y_pred"].to_numpy(dtype=np.float64)
        y_pred_raw_all = pred_df["y_pred_raw"].to_numpy(dtype=np.float64)

        stats = compute_overall_stats(y_true_all, y_pred_all)
        subtitle = (
            f"Final: RMSE_mean={df_cv['RMSE'].mean():.4f}, R2_mean={df_cv['R2'].mean():.4f}, "
            f"Spearman_mean={df_cv['Spearman'].mean():.4f}, Pearson_mean={df_cv['Pearson'].mean():.4f}"
        )
        scatter_path = os.path.join(d_dir, "scatter_ytrue_vs_ypred.png")
        save_scatter(
            scatter_path,
            y_true_all,
            y_pred_all,
            title=f"{d} (mRNA OOF across folds)",
            subtitle=subtitle,
            stats=stats,
        )

        stats_raw = compute_overall_stats(y_true_all, y_pred_raw_all)
        subtitle_raw = (
            f"Raw: RMSE_mean={df_cv['RMSE_raw'].mean():.4f}, R2_mean={df_cv['R2_raw'].mean():.4f}, "
            f"Spearman_mean={df_cv['Spearman_raw'].mean():.4f}, Pearson_mean={df_cv['Pearson_raw'].mean():.4f}"
        )
        save_scatter(
            os.path.join(d_dir, "scatter_ytrue_vs_ypred_raw.png"),
            y_true_all,
            y_pred_raw_all,
            title=f"{d} (mRNA raw OOF across folds)",
            subtitle=subtitle_raw,
            stats=stats_raw,
        )

        img = plt.imread(scatter_path)
        plt.figure()
        plt.imshow(img)
        plt.axis("off")
        plt.title(d)
        pdf.savefig()
        plt.close()

    pdf.close()

    if all_cv:
        pd.concat(all_cv, ignore_index=True).to_csv(os.path.join(RESULTS_ROOT, "all_cv_metrics.csv"), index=False)
        if all_pred:
            pd.concat(all_pred, ignore_index=True).to_csv(os.path.join(RESULTS_ROOT, "all_cv_predictions.csv"), index=False)
        print(f"Done. Results saved to: {RESULTS_ROOT}")
        print(f"Successfully processed {len(all_cv)} drugs.")
    else:
        print("Warning: No valid results generated for any drug.")
        print("Possible reasons:")
        print("  - Cell line filtering (lung/lymphoid only) removed too many samples")
        print("  - Fold map filtering (if using --folds-mode reuse) removed too many samples")
        print("  - Minimum cell line requirement not met")
        print(f"Results directory created at: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
