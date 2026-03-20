# -*- coding: utf-8 -*-
"""Fold-wise blending / stacking for multi-omics OOF predictions (supports 2~4 omics).

Input per omics: a CSV with columns:
  drug, fold, cell_line, y_true, y_pred

Typical files produced by your training scripts:
- mRNA (gpu_train_foldmap*.py):  results/.../all_topK_predictions.csv
- CNV  (cnv_train_foldmap.py):   results_cnv/.../all_predictions.csv
- Methylation:                   results_methylation/.../all_predictions.csv
- miRNA:                         results_mirna/.../all_predictions.csv

This script (no leakage):
1) Merge OOF predictions by (drug, fold, cell_line)
   - IMPORTANT: We do NOT use y_true as a merge key.
2) After merge, we check that y_true is consistent across omics.
3) Baselines:
   - simple mean (unweighted)
   - fixed-weight average (optional)
4) Proper evaluation (cross-fitting, per drug):
   For each drug and each fold k:
     learn weights / meta-model on folds != k, predict on fold == k

Supported ensemble methods (--blend):
- stacking     : learn weights with a meta-model (ridge / linear / linear_pos)
- spearman     : weights ∝ max(Spearman(y_true, pred_i), 0)
- pearson      : weights ∝ max(Pearson(y_true, pred_i), 0)
- inverse_rmse : weights ∝ 1 / (RMSE_i + eps)
- top1         : pick the single best omics on train folds (by RMSE)
- top2         : average the best two omics on train folds (by RMSE)
- mean         : simple mean (same as baseline)
- median       : median across omics (robust to outliers)

Outputs (to --out-dir):
- ensemble_predictions.csv
- ensemble_metrics_folds.csv
- ensemble_summary.csv
  * per-drug OOF metrics computed by concatenating all 5 test folds first
  * plus fold mean/std metrics for reference
- learned_weights.csv
- final_weights_per_drug.csv
- coverage_report.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr


# -------------------------
# Metrics helpers
# -------------------------

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
        # SciPy >= 1.10 returns an object with .statistic
        c = pearsonr(y_true, y_pred).statistic
        return float(c) if np.isfinite(c) else float("nan")
    except Exception:
        try:
            c, _ = pearsonr(y_true, y_pred)
            return float(c) if np.isfinite(c) else float("nan")
        except Exception:
            return float("nan")


# -------------------------
# IO + preprocessing
# -------------------------

def read_pred(path: str, name: str) -> pd.DataFrame:
    """Read one omics prediction CSV and standardize column types."""
    df = pd.read_csv(path)
    need = {"drug", "fold", "cell_line", "y_true", "y_pred"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{name} pred missing columns: {missing} (file={path})")

    df = df[list(need)].copy()

    # Normalize dtypes to avoid merge surprises
    df["drug"] = df["drug"].astype(str)
    df["cell_line"] = df["cell_line"].astype(str)
    df["fold"] = df["fold"].astype(int)

    # Rename so we can merge multiple omics without column conflicts
    df = df.rename(
        columns={
            "y_pred": f"pred_{name}",
            "y_true": f"y_true_{name}",
        }
    )
    return df


def check_and_make_y_true(df: pd.DataFrame, names: list[str], tol: float = 1e-8) -> pd.DataFrame:
    """Create a single y_true column and verify y_true is consistent across omics."""
    y_cols = [f"y_true_{n}" for n in names]

    y0 = df[y_cols[0]].to_numpy(dtype=float)

    for c in y_cols[1:]:
        y1 = df[c].to_numpy(dtype=float)
        m = np.isfinite(y0) & np.isfinite(y1)
        if not np.any(m):
            continue
        max_diff = float(np.max(np.abs(y0[m] - y1[m])))
        if max_diff > tol:
            raise RuntimeError(
                "y_true mismatch detected across omics after merge. "
                f"Reference={y_cols[0]} vs {c}, max_abs_diff={max_diff}. "
                "This usually means the source files are not aligned, or y_true was computed/rounded differently."
            )

    df = df.copy()
    df["y_true"] = df[y_cols[0]].astype(float)
    df = df.drop(columns=y_cols)
    return df


def parse_fixed_weights(s: str, names: list[str]) -> dict:
    """Parse fixed weights.

    Examples:
      "mrna=0.4,cnv=0.2,meth=0.2,mirna=0.2"
      "0.4,0.2,0.2,0.2"  (aligned to names order)

    Returns: dict name -> weight
    """
    s = (s or "").strip()
    if not s:
        return {}

    if "=" in s:
        w = {}
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            k, v = part.split("=", 1)
            w[k.strip().lower()] = float(v.strip())
        return {k: w[k] for k in names if k in w}

    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) != len(names):
        raise ValueError(f"--fixed-weights has {len(vals)} numbers, but you provided {len(names)} omics: {names}")
    return dict(zip(names, vals))


def weighted_average_rowwise(df: pd.DataFrame, pred_cols: list[str], w: dict) -> np.ndarray:
    """Row-wise weighted average with missing predictions allowed.

    - Missing weight defaults to 0
    - For each row, renormalize weights over available (non-NaN) predictions
    """
    col_names = [c.replace("pred_", "") for c in pred_cols]
    W = np.array([float(w.get(n, 0.0)) for n in col_names], dtype=float)

    if (not w) or (np.nansum(W) <= 0):
        W = np.ones(len(pred_cols), dtype=float)

    P = df[pred_cols].to_numpy(dtype=float)
    out = np.full(len(df), np.nan, dtype=float)

    for i in range(len(df)):
        pi = P[i]
        m = np.isfinite(pi)
        if not np.any(m):
            continue

        wi = np.maximum(W[m], 0.0)
        if wi.sum() <= 0:
            wi = np.ones_like(wi)
        wi = wi / wi.sum()
        out[i] = float(np.dot(wi, pi[m]))

    return out


def drop_bad_rows(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop rows with NaN/inf in X or y."""
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[m], y[m], m


# -------------------------
# Ensemble weight learners
# -------------------------

def build_meta_model(meta_kind: str, alpha: float):
    if meta_kind == "ridge":
        return Ridge(alpha=float(alpha), fit_intercept=True)
    if meta_kind == "linear_pos":
        return LinearRegression(positive=True)
    return LinearRegression()


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.array(w, dtype=float)
    w[~np.isfinite(w)] = 0.0
    w = np.maximum(w, 0.0)
    s = float(w.sum())
    if s <= 0:
        w = np.ones_like(w)
        s = float(w.sum())
    return w / s


def learn_weights_spearman(tr: pd.DataFrame, pred_cols: list[str]) -> np.ndarray:
    """weights ∝ max(Spearman(y, pred_i), 0)"""
    y = tr["y_true"].to_numpy(dtype=float)
    w = []
    for c in pred_cols:
        p = tr[c].to_numpy(dtype=float)
        rho = safe_spearman(y, p)
        if not np.isfinite(rho):
            rho = 0.0
        w.append(max(rho, 0.0))
    return _normalize_weights(np.array(w, dtype=float))


def learn_weights_pearson(tr: pd.DataFrame, pred_cols: list[str]) -> np.ndarray:
    """weights ∝ max(Pearson(y, pred_i), 0)

    Notes (plain language):
    - Pearson measures *linear* correlation and is more sensitive to outliers.
    - We clip negatives to 0 so a negatively correlated omics won't "pull" predictions the wrong way.
    """
    y = tr["y_true"].to_numpy(dtype=float)
    w = []
    for c in pred_cols:
        p = tr[c].to_numpy(dtype=float)
        r = safe_pearson(y, p)
        if not np.isfinite(r):
            r = 0.0
        w.append(max(r, 0.0))
    return _normalize_weights(np.array(w, dtype=float))


def learn_weights_inverse_rmse(tr: pd.DataFrame, pred_cols: list[str], eps: float = 1e-8) -> np.ndarray:
    """weights ∝ 1/(RMSE_i + eps)"""
    y = tr["y_true"].to_numpy(dtype=float)
    w = []
    for c in pred_cols:
        p = tr[c].to_numpy(dtype=float)
        r = rmse(y, p)
        w.append(1.0 / (float(r) + float(eps)))
    return _normalize_weights(np.array(w, dtype=float))


def learn_weights_topk(tr: pd.DataFrame, pred_cols: list[str], k: int = 1, eps: float = 1e-8) -> np.ndarray:
    """Pick best k omics by RMSE on train folds. Weights are equal among selected."""
    y = tr["y_true"].to_numpy(dtype=float)
    rmses = []
    for c in pred_cols:
        p = tr[c].to_numpy(dtype=float)
        rmses.append(rmse(y, p))

    idx = np.argsort(np.array(rmses, dtype=float))[: max(1, int(k))]
    w = np.zeros(len(pred_cols), dtype=float)
    w[idx] = 1.0
    return _normalize_weights(w)


def predict_with_constant_weights(te: pd.DataFrame, pred_cols: list[str], w: np.ndarray) -> np.ndarray:
    P = te[pred_cols].to_numpy(dtype=float)
    # Handle rare NaNs (even though inner-merge usually avoids this)
    out = np.full(len(te), np.nan, dtype=float)
    for i in range(len(te)):
        pi = P[i]
        m = np.isfinite(pi)
        if not np.any(m):
            continue
        wi = _normalize_weights(w[m])
        out[i] = float(np.dot(wi, pi[m]))
    return out


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--out-dir",
        default="results_ensemble",
        help="Base output directory (default: results_ensemble). A timestamped subfolder will be created.",
    )
    ap.add_argument(
        "--no-timestamp",
        action="store_true",
        help="If set, use --out-dir directly without creating a timestamped subfolder",
    )

    ap.add_argument("--mrna-pred", required=True)
    ap.add_argument("--cnv-pred", required=True)
    ap.add_argument("--meth-pred", required=True)
    ap.add_argument("--mirna-pred", default=None, help="Optional: miRNA predictions CSV")

    ap.add_argument(
        "--blend",
        type=str,
        default="stacking",
        choices=["stacking", "spearman", "pearson", "inverse_rmse", "top1", "top2", "mean", "median"],
        help="How to combine omics predictions.",
    )

    # Stacking meta-model params (only used when --blend=stacking)
    ap.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha (only for --meta ridge)")
    ap.add_argument(
        "--meta",
        type=str,
        default="ridge",
        choices=["ridge", "linear", "linear_pos"],
        help="Meta model for stacking: ridge, linear, or linear_pos (non-negative weights)",
    )

    ap.add_argument(
        "--fixed-weights",
        type=str,
        default="",
        help=(
            "Optional fixed weights baseline. Examples: 'mrna=0.4,cnv=0.2,meth=0.2,mirna=0.2' "
            "or '0.4,0.2,0.2,0.2'"
        ),
    )

    ap.add_argument("--eps", type=float, default=1e-8, help="Small epsilon for inverse_rmse (default: 1e-8)")

    args = ap.parse_args()

    # Output directory
    if args.no_timestamp:
        out_dir = args.out_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(args.out_dir, f"ensemble_{args.blend}_{ts}")

    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Read all omics pieces
    pieces = []
    names = []
    for name, path in [
        ("mrna", args.mrna_pred),
        ("cnv", args.cnv_pred),
        ("meth", args.meth_pred),
        ("mirna", args.mirna_pred),
    ]:
        if path:
            pieces.append(read_pred(path, name))
            names.append(name)

    if len(pieces) < 2:
        raise ValueError("You must provide at least 2 omics prediction files.")

    # Coverage before merge (per drug)
    coverage = None
    for name, d in zip(names, pieces):
        c = d.groupby("drug").size().rename(f"n_{name}")
        coverage = c.to_frame() if coverage is None else coverage.join(c, how="outer")

    # Merge (inner join for fairness)
    df = pieces[0]
    for nxt in pieces[1:]:
        df = df.merge(nxt, on=["drug", "fold", "cell_line"], how="inner")

    if len(df) == 0:
        raise RuntimeError("Merged dataset is empty. Check that drug/fold/cell_line IDs match across omics.")

    # Check and unify y_true
    df = check_and_make_y_true(df, names, tol=1e-8)

    # Coverage after merge
    merged_counts = df.groupby("drug").size().rename("n_merged")
    coverage = coverage.join(merged_counts, how="outer").fillna(0).reset_index()
    for col in coverage.columns:
        if col != "drug":
            coverage[col] = coverage[col].astype(int)

    cov_path = os.path.join(out_dir, "coverage_report.csv")
    coverage.to_csv(cov_path, index=False)

    pred_cols = [f"pred_{n}" for n in names]

    # Baselines
    df["pred_mean"] = df[pred_cols].mean(axis=1)

    fixed_w = parse_fixed_weights(args.fixed_weights, names)
    if fixed_w:
        df["pred_fixed"] = weighted_average_rowwise(df, pred_cols, fixed_w)
    else:
        df["pred_fixed"] = np.nan

    # Fold-wise ensemble per drug
    pred_parts = []
    weights_rows = []
    metrics_rows = []

    for drug, g in df.groupby("drug"):
        folds = sorted(g["fold"].unique().tolist())
        for f in folds:
            te = g[g["fold"] == f].copy()
            tr = g[g["fold"] != f].copy()

            # Safety drop NaNs
            X_tr = tr[pred_cols].to_numpy(dtype=np.float64)
            y_tr = tr["y_true"].to_numpy(dtype=np.float64)
            X_te = te[pred_cols].to_numpy(dtype=np.float64)
            y_te = te["y_true"].to_numpy(dtype=np.float64)

            X_tr2, y_tr2, _ = drop_bad_rows(X_tr, y_tr)
            X_te2, y_te2, m_te = drop_bad_rows(X_te, y_te)

            if len(y_tr2) < 2 or len(y_te2) < 1:
                continue

            # Prepare reduced tr/te for learners
            tr2 = tr.iloc[np.where(np.isfinite(y_tr) & np.all(np.isfinite(X_tr), axis=1))[0]].copy()
            te2 = te.iloc[np.where(m_te)[0]].copy()

            method = args.blend
            intercept = 0.0
            w_vec = None

            if method == "stacking":
                meta = build_meta_model(args.meta, args.alpha)
                meta.fit(X_tr2, y_tr2)
                yhat2 = meta.predict(X_te2)
                w_vec = getattr(meta, "coef_", None)
                intercept = float(getattr(meta, "intercept_", 0.0))
                if w_vec is None:
                    w_vec = np.full(len(pred_cols), np.nan, dtype=float)

            elif method == "mean":
                # same as baseline mean
                yhat2 = te2[pred_cols].mean(axis=1).to_numpy(dtype=float)
                w_vec = _normalize_weights(np.ones(len(pred_cols), dtype=float))

            elif method == "median":
                yhat2 = te2[pred_cols].median(axis=1).to_numpy(dtype=float)
                w_vec = np.full(len(pred_cols), np.nan, dtype=float)
                intercept = float("nan")

            elif method == "inverse_rmse":
                w_vec = learn_weights_inverse_rmse(tr2, pred_cols, eps=float(args.eps))
                yhat2 = predict_with_constant_weights(te2, pred_cols, w_vec)

            elif method == "spearman":
                w_vec = learn_weights_spearman(tr2, pred_cols)
                yhat2 = predict_with_constant_weights(te2, pred_cols, w_vec)

            elif method == "pearson":
                w_vec = learn_weights_pearson(tr2, pred_cols)
                yhat2 = predict_with_constant_weights(te2, pred_cols, w_vec)

            elif method == "top1":
                w_vec = learn_weights_topk(tr2, pred_cols, k=1, eps=float(args.eps))
                yhat2 = predict_with_constant_weights(te2, pred_cols, w_vec)

            elif method == "top2":
                w_vec = learn_weights_topk(tr2, pred_cols, k=2, eps=float(args.eps))
                yhat2 = predict_with_constant_weights(te2, pred_cols, w_vec)

            else:
                raise ValueError(f"Unknown --blend: {method}")

            # Put predictions back into te
            te["pred_ens"] = np.nan
            te.loc[m_te, "pred_ens"] = yhat2
            te["blend"] = method

            keep_cols = ["drug", "fold", "cell_line", "y_true", "blend", "pred_ens", "pred_mean"] + pred_cols
            if fixed_w:
                keep_cols.insert(7, "pred_fixed")
            pred_parts.append(te[keep_cols])

            # Save learned weights per fold
            row = {
                "drug": drug,
                "fold": int(f),
                "blend": method,
                "meta": args.meta if method == "stacking" else "",
                "alpha": float(args.alpha) if method == "stacking" else float("nan"),
                "intercept": float(intercept) if np.isfinite(intercept) else float("nan"),
                "n_train": int(len(y_tr2)),
                "n_test": int(len(y_te2)),
            }
            if w_vec is None:
                w_vec = np.full(len(pred_cols), np.nan, dtype=float)
            for i, n in enumerate(names):
                row[f"w_{n}"] = float(w_vec[i]) if np.isfinite(w_vec[i]) else float("nan")
            weights_rows.append(row)

            # Metrics (only where predicted)
            pred_mean_te = te.loc[m_te, "pred_mean"].to_numpy(dtype=float)
            mrow = {
                "drug": drug,
                "fold": int(f),
                "blend": method,
                "RMSE_ens": rmse(y_te2, yhat2),
                "R2_ens": float(r2_score(y_te2, yhat2)),
                "Spearman_ens": safe_spearman(y_te2, yhat2),
                "Pearson_ens": safe_pearson(y_te2, yhat2),
                "RMSE_mean": rmse(y_te2, pred_mean_te),
                "R2_mean": float(r2_score(y_te2, pred_mean_te)),
            }
            if fixed_w:
                pred_fixed_te = te.loc[m_te, "pred_fixed"].to_numpy(dtype=float)
                mrow["RMSE_fixed"] = rmse(y_te2, pred_fixed_te)
                mrow["R2_fixed"] = float(r2_score(y_te2, pred_fixed_te))
            metrics_rows.append(mrow)

    if not pred_parts:
        raise RuntimeError("No predictions were produced. Possibly not enough folds/data per drug.")

    pred_out = pd.concat(pred_parts, ignore_index=True)
    weights_out = pd.DataFrame(weights_rows)
    metrics_out = pd.DataFrame(metrics_rows)

    # Summarize per drug
    # 1) OOF summary: concatenate all test-fold predictions for the same drug, then compute metrics once.
    #    This matches the PDF logic in gpu_train_foldmap*.py.
    summary_rows = []
    for drug, g in pred_out.groupby("drug"):
        row = {"drug": drug}

        y = g["y_true"].to_numpy(dtype=float)

        # Ensemble OOF metrics
        p_ens = g["pred_ens"].to_numpy(dtype=float)
        m_ens = np.isfinite(y) & np.isfinite(p_ens)
        if np.any(m_ens):
            y_ens = y[m_ens]
            p_ens2 = p_ens[m_ens]
            row["n_oof"] = int(m_ens.sum())
            row["RMSE_ens"] = rmse(y_ens, p_ens2)
            row["R2_ens"] = float(r2_score(y_ens, p_ens2))
            row["Spearman_ens"] = safe_spearman(y_ens, p_ens2)
            row["Pearson_ens"] = safe_pearson(y_ens, p_ens2)
        else:
            row["n_oof"] = 0
            row["RMSE_ens"] = float("nan")
            row["R2_ens"] = float("nan")
            row["Spearman_ens"] = float("nan")
            row["Pearson_ens"] = float("nan")

        # Mean-baseline OOF metrics
        p_mean = g["pred_mean"].to_numpy(dtype=float)
        m_mean = np.isfinite(y) & np.isfinite(p_mean)
        if np.any(m_mean):
            y_mean = y[m_mean]
            p_mean2 = p_mean[m_mean]
            row["RMSE_mean"] = rmse(y_mean, p_mean2)
            row["R2_mean"] = float(r2_score(y_mean, p_mean2))
        else:
            row["RMSE_mean"] = float("nan")
            row["R2_mean"] = float("nan")

        # Fixed-weight baseline OOF metrics
        if fixed_w:
            p_fixed = g["pred_fixed"].to_numpy(dtype=float)
            m_fixed = np.isfinite(y) & np.isfinite(p_fixed)
            if np.any(m_fixed):
                y_fixed = y[m_fixed]
                p_fixed2 = p_fixed[m_fixed]
                row["RMSE_fixed"] = rmse(y_fixed, p_fixed2)
                row["R2_fixed"] = float(r2_score(y_fixed, p_fixed2))
            else:
                row["RMSE_fixed"] = float("nan")
                row["R2_fixed"] = float("nan")

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    # 2) Keep old fold mean/std metrics as extra reference columns.
    agg_cols = ["RMSE_ens", "R2_ens", "Spearman_ens", "Pearson_ens", "RMSE_mean", "R2_mean"]
    if fixed_w:
        agg_cols += ["RMSE_fixed", "R2_fixed"]

    fold_stats = metrics_out.groupby("drug")[agg_cols].agg(["mean", "std"])
    fold_stats.columns = [f"{a}_fold_{b}" for a, b in fold_stats.columns]
    fold_stats = fold_stats.reset_index()

    summary = summary.merge(fold_stats, on="drug", how="left")

    # Final weights per drug (for deployment)
    final_rows = []
    for drug, g in df.groupby("drug"):
        X = g[pred_cols].to_numpy(dtype=np.float64)
        y = g["y_true"].to_numpy(dtype=np.float64)
        X2, y2, _ = drop_bad_rows(X, y)
        if len(y2) < 2:
            continue

        method = args.blend
        intercept = 0.0
        w_vec = None

        g2 = g.iloc[np.where(np.isfinite(y) & np.all(np.isfinite(X), axis=1))[0]].copy()

        if method == "stacking":
            meta = build_meta_model(args.meta, args.alpha)
            meta.fit(X2, y2)
            w_vec = getattr(meta, "coef_", None)
            intercept = float(getattr(meta, "intercept_", 0.0))
            if w_vec is None:
                w_vec = np.full(len(pred_cols), np.nan, dtype=float)

        elif method == "mean":
            w_vec = _normalize_weights(np.ones(len(pred_cols), dtype=float))
            intercept = 0.0

        elif method == "median":
            w_vec = np.full(len(pred_cols), np.nan, dtype=float)
            intercept = float("nan")

        elif method == "inverse_rmse":
            w_vec = learn_weights_inverse_rmse(g2, pred_cols, eps=float(args.eps))
            intercept = 0.0

        elif method == "spearman":
            w_vec = learn_weights_spearman(g2, pred_cols)
            intercept = 0.0

        elif method == "pearson":
            w_vec = learn_weights_pearson(g2, pred_cols)
            intercept = 0.0

        elif method == "top1":
            w_vec = learn_weights_topk(g2, pred_cols, k=1, eps=float(args.eps))
            intercept = 0.0

        elif method == "top2":
            w_vec = learn_weights_topk(g2, pred_cols, k=2, eps=float(args.eps))
            intercept = 0.0

        else:
            w_vec = np.full(len(pred_cols), np.nan, dtype=float)
            intercept = float("nan")

        row = {
            "drug": drug,
            "blend": method,
            "meta": args.meta if method == "stacking" else "",
            "alpha": float(args.alpha) if method == "stacking" else float("nan"),
            "intercept": float(intercept) if np.isfinite(intercept) else float("nan"),
            "n": int(len(y2)),
        }
        for i, n in enumerate(names):
            row[f"w_{n}"] = float(w_vec[i]) if (w_vec is not None and np.isfinite(w_vec[i])) else float("nan")
        final_rows.append(row)

    final_weights = pd.DataFrame(final_rows)

    # Save outputs
    pred_path = os.path.join(out_dir, "ensemble_predictions.csv")
    weights_path = os.path.join(out_dir, "learned_weights.csv")
    metrics_path = os.path.join(out_dir, "ensemble_metrics_folds.csv")
    summary_path = os.path.join(out_dir, "ensemble_summary.csv")
    final_path = os.path.join(out_dir, "final_weights_per_drug.csv")

    pred_out.to_csv(pred_path, index=False)
    weights_out.to_csv(weights_path, index=False)
    metrics_out.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    final_weights.to_csv(final_path, index=False)

    print("Merged rows:", len(df), "| drugs:", df["drug"].nunique(), "| omics:", names)
    print("Blend method:", args.blend)
    print("Saved:")
    print(" -", pred_path)
    print(" -", weights_path)
    print(" -", metrics_path)
    print(" -", summary_path)
    print(" -", final_path)
    print(" -", cov_path)

    if "RMSE_ens" in summary.columns:
        print("\nTop by RMSE_ens (OOF over concatenated test folds):")
        print(summary.sort_values("RMSE_ens").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
