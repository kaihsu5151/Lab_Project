import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def standardize_name(name: str) -> Optional[str]:
    if pd.isna(name):
        return None
    x = str(name).strip().upper()
    x = re.sub(r"_(LUNG|HAEMATOPOIETIC_AND_LYMPHOID_TISSUE|BLOOD|MYELOID|LYMPHOID|HAEMATOPOIETIC|[A-Z]+)$", "", x)
    x = re.sub(r"[^A-Z0-9]", "", x)
    return x if x else None


def standardize_drug(drug: str) -> Optional[str]:
    if pd.isna(drug):
        return None
    x = str(drug).strip()
    x = re.sub(r"\s+", " ", x)
    x = x.upper()
    return x if x else None


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def infer_tissue_from_expression_name(expr_name: str) -> str:
    x = str(expr_name or "").upper()
    if "_LUNG" in x:
        return "LUNG"
    if "_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE" in x:
        return "LYMPH"
    # fallback: heuristic
    if "LYMPHOID" in x or "HAEMATOPOIETIC" in x or "BLOOD" in x:
        return "LYMPH"
    return "OTHER"


def load_tissue_labels(map_csv: Path) -> Dict[str, str]:
    """
    Returns dict: std_cell_line -> tissue in {LUNG, LYMPH, OTHER}
    Uses `expression_mapping_lung_and_lymphoid_only.csv`.
    """
    m = pd.read_csv(map_csv, low_memory=False)
    need = {"standardized", "expression_name"}
    missing = need - set(m.columns)
    if missing:
        raise ValueError(f"Mapping CSV missing columns {missing}: {map_csv}")
    m = m.copy()
    m["std"] = m["standardized"].map(standardize_name)
    m["tissue"] = m["expression_name"].map(infer_tissue_from_expression_name)
    m = m.dropna(subset=["std"])
    m = m.drop_duplicates("std")
    return dict(zip(m["std"].astype(str), m["tissue"].astype(str)))


def cohen_d_markers(feature_df: pd.DataFrame, tissue: pd.Series, d_thresh: float = 1.0) -> set:
    """
    Compute tissue markers by Cohen's d (LUNG vs LYMPH) using feature_df only (never touches y).
    feature_df index: std cell line
    tissue: Series aligned to feature_df index, values in {LUNG, LYMPH, ...}
    """
    lab = tissue.reindex(feature_df.index).astype(str)
    mask_a = lab == "LUNG"
    mask_b = lab == "LYMPH"
    if int(mask_a.sum()) < 5 or int(mask_b.sum()) < 5:
        return set()

    X = feature_df.to_numpy(dtype=np.float64, copy=False)
    XA = X[mask_a.to_numpy(), :]
    XB = X[mask_b.to_numpy(), :]

    meanA = np.nanmean(XA, axis=0)
    meanB = np.nanmean(XB, axis=0)
    varA = np.nanvar(XA, axis=0, ddof=1)
    varB = np.nanvar(XB, axis=0, ddof=1)
    nA = XA.shape[0]
    nB = XB.shape[0]
    pooled = np.sqrt(((nA - 1) * varA + (nB - 1) * varB) / max(nA + nB - 2, 1))
    d = (meanA - meanB) / pooled
    d[~np.isfinite(d)] = 0.0
    keep = np.abs(d) >= float(d_thresh)
    return set(feature_df.columns[keep].astype(str).tolist())


def impute_mean(X: np.ndarray) -> np.ndarray:
    """Column-mean impute for NaN/inf."""
    X = np.asarray(X, dtype=np.float32)
    bad = ~np.isfinite(X)
    if not bad.any():
        return X
    col_mean = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0).astype(np.float32)
    X2 = X.copy()
    rr, cc = np.where(bad)
    X2[rr, cc] = col_mean[cc]
    return X2


def topn_corr_drop_refill(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: np.ndarray,
    marker_set: set,
    top_n: int,
    *,
    big_n_factor: int = 5,
    big_n_extra: int = 5000,
) -> np.ndarray:
    """
    Approximate the training-time drop_refill feature selection using ALL samples for visualization:
    - rank features by abs(Pearson corr) with y
    - take a larger pool (big_n), drop markers, then take first top_n non-markers
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if len(y) != X.shape[0]:
        raise ValueError("X and y row mismatch")

    p = X.shape[1]
    if top_n <= 0:
        return np.arange(p, dtype=np.int32)

    big_n = int(min(p, max(top_n * big_n_factor, top_n + big_n_extra)))

    y0 = y - np.nanmean(y)
    X0 = X - np.nanmean(X, axis=0, keepdims=True)
    num = np.nansum(X0 * y0[:, None], axis=0)
    den = np.sqrt(np.nansum(X0**2, axis=0) * np.nansum(y0**2))
    corr = np.divide(num, den, out=np.zeros_like(num), where=(den != 0))
    score = np.abs(corr)

    # pick big_n by partial sort
    idx_pool = np.argpartition(score, -big_n)[-big_n:]
    idx_pool = idx_pool[np.argsort(score[idx_pool])[::-1]]

    if marker_set:
        keep = np.array([str(feature_names[j]) not in marker_set for j in idx_pool], dtype=bool)
        idx_pool = idx_pool[keep]

    if len(idx_pool) == 0:
        # fallback: ignore marker dropping
        idx_pool = np.argsort(score)[::-1]

    return idx_pool[: int(min(top_n, len(idx_pool)))].astype(np.int32)


def topn_corr_baseline(
    X: np.ndarray,
    y: np.ndarray,
    top_n: int,
) -> np.ndarray:
    """
    Baseline feature selection (no tissue marker removal):
    rank features by abs(Pearson corr) with y and take top_n.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if len(y) != X.shape[0]:
        raise ValueError("X and y row mismatch")

    p = X.shape[1]
    if top_n <= 0:
        return np.arange(p, dtype=np.int32)

    y0 = y - np.nanmean(y)
    X0 = X - np.nanmean(X, axis=0, keepdims=True)
    num = np.nansum(X0 * y0[:, None], axis=0)
    den = np.sqrt(np.nansum(X0**2, axis=0) * np.nansum(y0**2))
    corr = np.divide(num, den, out=np.zeros_like(num), where=(den != 0))
    score = np.abs(corr)

    idx = np.argpartition(score, -int(min(top_n, p)))[-int(min(top_n, p)) :]
    idx = idx[np.argsort(score[idx])[::-1]]
    return idx.astype(np.int32)


@dataclass(frozen=True)
class OmicsSpec:
    name: str
    csv_file: str
    top_n: int


OMICS_SPECS: List[OmicsSpec] = [
    OmicsSpec(name="mRNA", csv_file="processed_expression_unnormalized_lung.csv", top_n=1000),
    OmicsSpec(name="CNV", csv_file="DNACopyNumber_final_lung_and_blood.csv", top_n=1000),
    OmicsSpec(name="Methylation", csv_file="Methylation_final_lung_and_blood.csv", top_n=2000),
    OmicsSpec(name="miRNA", csv_file="miRNA_final_lung_and_blood.csv", top_n=300),
]


def axis_limits_from_embeddings(
    embeddings: Sequence[np.ndarray],
    pad_frac: float = 0.05,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Union min/max of UMAP1/UMAP2 across multiple embeddings (same display scale)."""
    parts = [e for e in embeddings if e is not None and len(e) > 0]
    if not parts:
        return (-1.0, 1.0), (-1.0, 1.0)
    xs = np.concatenate([np.asarray(p)[:, 0] for p in parts])
    ys = np.concatenate([np.asarray(p)[:, 1] for p in parts])
    xlo, xhi = float(np.nanmin(xs)), float(np.nanmax(xs))
    ylo, yhi = float(np.nanmin(ys)), float(np.nanmax(ys))
    xpad = (xhi - xlo) * float(pad_frac) + 1e-9
    ypad = (yhi - ylo) * float(pad_frac) + 1e-9
    return (xlo - xpad, xhi + xpad), (ylo - ypad, yhi + ypad)


def plot_umap(
    emb: np.ndarray,
    out_png: Path,
    title: str,
    *,
    color: np.ndarray,
    is_continuous: bool,
    cmap: str = "viridis",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    import matplotlib.pyplot as plt

    safe_mkdir(out_png.parent)

    x = emb[:, 0]
    y = emb[:, 1]

    plt.figure(figsize=(6.2, 5.2))
    if is_continuous:
        sc = plt.scatter(x, y, c=color, s=12, alpha=0.9, cmap=cmap)
        cb = plt.colorbar(sc)
        cb.set_label("IC50 (Y)")
    else:
        # categorical
        categories = pd.Series(color).astype(str).fillna("NA")
        palette = {"LUNG": "#1f77b4", "LYMPH": "#ff7f0e", "OTHER": "#7f7f7f", "NA": "#7f7f7f"}
        for cat in categories.unique():
            mask = categories == cat
            plt.scatter(x[mask], y[mask], s=12, alpha=0.9, label=cat, c=palette.get(cat, "#7f7f7f"))
        plt.legend(title="tissue", loc="best", fontsize=9)

    plt.title(title)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    ax = plt.gca()
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def compute_umap(X: np.ndarray, random_state: int = 42) -> np.ndarray:
    try:
        import umap  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "缺少 umap-learn。請先安裝：pip install umap-learn"
        ) from e

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=int(random_state),
    )
    return reducer.fit_transform(X)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=str(Path("train_data")))
    ap.add_argument("--out-dir", type=str, default=str(Path("umap_outputs")))
    ap.add_argument("--drugs-csv", type=str, default=str(Path("train_data") / "drugs_57.csv"))
    ap.add_argument("--drug-response-csv", type=str, default=str(Path("train_data") / "processed_drug_response_with_prism.csv"))
    ap.add_argument("--tissue-map-csv", type=str, default=str(Path("train_data") / "expression_mapping_lung_and_lymphoid_only.csv"))
    ap.add_argument("--marker-d", type=float, default=1.0)
    ap.add_argument(
        "--ablation-mode",
        type=str,
        default="drop_refill",
        choices=["baseline", "drop_refill", "both"],
        help="baseline / drop_refill 單一模式；both：同一藥物同資料夾輸出兩種，且 UMAP 座標軸範圍一致以便比較",
    )
    ap.add_argument("--only-omics", type=str, default=None, help="Comma-separated subset: mRNA,CNV,Methylation,miRNA")
    ap.add_argument("--only-drug", type=str, default=None, help="Run a single drug (name).")
    ap.add_argument("--max-drugs", type=int, default=0, help="0 = no limit")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    tissue_map = load_tissue_labels(Path(args.tissue_map_csv))
    tissue_series = pd.Series(tissue_map, name="tissue")

    drug_list_df = pd.read_csv(args.drugs_csv, low_memory=False)
    drug_col = "drug" if "drug" in drug_list_df.columns else drug_list_df.columns[0]
    drugs = [standardize_drug(x) for x in drug_list_df[drug_col].dropna().astype(str).tolist()]
    drugs = [d for d in drugs if d]
    # de-dup preserve order
    seen = set()
    drugs2 = []
    for d in drugs:
        if d in seen:
            continue
        seen.add(d)
        drugs2.append(d)
    drugs = drugs2

    if args.only_drug:
        drugs = [standardize_drug(args.only_drug)]

    if args.max_drugs and args.max_drugs > 0:
        drugs = drugs[: int(args.max_drugs)]

    keep_omics = None
    if args.only_omics:
        keep_omics = {x.strip() for x in str(args.only_omics).split(",") if x.strip()}

    # load drug response (only needed cols)
    dr = pd.read_csv(args.drug_response_csv, low_memory=False, usecols=["Drug", "CellLineName", "Y"])
    dr["Drug"] = dr["Drug"].map(standardize_drug)
    dr["std"] = dr["CellLineName"].map(standardize_name)
    dr["Y"] = pd.to_numeric(dr["Y"], errors="coerce")
    dr = dr.dropna(subset=["Drug", "std", "Y"])

    # loop omics
    for spec in OMICS_SPECS:
        if keep_omics is not None and spec.name not in keep_omics:
            continue

        omics_path = data_dir / spec.csv_file
        if not omics_path.exists():
            raise FileNotFoundError(f"找不到組學檔：{omics_path}")

        # load omics wide matrix
        feat_df = pd.read_csv(omics_path, low_memory=False, index_col=0)
        # standardize index defensively
        feat_df.index = feat_df.index.map(standardize_name)
        feat_df = feat_df[~feat_df.index.isna()].copy()
        feat_df = feat_df[~feat_df.index.duplicated(keep="first")]

        # align tissue labels
        tissue_aligned = tissue_series.reindex(feat_df.index).fillna("OTHER")
        marker_set = cohen_d_markers(feat_df, tissue_aligned, d_thresh=float(args.marker_d))

        # per drug
        for drug in drugs:
            if drug is None:
                continue
            sub = dr.loc[dr["Drug"] == drug, ["std", "Y"]].copy()
            if len(sub) == 0:
                continue
            sub_agg = sub.groupby("std", as_index=False).agg(Y=("Y", "median"), n=("Y", "size"))
            sub_agg = sub_agg[sub_agg["std"].isin(feat_df.index)].copy()
            if len(sub_agg) < 10:
                continue

            X_all = feat_df.loc[sub_agg["std"]].to_numpy(dtype=np.float32, copy=False)
            y_all = sub_agg["Y"].to_numpy(dtype=np.float32, copy=False)
            X_all = impute_mean(X_all)

            feat_names = feat_df.columns.to_numpy(dtype=object, copy=False)
            tissue = tissue_aligned.reindex(sub_agg["std"]).to_numpy(dtype=object)

            d_out = out_dir / spec.name / drug
            safe_mkdir(d_out)

            from sklearn.preprocessing import StandardScaler

            if args.ablation_mode == "both":
                idx_b = topn_corr_baseline(X_all, y_all, top_n=int(spec.top_n))
                idx_d = topn_corr_drop_refill(
                    X=X_all,
                    y=y_all,
                    feature_names=feat_names,
                    marker_set=marker_set,
                    top_n=int(spec.top_n),
                )

                X_b = impute_mean(X_all[:, idx_b])
                X_d = impute_mean(X_all[:, idx_d])
                X_b = StandardScaler().fit_transform(X_b).astype(np.float32, copy=False)
                X_d = StandardScaler().fit_transform(X_d).astype(np.float32, copy=False)

                emb_b = compute_umap(X_b, random_state=int(args.random_state))
                emb_d = compute_umap(X_d, random_state=int(args.random_state))
                xlim, ylim = axis_limits_from_embeddings([emb_b, emb_d])

                plot_umap(
                    emb_b,
                    d_out / "umap_tissue_baseline.png",
                    title=f"{spec.name} | {drug} | tissue (baseline)",
                    color=tissue,
                    is_continuous=False,
                    xlim=xlim,
                    ylim=ylim,
                )
                plot_umap(
                    emb_d,
                    d_out / "umap_tissue_drop_refill.png",
                    title=f"{spec.name} | {drug} | tissue (drop_refill)",
                    color=tissue,
                    is_continuous=False,
                    xlim=xlim,
                    ylim=ylim,
                )
                plot_umap(
                    emb_b,
                    d_out / "umap_ic50_baseline.png",
                    title=f"{spec.name} | {drug} | IC50(Y) (baseline)",
                    color=y_all,
                    is_continuous=True,
                    xlim=xlim,
                    ylim=ylim,
                )
                plot_umap(
                    emb_d,
                    d_out / "umap_ic50_drop_refill.png",
                    title=f"{spec.name} | {drug} | IC50(Y) (drop_refill)",
                    color=y_all,
                    is_continuous=True,
                    xlim=xlim,
                    ylim=ylim,
                )

                meta = sub_agg.copy()
                meta["tissue"] = tissue
                meta.to_csv(d_out / "points_meta.csv", index=False)
                pd.Series(feat_names[idx_b].astype(str)).to_csv(
                    d_out / "selected_features_baseline.csv",
                    index=False,
                    header=["feature"],
                )
                pd.Series(feat_names[idx_d].astype(str)).to_csv(
                    d_out / "selected_features_drop_refill.csv",
                    index=False,
                    header=["feature"],
                )
                continue

            if args.ablation_mode == "baseline":
                idx = topn_corr_baseline(X_all, y_all, top_n=int(spec.top_n))
            else:
                idx = topn_corr_drop_refill(
                    X=X_all,
                    y=y_all,
                    feature_names=feat_names,
                    marker_set=marker_set,
                    top_n=int(spec.top_n),
                )

            X = X_all[:, idx]
            X = impute_mean(X)

            X = StandardScaler().fit_transform(X).astype(np.float32, copy=False)
            emb = compute_umap(X, random_state=int(args.random_state))

            plot_umap(
                emb,
                d_out / "umap_tissue.png",
                title=f"{spec.name} | {drug} | tissue ({args.ablation_mode})",
                color=tissue,
                is_continuous=False,
            )
            plot_umap(
                emb,
                d_out / "umap_ic50.png",
                title=f"{spec.name} | {drug} | IC50(Y) ({args.ablation_mode})",
                color=y_all,
                is_continuous=True,
            )

            meta = sub_agg.copy()
            meta["tissue"] = tissue
            meta.to_csv(d_out / "points_meta.csv", index=False)
            pd.Series(feat_names[idx].astype(str)).to_csv(
                d_out / f"selected_features_{args.ablation_mode}.csv",
                index=False,
                header=["feature"],
            )


if __name__ == "__main__":
    main()

