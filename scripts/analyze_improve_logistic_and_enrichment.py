"""
進一步分析（二合一）：
1) 多變量邏輯迴歸：P(improve) ~ theme_* 二元特徵 + 物化／臨床共變量（標準化）
2) 標靶富集：targets_long.csv 的 SINGLE PROTEIN（UniProt）→ 基因符號 → g:Profiler 路徑／GO 富集

依賴：pandas, numpy, scipy, scikit-learn, mygene, requests
  pip install pandas numpy scipy scikit-learn mygene requests

輸出目錄：compare_final/improve_followup_analysis/

注意：樣本數小（~57），迴歸係數解讀宜保守；富集需網路連線 g:Profiler。
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 路徑（相對於專案根目錄 Lab_Project）
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DRUG_TABLE = ROOT / "compare_final/analyze_improve_worsen_v7_theme_token_summary/drug_delta_table.csv"
TARGETS_LONG = ROOT / "compare_final/targets_long.csv"
OUT_DIR = ROOT / "compare_final/improve_followup_analysis"

# 物化／結構共變量（與 drug_delta_table 欄位一致；缺失會以中位數填補）
COVAR_COLS = [
    "full_mwt",
    "alogp_xlogp",
    "psa_tpsa",
    "hba",
    "hbd",
    "rtb",
    "aromatic_rings",
    "heavy_atoms",
    "qed_weighted",
    "max_phase",
]

GPROFILER_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"


def load_drug_frame() -> pd.DataFrame:
    df = pd.read_csv(DRUG_TABLE)
    df["drug_key"] = df["drug"].astype(str).str.strip().str.upper()
    df["y_improve"] = (df["delta_group"].astype(str).str.lower() == "improve").astype(int)
    return df


def run_theme_logistic(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    theme_cols = [c for c in df.columns if c.startswith("theme_")]
    use_cols = theme_cols + COVAR_COLS
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"缺少欄位: {missing}")

    X = df[use_cols].copy()
    y = df["y_improve"].values

    # 僅保留 y 可辨識者
    mask = np.isfinite(y)
    X = X.loc[mask]
    y = y[mask]

    n_improve = int(y.sum())
    n_total = len(y)
    print(f"[logistic] n_drugs={n_total}, n_improve={n_improve}, prevalence={n_improve/n_total:.3f}")

    preproc = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), COVAR_COLS),
            ("theme", "passthrough", theme_cols),
        ]
    )

    # 類別不平衡時略加重少數類權重
    n_neg = n_total - n_improve
    w_pos = n_neg / max(n_improve, 1)
    class_weight = {0: 1.0, 1: w_pos}

    clf = LogisticRegressionCV(
        Cs=10,
        cv=min(5, n_total // 3) or 3,
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        class_weight=class_weight,
        random_state=42,
        n_jobs=None,
    )
    pipe = Pipeline([("prep", preproc), ("clf", clf)])
    pipe.fit(X, y)

    model: LogisticRegressionCV = pipe.named_steps["clf"]
    feature_names = theme_cols + COVAR_COLS
    coefs = model.coef_.ravel()
    out = pd.DataFrame(
        {
            "feature": feature_names,
            "coef": coefs,
            "odds_ratio": np.exp(coefs),
        }
    ).sort_values("coef", key=abs, ascending=False)

    summary = {
        "cv_scores_mean": float(np.mean(model.scores_[1])),
        "best_C": float(model.C_[0]),
    }
    return out, summary


def targets_to_uniprot_by_drug() -> pd.DataFrame:
    t = pd.read_csv(TARGETS_LONG)
    t["drug_key"] = t["drug"].astype(str).str.strip().str.upper()
    prot = t[t["target_type"].astype(str).str.upper() == "SINGLE PROTEIN"].copy()
    prot["accessions"] = prot["accessions"].astype(str).str.strip()
    prot = prot[prot["accessions"].str.len() > 0]
    prot = prot[prot["accessions"] != "nan"]
    # 一藥多列：同一 UniProt 只保留一次
    prot = prot.drop_duplicates(subset=["drug_key", "accessions"])
    return prot[["drug_key", "accessions", "pref_name"]]


def uniprot_to_genesymbols(uniprot_ids: list[str]) -> dict[str, str]:
    """UniProt accession → gene symbol（需 mygene）。"""
    try:
        import mygene
    except ImportError as e:
        raise SystemExit("請安裝 mygene: pip install mygene") from e

    mg = mygene.MyGeneInfo()
    # batch query
    q = mg.querymany(
        list(set(uniprot_ids)),
        scopes="uniprot",
        fields="symbol",
        species="human",
        verbose=False,
    )
    m: dict[str, str] = {}
    for row in q:
        uid = row.get("query")
        sym = row.get("symbol")
        if isinstance(sym, list):
            sym = sym[0] if sym else None
        if uid and sym:
            m[str(uid).upper()] = str(sym)
    return m


def build_gene_foreground_background(
    drug_df: pd.DataFrame, prot_df: pd.DataFrame, u2g: dict[str, str]
) -> tuple[list[str], list[str]]:
    keys = set(drug_df["drug_key"])
    improv_keys = set(drug_df.loc[drug_df["y_improve"] == 1, "drug_key"])

    def genes_for(keyset: set[str]) -> set[str]:
        g: set[str] = set()
        sub = prot_df[prot_df["drug_key"].isin(keyset)]
        for acc in sub["accessions"].str.upper():
            if acc in u2g:
                g.add(u2g[acc])
        return g

    fg = sorted(genes_for(improv_keys))
    bg = sorted(genes_for(keys))
    return fg, bg


def run_gprofiler(foreground: list[str], background: list[str]) -> pd.DataFrame:
    """g:Profiler GOST；background 為自訂背景（全 57 藥標靶基因）。"""
    # g:Profiler：勿使用已移除參數 `significant`。要取得完整列表需設 all_results。
    payload = {
        "organism": "hsapiens",
        "query": foreground,
        "background": background,
        "domain_scope": "custom",
        "sources": ["GO:BP", "GO:MF", "GO:CC", "REAC", "KEGG"],
        "user_threshold": 0.05,
        "ordered": False,
        "all_results": True,
    }
    r = requests.post(GPROFILER_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    rows = data.get("result") or []
    if not rows:
        return pd.DataFrame()

    recs = []
    for row in rows:
        ints = row.get("intersections") or []
        flat: list[str] = []
        if ints and isinstance(ints[0], list):
            flat = [str(x) for x in ints[0]]
        elif ints:
            flat = [str(x) for x in ints]
        recs.append(
            {
                "source": row.get("source"),
                "native": row.get("native"),
                "name": row.get("name"),
                "p_value": row.get("p_value"),
                "significant": row.get("significant"),
                "term_size": row.get("term_size"),
                "query_size": row.get("query_size"),
                "intersection_size": row.get("intersection_size"),
                "intersection_genes": ",".join(flat),
            }
        )
    out = pd.DataFrame(recs)
    if "p_value" in out.columns:
        out = out.sort_values("p_value")
    return out


def filter_significant(enrich_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    if enrich_df.empty or "p_value" not in enrich_df.columns:
        return enrich_df
    return enrich_df.loc[enrich_df["p_value"] <= alpha].copy()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    drug_df = load_drug_frame()
    drug_df.to_csv(OUT_DIR / "drug_table_with_y.csv", index=False)

    # --- 1. Logistic ---
    coef_df, summ = run_theme_logistic(drug_df)
    coef_df.to_csv(OUT_DIR / "logistic_theme_and_covariates_coef.csv", index=False)
    with open(OUT_DIR / "logistic_meta.json", "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2)
    print(f"[logistic] 係數表: {OUT_DIR / 'logistic_theme_and_covariates_coef.csv'}")

    # --- 2. Targets → genes → enrichment ---
    prot_df = targets_to_uniprot_by_drug()
    all_acc = prot_df["accessions"].str.upper().unique().tolist()
    print(f"[targets] SINGLE PROTEIN 列數（去重 drug×acc）: {len(prot_df)}, 唯一 UniProt: {len(all_acc)}")

    u2g = uniprot_to_genesymbols(all_acc)
    print(f"[mygene] 對應到基因符號: {len(u2g)} / {len(all_acc)}")

    fg_genes, bg_genes = build_gene_foreground_background(drug_df, prot_df, u2g)
    Path(OUT_DIR / "genes_foreground_improve.txt").write_text("\n".join(fg_genes), encoding="utf-8")
    Path(OUT_DIR / "genes_background_all_drugs.txt").write_text("\n".join(bg_genes), encoding="utf-8")
    print(f"[genes] foreground (improve): {len(fg_genes)}, background (all drugs): {len(bg_genes)}")

    if len(fg_genes) < 3:
        print("[warn] 前景基因過少，跳過 g:Profiler。", file=sys.stderr)
        return

    try:
        enrich_df = run_gprofiler(fg_genes, bg_genes)
        enrich_df.to_csv(OUT_DIR / "gprofiler_enrichment_improve_vs_background.csv", index=False)
        sig_df = filter_significant(enrich_df)
        sig_df.to_csv(OUT_DIR / "gprofiler_enrichment_p_le_0.05.csv", index=False)
        print(
            f"[gprofiler] 總條目: {len(enrich_df)}, p<=0.05: {len(sig_df)} → "
            f"{OUT_DIR / 'gprofiler_enrichment_improve_vs_background.csv'}"
        )
    except requests.RequestException as e:
        print(f"[error] g:Profiler 請求失敗（需網路）: {e}", file=sys.stderr)
        print("已寫入基因清單，可改用手動上傳 g:Profiler 網頁。", file=sys.stderr)


if __name__ == "__main__":
    main()
