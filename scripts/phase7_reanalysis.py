"""
Phase 7: Re-Analyse mit echtem externen Authority-Signal
Ersetzt domain_rank_decile (= n_appearances-Proxy) durch
externes Signal aus DataForSEO domain_overview.

Input:
  data/04_dataset_clean.csv           (Original Clean Dataset)
  data/05_domain_authority.csv        (Neue externe Authority-Daten)

Output:
  data/06_dataset_with_authority.csv  (Merged)
  analysis_v2/model_results.json
  analysis_v2/ymyl_comparison.json
  analysis_v2/robustness_checks.json
  analysis_v2/methodology_log.json    (was wurde geaendert)
"""

import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from pathlib import Path
from datetime import datetime

ROOT = Path("/home/christian/url-studie")
DATA = ROOT / "data"
OUT  = ROOT / "analysis_v2"
OUT.mkdir(exist_ok=True)


def load_and_merge():
    """Lade Clean-Dataset und merge externe Authority-Daten."""
    clean = pd.read_csv(DATA / "04_dataset_clean.csv", low_memory=False)
    auth  = pd.read_csv(DATA / "05_domain_authority.csv")

    auth_ok = auth[auth["status"] == "ok"].copy()
    print(f"Clean: {len(clean):,} Zeilen, {clean['domain_clean'].nunique():,} Domains")
    print(f"Authority: {len(auth_ok):,} Domains mit OK-Status")

    # Merge auf domain_clean
    merged = clean.merge(
        auth_ok[["domain", "organic_keywords_count", "organic_etv",
                 "organic_pos_1", "organic_pos_2_3", "organic_pos_4_10"]],
        left_on="domain_clean", right_on="domain", how="left",
    )
    print(f"Gematched: {merged['organic_keywords_count'].notna().sum():,} von {len(merged):,}")

    # log-transform fuer Skalenrobustheit
    merged["log_organic_kw"]   = np.log1p(merged["organic_keywords_count"])
    merged["log_organic_etv"]  = np.log1p(merged["organic_etv"])
    merged["log_organic_top3"] = np.log1p((merged["organic_pos_1"].fillna(0) + merged["organic_pos_2_3"].fillna(0)))

    # NEUER Authority-Decile basierend auf log_organic_kw
    merged["authority_decile_v2"] = pd.qcut(
        merged["log_organic_kw"].fillna(-1), 10, labels=False, duplicates="drop"
    ) + 1

    # Numerische Spalten erzwingen
    for col in ["path_length", "position", "is_ymyl", "path_depth",
                "keyword_in_url", "has_parameters", "word_count_path",
                "has_featured_snippet", "keyword_difficulty"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    return merged


def compute_correlations(df):
    """Spearman-Korrelationen aller Schluessel-Variablen vs. position."""
    out = {}
    for col, label in [
        ("path_length",          "path_length"),
        ("path_depth",           "path_depth"),
        ("keyword_in_url",       "keyword_in_url"),
        ("authority_decile_v2",  "authority_decile_v2 (NEU, extern)"),
        ("log_organic_kw",       "log_organic_kw (NEU)"),
        ("log_organic_etv",      "log_organic_etv (NEU)"),
        ("domain_rank_decile",   "domain_rank_decile (ALT, n_appearances-Proxy)"),
    ]:
        d = df.dropna(subset=[col, "position"])
        if len(d) > 100:
            rho, p = stats.spearmanr(d[col], d["position"])
            out[col] = {"label": label, "rho": round(rho, 4), "p": round(p, 6), "n": len(d)}
    return out


def run_ols(df, formula, label):
    """OLS mit HC3 robusten Standardfehlern."""
    sub = df.dropna(subset=["path_length","position","authority_decile_v2","page_type",
                            "keyword_difficulty","search_intent","path_depth","keyword_in_url",
                            "has_parameters","word_count_path","has_featured_snippet"])
    try:
        m = smf.ols(formula, data=sub).fit(cov_type="HC3")
        beta = m.params.get("path_length")
        pval = m.pvalues.get("path_length")
        return {
            "label": label,
            "formula": formula,
            "n": int(m.nobs),
            "r2": round(m.rsquared, 4),
            "r2_adj": round(m.rsquared_adj, 4),
            "aic": round(m.aic, 2),
            "path_length_beta": round(float(beta), 6) if beta is not None else None,
            "path_length_pval": round(float(pval), 6) if pval is not None else None,
        }
    except Exception as e:
        return {"label": label, "error": str(e)}


def run_ymyl_segmented(df):
    """YMYL-Analyse mit neuem Authority-Signal."""
    formula = ("position ~ path_length + authority_decile_v2 + C(page_type) + "
               "path_depth + keyword_in_url + keyword_difficulty + C(search_intent)")
    out = {}
    for label, mask in [
        ("YMYL",     df["is_ymyl"] == 1),
        ("NonYMYL",  df["is_ymyl"] == 0),
        ("Gambling", df["vertical"] == "gambling"),
    ]:
        sub = df[mask].dropna(subset=["path_length","position","authority_decile_v2","page_type",
                                       "keyword_difficulty","search_intent","path_depth","keyword_in_url"])
        m = smf.ols(formula, data=sub).fit(cov_type="HC3")
        out[label] = {
            "n": int(m.nobs),
            "r2": round(m.rsquared, 4),
            "path_length_beta": round(float(m.params["path_length"]), 6),
            "path_length_pval": round(float(m.pvalues["path_length"]), 6),
            "median_path_length": float(sub["path_length"].median()),
        }
    return out


def main():
    print("\n" + "=" * 70)
    print(f"PHASE 7: RE-ANALYSE MIT EXTERNEM AUTHORITY-SIGNAL  ({datetime.now():%H:%M})")
    print("=" * 70)

    df = load_and_merge()
    df.to_csv(DATA / "06_dataset_with_authority.csv", index=False)
    print(f"Merged dataset gespeichert: {DATA/'06_dataset_with_authority.csv'}")

    # 1. Korrelationen
    print("\n[1] Spearman-Korrelationen (vs. position)")
    corrs = compute_correlations(df)
    for k, v in corrs.items():
        print(f"   {v['label']:50s}  ρ={v['rho']:+.4f}  p={v['p']:.4g}  n={v['n']:,}")

    # 2. Stepwise OLS (8 Modelle wie zuvor, aber mit authority_decile_v2)
    print("\n[2] OLS-Modelle M1-M4 mit neuer Authority-Variable (HC3)")
    models = {}
    for label, f in [
        ("M1_naive",     "position ~ path_length"),
        ("M2_authority", "position ~ path_length + authority_decile_v2"),
        ("M3_pagetype",  "position ~ path_length + authority_decile_v2 + C(page_type)"),
        ("M4_full",      "position ~ path_length + authority_decile_v2 + C(page_type) + "
                         "path_depth + keyword_in_url + has_parameters + word_count_path + "
                         "keyword_difficulty + C(search_intent) + has_featured_snippet"),
    ]:
        res = run_ols(df, f, label)
        models[label] = res
        sig = "***" if res.get("path_length_pval", 1) < 0.001 else (
              "**"  if res.get("path_length_pval", 1) < 0.01  else (
              "*"   if res.get("path_length_pval", 1) < 0.05  else "n.s."))
        print(f"   {label:14s}  N={res['n']:>5}  R²={res['r2']:.4f}  "
              f"β(path_length)={res['path_length_beta']:+.6f}  p={res['path_length_pval']:.6f}  [{sig}]")

    # 3. Shrinkage
    b1 = models["M1_naive"]["path_length_beta"]
    b4 = models["M4_full"]["path_length_beta"]
    shrinkage_v2 = (b1 - b4) / b1 * 100
    print(f"\n[3] Shrinkage M1 → M4: {shrinkage_v2:.1f}%")

    # 4. YMYL-Analyse
    print("\n[4] YMYL-Segmente mit neuer Authority")
    ymyl = run_ymyl_segmented(df)
    for label, r in ymyl.items():
        sig = "***" if r["path_length_pval"] < 0.001 else (
              "**"  if r["path_length_pval"] < 0.01  else (
              "*"   if r["path_length_pval"] < 0.05  else "n.s."))
        print(f"   {label:10s}  N={r['n']:>5}  R²={r['r2']:.4f}  "
              f"β={r['path_length_beta']:+.6f}  p={r['path_length_pval']:.6f}  [{sig}]  "
              f"Median PL={r['median_path_length']:.0f}")

    # 5. seo-kreativ Benchmark
    sk = df[df["is_seo_kreativ"] == 1]
    sk_auth = sk["log_organic_kw"].dropna()
    print(f"\n[5] seo-kreativ.de Benchmark")
    print(f"   N={len(sk):,}  Median pos={sk['position'].median():.1f}  "
          f"Median path={sk['path_length'].median():.0f}  "
          f"log_organic_kw median={sk_auth.median() if len(sk_auth) else 'n/a':.2f}")

    # 6. Save all
    output = {
        "meta": {
            "timestamp": datetime.utcnow().isoformat(),
            "n_rows":     int(len(df)),
            "n_keywords": int(df["keyword"].nunique()),
            "n_domains_with_authority": int(df["organic_keywords_count"].notna().sum()),
            "authority_source": "DataForSEO domain_rank_overview (Germany/de)",
        },
        "correlations": corrs,
        "models": models,
        "shrinkage_M1_to_M4": round(shrinkage_v2, 2),
        "ymyl_segmented": ymyl,
        "methodology_changes": {
            "previous": "domain_rank_decile (was actually n_appearances-derived, in-sample frequency proxy, ρ=0.78 with n_appearances)",
            "current":  "authority_decile_v2 derived from external DataForSEO organic_keywords_count via log+decile binning",
            "why":      "n_appearances is circular (rank-frequency in our sample predicting rank). External signal is independent.",
        },
    }
    out_path = OUT / "model_results.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n[6] Gespeichert: {out_path}")


if __name__ == "__main__":
    main()
