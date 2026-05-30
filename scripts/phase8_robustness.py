"""
Phase 8: Robustness-Check der v2-Befunde

Tests:
  1. Verschiedene Authority-Metriken vergleichen
  2. VIF-Check fuer neues Modell
  3. Cross-Validation 80/20 (Train/Test)
  4. Bootstrap Confidence Intervals fuer Beta-Werte
  5. Sensitivitaet gegenueber Outliern (Trim 5% an beiden Enden)
  6. Spec-Curve: M4 mit verschiedenen Spezifikationen

Input:  data/06_dataset_with_authority.csv
Output: analysis_v2/robustness_v2.json
"""

import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

ROOT = Path("/home/christian/url-studie")
DATA = ROOT / "data"
OUT  = ROOT / "analysis_v2"

CONTROLS = ("C(page_type) + path_depth + keyword_in_url + has_parameters + "
            "word_count_path + keyword_difficulty + C(search_intent) + has_featured_snippet")


def load():
    df = pd.read_csv(DATA / "06_dataset_with_authority.csv", low_memory=False)
    for col in ["path_length","position","is_ymyl","path_depth","keyword_in_url",
                "has_parameters","word_count_path","has_featured_snippet","keyword_difficulty",
                "organic_keywords_count","organic_etv","organic_pos_1","organic_pos_2_3","organic_pos_4_10"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fit_path_beta(df, formula):
    """Fit OLS und liefere (beta, pval, n, r2)."""
    sub = df.dropna(subset=["path_length","position","page_type","keyword_difficulty","search_intent",
                            "path_depth","keyword_in_url","has_parameters","word_count_path","has_featured_snippet"])
    # Drop NaN fuer alle Variablen in der Formel
    try:
        m = smf.ols(formula, data=sub).fit(cov_type="HC3")
        if "path_length" not in m.params:
            return None
        return {
            "n": int(m.nobs),
            "r2": round(m.rsquared, 4),
            "beta": round(float(m.params["path_length"]), 6),
            "pval": round(float(m.pvalues["path_length"]), 6),
            "se":   round(float(m.bse["path_length"]), 6),
            "ci_low":  round(float(m.conf_int().loc["path_length", 0]), 6),
            "ci_high": round(float(m.conf_int().loc["path_length", 1]), 6),
        }
    except Exception as e:
        return {"error": str(e)}


def test_authority_specifications(df):
    """Vergleiche 7 verschiedene Authority-Metriken im selben Modell-Rahmen."""
    df = df.copy()
    df["log_organic_kw"]    = np.log1p(df["organic_keywords_count"])
    df["log_organic_etv"]   = np.log1p(df["organic_etv"])
    df["log_organic_top3"]  = np.log1p((df["organic_pos_1"].fillna(0) + df["organic_pos_2_3"].fillna(0)))
    df["sqrt_organic_kw"]   = np.sqrt(df["organic_keywords_count"].fillna(0))
    # Decile-Binning auf log(organic_kw)
    df["decile_log_kw"]     = pd.qcut(df["log_organic_kw"].fillna(-1), 10, labels=False, duplicates="drop") + 1

    out = {}
    for auth_name, auth_var in [
        ("none",                 None),
        ("organic_kw_raw",       "organic_keywords_count"),
        ("organic_etv_raw",      "organic_etv"),
        ("organic_top3_raw",     "organic_pos_1"),
        ("log_organic_kw",       "log_organic_kw"),
        ("log_organic_etv",      "log_organic_etv"),
        ("log_organic_top3",     "log_organic_top3"),
        ("sqrt_organic_kw",      "sqrt_organic_kw"),
        ("decile_log_kw",        "decile_log_kw"),
        ("OLD_domain_rank_decile (Vergleich)", "domain_rank_decile"),
    ]:
        if auth_var is None:
            formula = f"position ~ path_length + {CONTROLS}"
        else:
            formula = f"position ~ path_length + {auth_var} + {CONTROLS}"
        res = fit_path_beta(df, formula)
        out[auth_name] = res
    return out


def vif_check(df):
    """VIF mit der neuen Authority-Variable."""
    cols = ["path_length","log_organic_kw","path_depth","keyword_in_url","word_count_path",
            "has_featured_snippet","has_parameters","is_ymyl"]
    df_v = df.copy()
    df_v["log_organic_kw"] = np.log1p(df_v["organic_keywords_count"])
    sub = df_v[cols].dropna()
    sub = sub.assign(intercept=1)
    out = []
    for i, c in enumerate(cols):
        try:
            vif = variance_inflation_factor(sub[cols + ["intercept"]].values, i)
        except Exception:
            vif = float("nan")
        out.append({"feature": c, "VIF": round(vif, 2) if not np.isnan(vif) else None})
    return out


def cross_validate_M4(df, k=5):
    """K-fold Cross-Validation: Beta-Stabilitaet ueber Splits."""
    df = df.copy()
    df["log_organic_kw"] = np.log1p(df["organic_keywords_count"])
    sub = df.dropna(subset=["path_length","position","log_organic_kw","page_type","keyword_difficulty",
                            "search_intent","path_depth","keyword_in_url","has_parameters",
                            "word_count_path","has_featured_snippet"])
    formula = f"position ~ path_length + log_organic_kw + {CONTROLS}"
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    folds = []
    for i, (train_idx, test_idx) in enumerate(kf.split(sub)):
        train = sub.iloc[train_idx]
        test  = sub.iloc[test_idx]
        try:
            m = smf.ols(formula, data=train).fit(cov_type="HC3")
            pred = m.predict(test)
            mse  = float(((test["position"] - pred) ** 2).mean())
            folds.append({
                "fold": i+1,
                "n_train": len(train),
                "n_test": len(test),
                "path_beta": round(float(m.params["path_length"]), 6),
                "path_pval": round(float(m.pvalues["path_length"]), 6),
                "train_r2": round(float(m.rsquared), 4),
                "test_mse": round(mse, 4),
            })
        except Exception as e:
            folds.append({"fold": i+1, "error": str(e)})
    return folds


def bootstrap_ci_path_beta(df, n_boot=500, seed=42):
    """Bootstrap CIs fuer path_length-Koeffizient."""
    df = df.copy()
    df["log_organic_kw"] = np.log1p(df["organic_keywords_count"])
    sub = df.dropna(subset=["path_length","position","log_organic_kw","page_type","keyword_difficulty",
                            "search_intent","path_depth","keyword_in_url","has_parameters",
                            "word_count_path","has_featured_snippet"])
    formula = f"position ~ path_length + log_organic_kw + {CONTROLS}"

    rng = np.random.default_rng(seed)
    betas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(sub), len(sub))
        boot = sub.iloc[idx]
        try:
            m = smf.ols(formula, data=boot).fit()
            betas.append(float(m.params["path_length"]))
        except Exception:
            pass
    betas = np.array(betas)
    return {
        "n_bootstraps": int(len(betas)),
        "mean":   round(float(betas.mean()), 6),
        "median": round(float(np.median(betas)), 6),
        "ci_95_low":  round(float(np.percentile(betas, 2.5)), 6),
        "ci_95_high": round(float(np.percentile(betas, 97.5)), 6),
        "std":  round(float(betas.std()), 6),
    }


def outlier_sensitivity(df, trim_pct=5):
    """Trim Top/Bottom 5% von position bzw path_length, dann fitten."""
    df = df.copy()
    df["log_organic_kw"] = np.log1p(df["organic_keywords_count"])
    # Trim path_length outliers
    lo = df["path_length"].quantile(trim_pct/100)
    hi = df["path_length"].quantile(1 - trim_pct/100)
    trimmed = df[(df["path_length"]>=lo) & (df["path_length"]<=hi)]
    sub = trimmed.dropna(subset=["path_length","position","log_organic_kw","page_type","keyword_difficulty",
                                  "search_intent","path_depth","keyword_in_url","has_parameters",
                                  "word_count_path","has_featured_snippet"])
    formula = f"position ~ path_length + log_organic_kw + {CONTROLS}"
    m = smf.ols(formula, data=sub).fit(cov_type="HC3")
    return {
        "trim_pct_each_side": trim_pct,
        "n_kept": int(m.nobs),
        "n_dropped": int(len(df) - m.nobs),
        "path_beta": round(float(m.params["path_length"]), 6),
        "path_pval": round(float(m.pvalues["path_length"]), 6),
        "r2": round(float(m.rsquared), 4),
    }


def ymyl_robustness(df):
    """YMYL/NonYMYL/Gambling mit verschiedenen Authority-Spezifikationen."""
    df = df.copy()
    df["log_organic_kw"]  = np.log1p(df["organic_keywords_count"])
    df["log_organic_etv"] = np.log1p(df["organic_etv"])

    segments = {
        "YMYL":     df[df["is_ymyl"] == 1],
        "NonYMYL":  df[df["is_ymyl"] == 0],
        "Gambling": df[df["vertical"] == "gambling"],
    }
    auth_options = {
        "log_organic_kw":  "log_organic_kw",
        "log_organic_etv": "log_organic_etv",
        "no_authority":    None,
    }

    out = {}
    for seg_name, seg_df in segments.items():
        out[seg_name] = {}
        for auth_name, auth_var in auth_options.items():
            if auth_var:
                f = f"position ~ path_length + {auth_var} + {CONTROLS}"
            else:
                f = f"position ~ path_length + {CONTROLS}"
            out[seg_name][auth_name] = fit_path_beta(seg_df, f)
    return out


def main():
    print(f"\n{'='*70}\nPHASE 8: ROBUSTNESS-CHECK ({datetime.now():%H:%M})\n{'='*70}")
    df = load()
    print(f"N = {len(df):,} rows, {df['organic_keywords_count'].notna().sum():,} mit Authority\n")

    print("[1] AUTHORITY-SPEZIFIKATIONEN — Wie stabil ist path_length β?")
    print("-"*70)
    auth_test = test_authority_specifications(df)
    for k, r in auth_test.items():
        if r and "beta" in r:
            sig = "***" if r["pval"]<0.001 else ("**" if r["pval"]<0.01 else ("*" if r["pval"]<0.05 else "n.s."))
            print(f"  {k:42s}  N={r['n']:>5}  R²={r['r2']:.4f}  β={r['beta']:+.6f}  p={r['pval']:.4f}  [{sig}]  CI=[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
        else:
            print(f"  {k:42s}  ERROR: {r}")

    print("\n[2] VIF-CHECK (Multicollinearity)")
    print("-"*70)
    vif = vif_check(df)
    for v in vif:
        mark = " <-- HOCH" if v["VIF"] and v["VIF"] > 10 else ""
        print(f"  {v['feature']:25s}  VIF = {v['VIF']}{mark}")

    print("\n[3] CROSS-VALIDATION (5-fold)")
    print("-"*70)
    cv = cross_validate_M4(df)
    for f in cv:
        if "error" not in f:
            print(f"  Fold {f['fold']}  β={f['path_beta']:+.6f}  p={f['path_pval']:.4f}  train_R²={f['train_r2']:.4f}  test_MSE={f['test_mse']:.4f}")
    if cv and all("path_beta" in f for f in cv):
        betas = [f["path_beta"] for f in cv]
        print(f"  Stabilitaet: mean={np.mean(betas):+.6f}  std={np.std(betas):.6f}  range=[{min(betas):+.6f}, {max(betas):+.6f}]")

    print("\n[4] BOOTSTRAP 95% CI (500 Resamples)")
    print("-"*70)
    boot = bootstrap_ci_path_beta(df, n_boot=500)
    print(f"  Bootstrap β mean   : {boot['mean']:+.6f}")
    print(f"  Bootstrap β median : {boot['median']:+.6f}")
    print(f"  Bootstrap 95% CI   : [{boot['ci_95_low']:+.6f}, {boot['ci_95_high']:+.6f}]")
    print(f"  Bootstrap SD       : {boot['std']:.6f}")

    print("\n[5] OUTLIER-SENSITIVITAET (Trim 5% jede Seite)")
    print("-"*70)
    out_sens = outlier_sensitivity(df, 5)
    print(f"  Original N: {len(df):,}, getrimmt N: {out_sens['n_kept']:,} (dropped {out_sens['n_dropped']:,})")
    print(f"  Getrimmtes β: {out_sens['path_beta']:+.6f}  p={out_sens['path_pval']:.4f}  R²={out_sens['r2']:.4f}")

    print("\n[6] YMYL-ROBUSTNESS (3 Authority-Spezifikationen × 3 Segmente)")
    print("-"*70)
    ymyl = ymyl_robustness(df)
    for seg, results in ymyl.items():
        print(f"\n  Segment: {seg}")
        for auth_name, r in results.items():
            if r and "beta" in r:
                sig = "***" if r["pval"]<0.001 else ("**" if r["pval"]<0.01 else ("*" if r["pval"]<0.05 else "n.s."))
                print(f"    {auth_name:18s}  N={r['n']:>5}  β={r['beta']:+.6f}  p={r['pval']:.4f}  [{sig}]")

    # Save
    output = {
        "meta": {"timestamp": datetime.utcnow().isoformat()},
        "authority_specifications": auth_test,
        "vif": vif,
        "cross_validation_5fold": cv,
        "bootstrap_95ci": boot,
        "outlier_sensitivity_5pct": out_sens,
        "ymyl_robustness": ymyl,
    }
    (OUT / "robustness_v2.json").write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n[7] Gespeichert: {OUT/'robustness_v2.json'}")


if __name__ == "__main__":
    main()
