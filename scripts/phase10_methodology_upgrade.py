"""
Phase 10: Publishable-Grade Methodology Upgrade

Antwort auf Perplexity-Review der v2-Studie. Setzt 4 methodische Verbesserungen um:

  1. Cluster-Robuste SE auf Keyword-Level (statt nur HC3)
     → Korrigiert fuer Non-Independence innerhalb einer SERP
  2. Ordered Logit als Vergleichsmodell
     → Position 1-10 ist ordinal, nicht kontinuierlich
  3. Cubic Splines fuer path_length (statt lineare Annahme)
     → Erlaubt nicht-monotone Effekte
  4. Multikollinearitaets-Fix: Modell ohne word_count_path
     → Eindeutige Pfadlaengen-Interpretation (war VIF=16.35)

Input:  data/06_dataset_with_authority.csv
Output: analysis_v2/methodology_upgrade.json
        charts_v2/12_methodology_comparison.png
"""

import json, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

ROOT = Path("/home/christian/url-studie")
DATA = ROOT / "data"
OUT  = ROOT / "analysis_v2"
CHARTS = ROOT / "charts_v2"

# Branding
BG = "#0f1117"; BG2 = "#1a1d2e"; ACCENT = "#6c63ff"; ACCENT2 = "#ff6584"
GREEN = "#43d9ad"; YELLOW = "#ffd166"; TEXT = "#e2e8f0"; TEXT_DIM = "#94a3b8"; GRID = "#2d3148"


def load_data():
    df = pd.read_csv(DATA / "06_dataset_with_authority.csv", low_memory=False)
    for col in ["path_length","position","is_ymyl","path_depth","keyword_in_url",
                "has_parameters","word_count_path","has_featured_snippet","keyword_difficulty",
                "organic_keywords_count","organic_etv"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["log_organic_kw"] = np.log1p(df["organic_keywords_count"])
    return df


# ─── 1. Clustered SE auf Keyword-Level ──────────────────────────────────────────
def model_clustered_se(df, formula):
    """OLS mit Cluster-Robuster SE auf keyword-Level."""
    sub = df.dropna(subset=["path_length","position","log_organic_kw","page_type","keyword_difficulty",
                            "search_intent","path_depth","keyword_in_url","has_parameters",
                            "word_count_path","has_featured_snippet"])
    # cov_type='cluster' mit groups
    m = smf.ols(formula, data=sub).fit(
        cov_type="cluster",
        cov_kwds={"groups": sub["keyword"]}
    )
    return {
        "n": int(m.nobs),
        "n_clusters": int(sub["keyword"].nunique()),
        "r2": round(m.rsquared, 4),
        "path_length_beta": round(float(m.params["path_length"]), 6),
        "path_length_se":   round(float(m.bse["path_length"]), 6),
        "path_length_pval": round(float(m.pvalues["path_length"]), 6),
        "ci_low":  round(float(m.conf_int().loc["path_length", 0]), 6),
        "ci_high": round(float(m.conf_int().loc["path_length", 1]), 6),
    }


# ─── 2. Ordered Logit ───────────────────────────────────────────────────────────
def model_ordered_logit(df):
    """
    Ordered Logit: Position als ordinale Variable.
    Wir buckeln Position in 4 Bins fuer Stabilitaet:
      1-2 (Top), 3-5 (High), 6-8 (Mid), 9-10 (Low)
    """
    sub = df.dropna(subset=["path_length","position","log_organic_kw","path_depth",
                            "keyword_in_url","has_parameters","word_count_path",
                            "keyword_difficulty","has_featured_snippet"]).copy()

    # Bucke Position in 4 Klassen
    def bucket(p):
        if p <= 2: return 0  # Top
        if p <= 5: return 1  # High
        if p <= 8: return 2  # Mid
        return 3              # Low
    sub["pos_bucket"] = sub["position"].apply(bucket)

    # Endogene Features (Designmatrix muss kontinuierlich sein, daher Dummies manuell)
    X = sub[["path_length", "log_organic_kw", "path_depth", "keyword_in_url",
             "has_parameters", "word_count_path", "keyword_difficulty",
             "has_featured_snippet"]].copy()

    # Search intent als Dummies (one-hot, ohne Referenz)
    intent_dummies = pd.get_dummies(sub["search_intent"], prefix="intent", drop_first=True)
    page_dummies   = pd.get_dummies(sub["page_type"],     prefix="ptype",  drop_first=True)
    X = pd.concat([X, intent_dummies, page_dummies], axis=1)
    X = X.astype(float)

    try:
        model = OrderedModel(sub["pos_bucket"], X, distr="logit")
        # method='lbfgs' ist stabiler fuer grosse N
        result = model.fit(method="lbfgs", maxiter=500, disp=False)
        # Falls SE NaN: nochmal mit numerischen Hessian
        if np.isnan(float(result.bse["path_length"])):
            result = model.fit(method="bfgs", maxiter=500, disp=False, retall=False)

        # path_length Effekt
        beta = float(result.params["path_length"])
        se   = float(result.bse["path_length"])
        pval = float(result.pvalues["path_length"])

        return {
            "method": "OrderedLogit (4 buckets: top/high/mid/low)",
            "n": int(len(sub)),
            "path_length_logit_beta": round(beta, 6),
            "path_length_logit_se":   round(se, 6),
            "path_length_logit_pval": round(pval, 6),
            "odds_ratio_per_char": round(float(np.exp(beta)), 6),
            "interpretation": (
                "OR = " + f"{np.exp(beta):.4f}"
                + " → pro zusaetzlichem URL-Zeichen aendern sich die Odds, in eine schlechtere Position-Klasse zu rutschen, um diesen Faktor"
            ),
            "convergence": str(result.mle_retvals.get("converged", "n/a")),
        }
    except Exception as e:
        return {"error": str(e), "method": "OrderedLogit"}


# ─── 3. Cubic Spline fuer path_length ──────────────────────────────────────────
def model_with_spline(df, n_knots=3):
    """
    Patsy bs(spline) fuer path_length, alle anderen Confounder linear.
    Erlaubt nicht-monotone Effekte. 3 Knoten = glatter Verlauf, weniger Overfitting.
    """
    sub = df.dropna(subset=["path_length","position","log_organic_kw","page_type","keyword_difficulty",
                            "search_intent","path_depth","keyword_in_url","has_parameters",
                            "word_count_path","has_featured_snippet"]).copy()

    # Knot-Positionen auf Quantilen (25%, 50%, 75%)
    knots = np.quantile(sub["path_length"], [0.25, 0.5, 0.75])
    knots_str = ",".join(f"{k:.0f}" for k in knots)

    formula = (
        f"position ~ bs(path_length, knots=({knots_str}), degree=3) + "
        f"log_organic_kw + C(page_type) + path_depth + keyword_in_url + has_parameters + "
        f"word_count_path + keyword_difficulty + C(search_intent) + has_featured_snippet"
    )

    m = smf.ols(formula, data=sub).fit(
        cov_type="cluster",
        cov_kwds={"groups": sub["keyword"]}
    )

    # Joint F-Test: kombiniere alle bs(...)[i] Terms
    spline_terms = [p for p in m.params.index if "bs(path_length" in p]
    f_stat = None
    f_pval = None
    if spline_terms:
        try:
            R = np.eye(len(m.params))
            indices = [list(m.params.index).index(t) for t in spline_terms]
            R_subset = R[indices, :]
            ftest = m.f_test(R_subset)
            f_stat = float(ftest.statistic)
            f_pval = float(ftest.pvalue)
        except Exception as e:
            f_stat = None
            f_pval = None

    # Spline-Praedictions fuer Visualisierung (zwischen 5%- und 95%-Quantil)
    pl_range = np.linspace(sub["path_length"].quantile(0.05), sub["path_length"].quantile(0.95), 100)
    pred_df = pd.DataFrame({
        "path_length": pl_range,
        "log_organic_kw": sub["log_organic_kw"].median(),
        "page_type": sub["page_type"].mode()[0],
        "path_depth": sub["path_depth"].median(),
        "keyword_in_url": sub["keyword_in_url"].median(),
        "has_parameters": 0,
        "word_count_path": sub["word_count_path"].median(),
        "keyword_difficulty": sub["keyword_difficulty"].median(),
        "search_intent": sub["search_intent"].mode()[0],
        "has_featured_snippet": 0,
    })
    pred = m.predict(pred_df)

    return {
        "method": f"Cubic Spline (degree=3, {n_knots} knots at quantiles)",
        "n": int(m.nobs),
        "n_clusters": int(sub["keyword"].nunique()),
        "r2": round(m.rsquared, 4),
        "f_stat_spline_joint": round(f_stat, 4) if f_stat is not None else None,
        "f_pval_spline_joint": round(f_pval, 6) if f_pval is not None else None,
        "predicted_curve_path_length": pl_range.tolist(),
        "predicted_curve_position":    pred.tolist(),
        "knots": knots.tolist(),
    }


# ─── 4. Multikollinearitaets-Fix: ohne word_count_path ──────────────────────────
def model_no_wordcount(df):
    """M4 ohne word_count_path, um path_length sauber zu isolieren."""
    formula = (
        "position ~ path_length + log_organic_kw + C(page_type) + path_depth + "
        "keyword_in_url + has_parameters + keyword_difficulty + C(search_intent) + "
        "has_featured_snippet"
    )
    return model_clustered_se(df, formula)


# ─── HAUPT ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*70}\nPHASE 10: PUBLISHABLE-GRADE UPGRADE  ({datetime.now():%H:%M})\n{'='*70}\n")
    df = load_data()
    print(f"Loaded: {len(df):,} rows, {df['keyword'].nunique():,} clusters\n")

    # 1. Clustered SE (HC3 → Cluster)
    print("[1] OLS mit CLUSTERED SE auf Keyword-Level")
    print("-"*70)
    formula_full = (
        "position ~ path_length + log_organic_kw + C(page_type) + path_depth + "
        "keyword_in_url + has_parameters + word_count_path + keyword_difficulty + "
        "C(search_intent) + has_featured_snippet"
    )
    clustered = model_clustered_se(df, formula_full)
    print(f"  N={clustered['n']:,}  Clusters={clustered['n_clusters']:,}")
    print(f"  β(path_length) = {clustered['path_length_beta']:+.6f}")
    print(f"  SE (clustered) = {clustered['path_length_se']:.6f}")
    print(f"  p-Wert         = {clustered['path_length_pval']:.6f}")
    print(f"  95% CI         = [{clustered['ci_low']:+.6f}, {clustered['ci_high']:+.6f}]")
    sig = "***" if clustered['path_length_pval']<0.001 else ("**" if clustered['path_length_pval']<0.01 else ("*" if clustered['path_length_pval']<0.05 else "n.s."))
    print(f"  Signifikanz    = [{sig}]")
    print(f"  Vergleich zu HC3 (v2): β=+0.0036, SE=0.0017, p=0.036, CI=[+0.0002, +0.0070]")

    # 2. Ordered Logit
    print("\n[2] ORDERED LOGIT (Position als ordinale Variable)")
    print("-"*70)
    ologit = model_ordered_logit(df)
    if "error" not in ologit:
        print(f"  Methode:      {ologit['method']}")
        print(f"  N:            {ologit['n']:,}")
        print(f"  β (logit):    {ologit['path_length_logit_beta']:+.6f}")
        print(f"  SE:           {ologit['path_length_logit_se']:.6f}")
        print(f"  p-Wert:       {ologit['path_length_logit_pval']:.6f}")
        print(f"  Odds Ratio:   {ologit['odds_ratio_per_char']:.4f} pro Zeichen")
        print(f"  Konvergenz:   {ologit['convergence']}")
        print(f"  Interpretation: {ologit['interpretation']}")
    else:
        print(f"  ERROR: {ologit['error']}")

    # 3. Cubic Spline
    print("\n[3] CUBIC SPLINE fuer path_length (erlaubt nicht-monotone Effekte)")
    print("-"*70)
    spline = model_with_spline(df)
    print(f"  Methode:      {spline['method']}")
    print(f"  N:            {spline['n']:,}")
    print(f"  R²:           {spline['r2']:.4f}")
    print(f"  Knots:        {[round(k,1) for k in spline['knots']]}")
    print(f"  F-Test joint significance der Spline-Terms: siehe JSON")

    # 4. Ohne word_count_path
    print("\n[4] MULTIKOLLINEARITAETS-FIX: Modell ohne word_count_path")
    print("-"*70)
    no_wc = model_no_wordcount(df)
    print(f"  N={no_wc['n']:,}  Clusters={no_wc['n_clusters']:,}")
    print(f"  β(path_length) = {no_wc['path_length_beta']:+.6f}")
    print(f"  SE (clustered) = {no_wc['path_length_se']:.6f}")
    print(f"  p-Wert         = {no_wc['path_length_pval']:.6f}")
    print(f"  95% CI         = [{no_wc['ci_low']:+.6f}, {no_wc['ci_high']:+.6f}]")
    sig2 = "***" if no_wc['path_length_pval']<0.001 else ("**" if no_wc['path_length_pval']<0.01 else ("*" if no_wc['path_length_pval']<0.05 else "n.s."))
    print(f"  Signifikanz    = [{sig2}]")

    # Save
    output = {
        "meta": {
            "timestamp": datetime.utcnow().isoformat(),
            "purpose": "Publishable-grade methodology upgrade in response to Perplexity review",
        },
        "1_clustered_se_keyword_level": clustered,
        "2_ordered_logit": ologit,
        "3_cubic_spline": spline,
        "4_no_wordcount_multicollinearity_fix": no_wc,
        "comparison_v2_full_model_hc3": {
            "method": "OLS HC3 (v2)",
            "n": 22250,
            "path_length_beta": 0.003612,
            "path_length_se":   0.001725,
            "path_length_pval": 0.0356,
            "ci_low":  0.000231,
            "ci_high": 0.006994,
        },
    }
    (OUT / "methodology_upgrade.json").write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n[5] Gespeichert: {OUT/'methodology_upgrade.json'}")

    # ─── Chart 12: Methodology Comparison ──────────────────────────────────────
    print("\n[6] Rendere Chart 12: Methodik-Vergleich")
    plt.rcParams.update({
        'figure.facecolor': BG, 'axes.facecolor': BG2, 'axes.edgecolor': GRID,
        'axes.labelcolor': TEXT, 'axes.titlecolor': TEXT, 'xtick.color': TEXT_DIM,
        'ytick.color': TEXT_DIM, 'text.color': TEXT, 'grid.color': GRID,
        'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 13,
    })

    fig, axes = plt.subplots(1, 2, figsize=(17, 7), facecolor=BG)
    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.07, right=0.97, wspace=0.30)

    # Panel A: Method comparison (β + CI)
    ax = axes[0]
    methods = [
        ("v2: OLS HC3",                       0.003612, 0.000231, 0.006994, ACCENT2),
        ("Clustered SE\n(Keyword-Level)",     clustered["path_length_beta"], clustered["ci_low"], clustered["ci_high"], ACCENT),
        ("Ohne word_count_path\n(Multikoll.-Fix)", no_wc["path_length_beta"], no_wc["ci_low"], no_wc["ci_high"], GREEN),
    ]
    y = np.arange(len(methods))
    for i, (lab, b, lo, hi, col) in enumerate(methods):
        ax.plot([lo, hi], [i, i], color=col, linewidth=2.5)
        ax.plot([lo, lo], [i-0.2, i+0.2], color=col, linewidth=1.8)
        ax.plot([hi, hi], [i-0.2, i+0.2], color=col, linewidth=1.8)
        ax.scatter([b], [i], color=col, s=140, edgecolor=TEXT, linewidth=1.5, zorder=3)
        ax.text(hi + 0.0008, i, f'β={b:+.4f}', va='center', fontsize=10, color=col, fontweight='bold')

    ax.axvline(0, color=ACCENT2, linestyle='--', linewidth=1, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([m[0] for m in methods], fontsize=10)
    ax.set_xlabel("path_length β mit 95% CI")
    ax.set_title("Methodik-Robustheit: OLS-Varianten im Vergleich", pad=10)
    ax.grid(axis='x', alpha=0.4)

    # Panel B: Cubic Spline Vorhersage
    ax2 = axes[1]
    spline_pl = spline["predicted_curve_path_length"]
    spline_pos = spline["predicted_curve_position"]

    ax2.plot(spline_pl, spline_pos, color=ACCENT, linewidth=3, label="Cubic Spline (3 Knoten)")

    # Knots als vertikale Linien mit Labels
    for k in spline["knots"]:
        ax2.axvline(k, color=TEXT_DIM, linestyle=':', linewidth=0.8, alpha=0.6)
        ax2.text(k, max(spline_pos)*0.99, f'Q{int(k)}', ha='center', fontsize=8, color=TEXT_DIM,
                 alpha=0.7, rotation=0)

    # Lineare Referenz mittig durch die Spline-Mitte
    mid_pl  = (spline_pl[0] + spline_pl[-1]) / 2
    mid_pos = (spline_pos[0] + spline_pos[-1]) / 2
    lin_y = [mid_pos + 0.0050 * (pl - mid_pl) for pl in spline_pl]
    ax2.plot(spline_pl, lin_y, color=ACCENT2, linewidth=2,
             linestyle='--', alpha=0.8, label="Lineare Annahme (β=+0,0050)")

    ax2.set_xlabel("URL-Pfadlänge (Zeichen)")
    ax2.set_ylabel("Vorhergesagte Position (alle anderen Variablen am Median)")
    ax2.set_title("Cubic Spline vs. lineare Annahme", pad=10)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(alpha=0.4)
    ax2.invert_yaxis()  # niedrige Position oben

    fig.suptitle("Methodik-Upgrade (Phase 10): Clustered SE + Ordered Logit + Splines",
                 fontsize=14, fontweight='bold', y=0.97)
    fig.text(0.5, 0.04,
             f'N = {clustered["n"]:,} Top-10 SERP-Ergebnisse, {clustered["n_clusters"]:,} Keyword-Cluster\n'
             f'Cluster-robuste SE auf Keyword-Level (Antwort auf Perplexity-Review)',
             ha='center', va='center', fontsize=9, color=TEXT_DIM)

    fig.savefig(CHARTS / "12_methodology_comparison.png", format='png', bbox_inches='tight', dpi=300)
    fig.savefig(CHARTS / "12_methodology_comparison.svg", format='svg', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Gespeichert: charts_v2/12_methodology_comparison.png")


if __name__ == "__main__":
    main()
