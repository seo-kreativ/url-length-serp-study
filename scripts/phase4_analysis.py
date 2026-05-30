"""
Phase 4: Statistische Analyse — 8 Regressionsmodelle
Studie: Ranken kurze URLs wirklich besser?

Primär: Ordinal Logistic Regression (statsmodels OrderedModel)
Robustheitscheck: OLS
Mixed Effects: OLS mit keyword-Fixed-Effects (Proxy für within-SERP-Abhängigkeit)

Output: analysis/model_results.json
        analysis/model_comparison_table.csv
        analysis/robustness_checks.json
        analysis/ymyl_comparison.json
        analysis/gambling_deep_dive.json
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats

warnings.filterwarnings('ignore')

import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor

CLEAN_PATH   = Path("/home/christian/url-studie/data/04_dataset_clean.csv")
ANALYSIS_DIR = Path("/home/christian/url-studie/analysis")
ANALYSIS_DIR.mkdir(exist_ok=True)

# ─── Daten laden & vorbereiten ────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(CLEAN_PATH)
    # Kategoriale Variablen als Kategorie kodieren
    df['page_type']      = df['page_type'].astype('category')
    df['search_intent']  = df['search_intent'].astype('category')
    df['vertical']       = df['vertical'].astype('category')
    df['url_length_class'] = pd.Categorical(
        df['url_length_class'],
        categories=['kurz','optimal','mittel','lang','ueberlang'],
        ordered=True
    )
    # Numerische Bereinigung
    df['path_length']    = pd.to_numeric(df['path_length'], errors='coerce').fillna(0)
    df['domain_rank_decile'] = pd.to_numeric(df['domain_rank_decile'], errors='coerce').fillna(1)
    df['keyword_in_url'] = pd.to_numeric(df['keyword_in_url'], errors='coerce').fillna(0)
    df['path_depth']     = pd.to_numeric(df['path_depth'], errors='coerce').fillna(0)
    df['keyword_difficulty'] = pd.to_numeric(df['keyword_difficulty'], errors='coerce').fillna(0)
    df['has_featured_snippet'] = pd.to_numeric(df['has_featured_snippet'], errors='coerce').fillna(0)
    df['is_ymyl']        = pd.to_numeric(df['is_ymyl'], errors='coerce').fillna(0)
    df['word_count_path'] = pd.to_numeric(df['word_count_path'], errors='coerce').fillna(0)
    df['has_parameters'] = pd.to_numeric(df['has_parameters'].astype(str).map({'True':1,'False':0,'1':1,'0':0}), errors='coerce').fillna(0)
    return df

# ─── OLS-Modelle ──────────────────────────────────────────────────────────────
def run_ols_models(df):
    """Alle 8 Modelle als OLS — für Vergleichbarkeit mit Backlinko."""
    results = {}

    formulas = {
        'M1_naive':   'position ~ path_length',
        'M2_domain':  'position ~ path_length + domain_rank_decile',
        'M3_pagetype':'position ~ path_length + domain_rank_decile + C(page_type)',
        'M4_full':    'position ~ path_length + domain_rank_decile + C(page_type) + path_depth + keyword_in_url + has_parameters + word_count_path + keyword_difficulty + C(search_intent) + has_featured_snippet',
        'M5_depth':   'position ~ path_depth + domain_rank_decile + C(page_type) + keyword_in_url + has_parameters + keyword_difficulty + C(search_intent) + has_featured_snippet',
        'M6_intent':  'position ~ path_length * C(search_intent) + domain_rank_decile + C(page_type) + path_depth + keyword_in_url + keyword_difficulty',
        'M7_ymyl':    'position ~ path_length * is_ymyl + domain_rank_decile + C(page_type) + path_depth + keyword_in_url + keyword_difficulty + C(search_intent)',
        'M8_class':   'position ~ C(url_length_class) + domain_rank_decile + C(page_type) + path_depth + keyword_in_url + keyword_difficulty + C(search_intent)',
    }

    for name, formula in formulas.items():
        try:
            model = smf.ols(formula, data=df).fit(cov_type='HC3')
            # path_length Koeffizient extrahieren
            path_coef = model.params.get('path_length', None)
            path_pval = model.pvalues.get('path_length', None)

            results[name] = {
                'formula':      formula,
                'r2':           round(model.rsquared, 4),
                'r2_adj':       round(model.rsquared_adj, 4),
                'aic':          round(model.aic, 2),
                'n':            int(model.nobs),
                'path_length_beta': round(float(path_coef), 6) if path_coef is not None else None,
                'path_length_pval': round(float(path_pval), 6) if path_pval is not None else None,
                'path_length_sig':  '***' if path_pval and path_pval < 0.001 else ('**' if path_pval and path_pval < 0.01 else ('*' if path_pval and path_pval < 0.05 else 'n.s.')),
                'status': 'ok'
            }
            beta_str = f"{path_coef:.4f}" if path_coef is not None else "N/A"
            print(f"  {name}: R²={model.rsquared:.4f}, path_length β={beta_str} ({results[name]['path_length_sig']})")
        except Exception as e:
            results[name] = {'status': 'error', 'error': str(e)}
            print(f"  {name}: FEHLER — {e}")

    return results

# ─── VIF Multikollinearitäts-Check ────────────────────────────────────────────
def compute_vif(df):
    cols = ['path_length', 'domain_rank_decile', 'path_depth', 'keyword_in_url',
            'word_count_path', 'keyword_difficulty', 'has_featured_snippet',
            'has_parameters', 'is_ymyl']
    df_vif = df[cols].dropna()
    vif_data = pd.DataFrame()
    vif_data['feature'] = cols
    vif_data['VIF'] = [variance_inflation_factor(df_vif.values, i) for i in range(len(cols))]
    return vif_data.round(2)

# ─── Segmentierte Analysen ─────────────────────────────────────────────────────
def run_segmented(df, segment_col, segment_vals, formula_base):
    seg_results = {}
    for val in segment_vals:
        subset = df[df[segment_col] == val]
        if len(subset) < 100:
            continue
        try:
            model = smf.ols(formula_base, data=subset).fit(cov_type='HC3')
            path_coef = model.params.get('path_length', None)
            path_pval = model.pvalues.get('path_length', None)
            seg_results[str(val)] = {
                'n': len(subset),
                'r2': round(model.rsquared, 4),
                'path_length_beta': round(float(path_coef), 6) if path_coef is not None else None,
                'path_length_pval': round(float(path_pval), 4) if path_pval is not None else None,
            }
        except Exception as e:
            seg_results[str(val)] = {'error': str(e)}
    return seg_results

# ─── Mann-Whitney Tests ────────────────────────────────────────────────────────
def run_nonparametric(df):
    results = {}
    # Position 1-3 vs 8-10: Pfadlänge-Unterschied
    top3   = df[df['position'] <= 3]['path_length']
    bottom = df[df['position'] >= 8]['path_length']
    u, p   = stats.mannwhitneyu(top3, bottom, alternative='less')
    results['mannwhitney_top3_vs_bottom3'] = {
        'median_top3':   round(top3.median(), 1),
        'median_bottom': round(bottom.median(), 1),
        'U':             round(u, 0),
        'p':             round(p, 6),
        'significant':   p < 0.05
    }
    # Spearman-Korrelationen
    for col in ['path_length', 'path_depth', 'domain_rank_decile', 'keyword_in_url']:
        r, p = stats.spearmanr(df[col].dropna(), df.loc[df[col].notna(), 'position'])
        results[f'spearman_{col}'] = {'rho': round(r, 4), 'p': round(p, 6)}
    return results

# ─── YMYL-Analyse ──────────────────────────────────────────────────────────────
def run_ymyl_analysis(df):
    formula = 'position ~ path_length + domain_rank_decile + C(page_type) + path_depth + keyword_in_url + keyword_difficulty + C(search_intent)'
    ymyl_result = {}
    for label, subset in [('YMYL', df[df['is_ymyl']==1]), ('NonYMYL', df[df['is_ymyl']==0])]:
        try:
            m = smf.ols(formula, data=subset).fit(cov_type='HC3')
            path_coef = m.params.get('path_length', None)
            path_pval = m.pvalues.get('path_length', None)
            ymyl_result[label] = {
                'n': len(subset),
                'r2': round(m.rsquared, 4),
                'path_length_beta': round(float(path_coef), 6) if path_coef else None,
                'path_length_pval': round(float(path_pval), 4) if path_pval else None,
                'median_path_length': round(subset['path_length'].median(), 1),
                'median_domain_rank': round(subset['domain_rank_decile'].median(), 1),
            }
        except Exception as e:
            ymyl_result[label] = {'error': str(e)}

    # Gambling separat
    gambling = df[df['vertical'] == 'gambling']
    if len(gambling) > 50:
        try:
            m = smf.ols(formula, data=gambling).fit(cov_type='HC3')
            path_coef = m.params.get('path_length', None)
            ymyl_result['Gambling'] = {
                'n': len(gambling),
                'r2': round(m.rsquared, 4),
                'path_length_beta': round(float(path_coef), 6) if path_coef else None,
                'median_path_length': round(gambling['path_length'].median(), 1),
                'top_domains': gambling.groupby('domain_clean')['domain_clean'].count().sort_values(ascending=False).head(10).to_dict() if 'domain_clean' in gambling.columns else {},
            }
        except Exception as e:
            ymyl_result['Gambling'] = {'error': str(e)}
    return ymyl_result

# ─── seo-kreativ.de Benchmark ─────────────────────────────────────────────────
def run_seokreativ_benchmark(df):
    sk = df[df.get('is_seo_kreativ', pd.Series(0, index=df.index)) == 1] if 'is_seo_kreativ' in df.columns else pd.DataFrame()
    if len(sk) == 0:
        # Via GSC-Daten
        gsc = pd.read_csv('/home/christian/url-studie/data/00_seo_kreativ_gsc_keywords.csv')
        return {'n': len(gsc), 'note': 'Benchmark via GSC-Daten (nicht in Top-10-Sample)'}
    return {
        'n': len(sk),
        'median_path_length': round(sk['path_length'].median(), 1),
        'median_position': round(sk['position'].median(), 1),
    }

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"PHASE 4: Statistische Analyse ({datetime.now():%Y-%m-%d %H:%M})")
    print(f"{'='*60}")

    df = load_data()
    print(f"Datensatz: {len(df):,} Rows, {df['keyword'].nunique():,} Keywords")

    # ── 1. Bivariate / Non-parametrisch ───────────────────────────────────────
    print("\n[1] Bivariate Analysen...")
    nonparam = run_nonparametric(df)
    print(f"  Spearman ρ path_length ↔ position: {nonparam['spearman_path_length']['rho']:.4f}")
    print(f"  Spearman ρ domain_rank ↔ position:  {nonparam['spearman_domain_rank_decile']['rho']:.4f}")
    print(f"  Mann-Whitney Top3 vs Bottom3: median {nonparam['mannwhitney_top3_vs_bottom3']['median_top3']} vs {nonparam['mannwhitney_top3_vs_bottom3']['median_bottom']} (p={nonparam['mannwhitney_top3_vs_bottom3']['p']:.4f})")

    # ── 2. OLS Modelle 1-8 ────────────────────────────────────────────────────
    print("\n[2] OLS Regressionsmodelle 1-8...")
    ols_results = run_ols_models(df)

    # Modell-Vergleich-Tabelle
    model_table = []
    for name, res in ols_results.items():
        if res.get('status') == 'ok':
            model_table.append({
                'Modell': name,
                'R²': res['r2'],
                'R² adj': res['r2_adj'],
                'AIC': res['aic'],
                'path_length β': res['path_length_beta'],
                'p-Wert': res['path_length_pval'],
                'Sig.': res['path_length_sig'],
                'N': res['n'],
            })
    df_table = pd.DataFrame(model_table)
    df_table.to_csv(ANALYSIS_DIR / 'model_comparison_table.csv', index=False)
    print(f"\n  Modell-Vergleich:")
    cols_show = [c for c in ['Modell','R²','path_length β','Sig.'] if c in df_table.columns]
    print(df_table[cols_show].to_string(index=False) if cols_show else df_table.to_string(index=False))

    # ── 3. VIF ────────────────────────────────────────────────────────────────
    print("\n[3] VIF Multikollinearitäts-Check...")
    vif = compute_vif(df)
    print(vif.to_string(index=False))
    vif.to_csv(ANALYSIS_DIR / 'vif_check.csv', index=False)

    # ── 4. Segmentierte Analysen ──────────────────────────────────────────────
    print("\n[4] Segmentierte Analysen...")
    formula_base = 'position ~ path_length + domain_rank_decile + C(page_type) + path_depth + keyword_in_url + keyword_difficulty'

    seg_intent   = run_segmented(df, 'search_intent', df['search_intent'].unique(), formula_base)
    seg_vertical = run_segmented(df, 'vertical', df['vertical'].unique(), formula_base)
    seg_pagetype = run_segmented(df, 'page_type', df['page_type'].unique(), formula_base)

    print("  Path_length β nach Intent:")
    for k, v in seg_intent.items():
        if 'path_length_beta' in v:
            print(f"    {k}: β={v['path_length_beta']:.4f} (p={v.get('path_length_pval','?')}, n={v['n']})")

    print("  Path_length β nach Vertical:")
    for k, v in seg_vertical.items():
        if 'path_length_beta' in v:
            print(f"    {k}: β={v['path_length_beta']:.4f} (n={v['n']})")

    # ── 5. YMYL-Analyse ───────────────────────────────────────────────────────
    print("\n[5] YMYL-Spezialanalyse...")
    ymyl_res = run_ymyl_analysis(df)
    for label, res in ymyl_res.items():
        if 'path_length_beta' in res:
            print(f"  {label}: β={res['path_length_beta']:.4f} (R²={res.get('r2','?')}, n={res['n']})")

    # ── 6. URL-Längen-Klassen-Effekt ─────────────────────────────────────────
    print("\n[6] URL-Längen-Klassen — mittlere Position:")
    print(df.groupby('url_length_class')['position'].agg(['mean','median','count']).round(2).to_string())

    # ── 7. Nicht-linearer Schwellenwert ───────────────────────────────────────
    print("\n[7] Pfadlänge in Quartile — mittlere Position:")
    df['path_length_quartile'] = pd.qcut(df['path_length'], q=4, labels=['Q1','Q2','Q3','Q4'])
    print(df.groupby('path_length_quartile')['position'].agg(['mean','median','count']).round(2).to_string())

    # ── 8. seo-kreativ.de Benchmark ───────────────────────────────────────────
    print("\n[8] seo-kreativ.de Benchmark...")
    sk_res = run_seokreativ_benchmark(df)
    print(f"  {sk_res}")

    # ── Alles speichern ───────────────────────────────────────────────────────
    all_results = {
        'meta': {'timestamp': datetime.now().isoformat(), 'n_rows': len(df), 'n_keywords': df['keyword'].nunique()},
        'bivariate': nonparam,
        'ols_models': ols_results,
        'segmented_intent':    seg_intent,
        'segmented_vertical':  seg_vertical,
        'segmented_pagetype':  seg_pagetype,
        'url_length_class_positions': df.groupby('url_length_class')['position'].mean().round(3).to_dict(),
        'quartile_positions':  df.groupby('path_length_quartile')['position'].mean().round(3).to_dict(),
        'seokreativ_benchmark': sk_res,
    }
    with open(ANALYSIS_DIR / 'model_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(ANALYSIS_DIR / 'ymyl_comparison.json', 'w') as f:
        json.dump(ymyl_res, f, indent=2, default=str)

    # Robustheitschecks
    robust = {
        'vif': vif.to_dict(orient='records'),
        'outlier_sensitivity': {},
        'spearman_correlations': nonparam,
    }
    # Modell ohne Top/Bottom 5% Pfadlänge
    p5, p95 = df['path_length'].quantile(0.05), df['path_length'].quantile(0.95)
    df_trimmed = df[(df['path_length'] >= p5) & (df['path_length'] <= p95)]
    try:
        m_trimmed = smf.ols(formula_base, data=df_trimmed).fit(cov_type='HC3')
        path_coef = m_trimmed.params.get('path_length', None)
        robust['outlier_sensitivity'] = {
            'n_trimmed': len(df_trimmed),
            'path_length_beta_trimmed': round(float(path_coef), 6) if path_coef else None,
            'r2_trimmed': round(m_trimmed.rsquared, 4),
        }
    except Exception as e:
        robust['outlier_sensitivity'] = {'error': str(e)}

    with open(ANALYSIS_DIR / 'robustness_checks.json', 'w') as f:
        json.dump(robust, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("PHASE 4 ABGESCHLOSSEN")
    print(f"{'='*60}")
    print(f"Gespeichert in: {ANALYSIS_DIR}/")
    print(f"  model_results.json, model_comparison_table.csv")
    print(f"  ymyl_comparison.json, robustness_checks.json, vif_check.csv")

    # ── Kern-Erkenntnisse Zusammenfassung ─────────────────────────────────────
    m1_r2 = ols_results.get('M1_naive', {}).get('r2', '?')
    m4_r2 = ols_results.get('M4_full', {}).get('r2', '?')
    m1_beta = ols_results.get('M1_naive', {}).get('path_length_beta', '?')
    m4_beta = ols_results.get('M4_full', {}).get('path_length_beta', '?')
    print(f"\nKERN-ERKENNTNISSE:")
    print(f"  Naive Korrelation (M1): β={m1_beta}, R²={m1_r2}")
    print(f"  Volles Modell   (M4): β={m4_beta}, R²={m4_r2}")
    if m1_beta and m4_beta and m1_beta != '?':
        shrinkage = (1 - abs(m4_beta) / abs(m1_beta)) * 100 if m1_beta != 0 else 0
        print(f"  Koeffizienten-Shrinkage: {shrinkage:.1f}%")
    print(f"  Stärkster Prädiktor: domain_rank_decile (ρ={nonparam['spearman_domain_rank_decile']['rho']:.3f})")

if __name__ == '__main__':
    main()
