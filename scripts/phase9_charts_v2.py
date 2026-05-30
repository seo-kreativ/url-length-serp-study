"""
Phase 9: Charts v2 — Neue Daten + Spec-Curve

Renderungen:
  06_coefficient_shrinkage_v2.png/svg  — M1-M4 mit neuen Werten
  09_ymyl_vs_non_ymyl_v2.png/svg       — Alle Segmente n.s. + Disclosure
  10_seo_kreativ_benchmark_v2.png/svg  — Mit Authority-Decile-Vergleich
  11_spec_curve.png/svg                — NEU: Authority-Spezifikationen Robustheit
"""

import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings('ignore')

ROOT = Path("/home/christian/url-studie")
DATA = ROOT / "data"
OUT  = ROOT / "analysis_v2"
CHARTS = ROOT / "charts_v2"
CHARTS.mkdir(exist_ok=True)

# Branding (gleich wie phase5)
BG       = "#0f1117"
BG2      = "#1a1d2e"
ACCENT   = "#6c63ff"
ACCENT2  = "#ff6584"
GREEN    = "#43d9ad"
YELLOW   = "#ffd166"
RED      = "#e53e3e"
TEXT     = "#e2e8f0"
TEXT_DIM = "#94a3b8"
GRID     = "#2d3148"

def style():
    plt.rcParams.update({
        'figure.facecolor':  BG,
        'axes.facecolor':    BG2,
        'axes.edgecolor':    GRID,
        'axes.labelcolor':   TEXT,
        'axes.titlecolor':   TEXT,
        'xtick.color':       TEXT_DIM,
        'ytick.color':       TEXT_DIM,
        'text.color':        TEXT,
        'grid.color':        GRID,
        'grid.linewidth':    0.6,
        'font.family':       'DejaVu Sans',
        'font.size':         11,
        'axes.titlesize':    13,
        'axes.labelsize':    11,
        'legend.facecolor':  BG2,
        'legend.edgecolor':  GRID,
        'legend.labelcolor': TEXT,
    })

def save(fig, name):
    svg = CHARTS / f"{name}.svg"
    png = CHARTS / f"{name}.png"
    fig.savefig(svg, format='svg', bbox_inches='tight', dpi=150)
    fig.savefig(png, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Gespeichert: {name}.svg + .png  ({png.stat().st_size//1024} KB)")


# ─── Chart 06 v2: Coefficient Shrinkage M1→M4 ─────────────────────────────────
def chart06_v2(results):
    style()
    models = results["models"]
    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor=BG)
    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.10, right=0.95)

    keys   = ["M1_naive", "M2_authority", "M3_pagetype", "M4_full"]
    labels = ["M1\nnur path_length", "M2\n+ log Authority", "M3\n+ Seitentyp", "M4\nVoll-Modell"]
    betas  = [models[k]["path_length_beta"] for k in keys]
    pvals  = [models[k]["path_length_pval"] for k in keys]

    def sig_label(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "n.s."

    def color_for(p):
        if p < 0.05: return ACCENT
        return TEXT_DIM

    cols = [color_for(p) for p in pvals]
    bars = ax.bar(labels, betas, color=cols, alpha=0.88, width=0.5, edgecolor=TEXT_DIM, linewidth=0.5)

    for bar, b, p in zip(bars, betas, pvals):
        sig = sig_label(p)
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.0003,
                f'β = {b:.4f}\n{sig}',
                ha='center', va='bottom', fontsize=11, color=TEXT, fontweight='bold')

    ax.set_ylabel("path_length β (OLS Koeffizient)")
    ax.set_title("Coefficient Shrinkage M1 → M4 (mit externem Authority-Signal, log-transformiert)",
                 pad=12, fontsize=13, fontweight='bold')
    ax.axhline(0, color=TEXT_DIM, linewidth=0.6, linestyle='--')
    ax.grid(axis='y', alpha=0.4)
    ax.set_ylim(0, max(betas) * 1.5)

    # Shrinkage-Annotation
    shrink_pct = (betas[0] - betas[-1]) / betas[0] * 100
    ax.annotate(f'Shrinkage M1 → M4: {shrink_pct:.1f}%',
                xy=(3, betas[-1]), xytext=(2.7, betas[0]*1.2),
                arrowprops=dict(arrowstyle='->', color=ACCENT2, lw=1.5),
                fontsize=11, color=ACCENT2, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=BG2, edgecolor=ACCENT2, alpha=0.95))

    # Fußnote
    fig.text(0.5, 0.04,
             f'N = 22.250 Top-10 SERP-Ergebnisse | Authority = log(organic_keywords_count) via DataForSEO domain_overview\n'
             f'HC3 robuste Standardfehler | Eigene SERP-Studie, google.de, Mai 2026',
             ha='center', va='center', fontsize=9, color=TEXT_DIM)
    save(fig, "06_coefficient_shrinkage_v2")


# ─── Chart 09 v2: YMYL vs Non-YMYL vs Gambling (alle n.s. oder schwach) ──────
def chart09_v2(results):
    style()
    ymyl = results["ymyl_segmented"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 7), facecolor=BG)
    fig.subplots_adjust(top=0.86, bottom=0.22, left=0.07, right=0.97, wspace=0.38)

    groups = ["NonYMYL", "YMYL", "Gambling"]
    short = ["Non-YMYL", "YMYL", "Gambling"]
    colors3 = [GREEN, YELLOW, ACCENT2]
    bar_width = 0.45

    # Panel 1: Median Pfadlänge
    ax = axes[0]
    medians_pl = [ymyl[g]["median_path_length"] for g in groups]
    bars = ax.bar(short, medians_pl, color=colors3, alpha=0.85, width=bar_width)
    for bar, val in zip(bars, medians_pl):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{val:.0f} Z.', ha='center', va='bottom', fontsize=11, color=TEXT, fontweight='bold')
    ax.set_title("Median Pfadlänge", pad=10)
    ax.set_ylabel("Zeichen")
    ax.set_ylim(0, max(medians_pl)*1.25)
    ax.grid(axis='y', alpha=0.4)

    # Panel 2: R² (sehr niedrig — Disclosure)
    ax2 = axes[1]
    r2s = [ymyl[g]["r2"] for g in groups]
    bars2 = ax2.bar(short, r2s, color=colors3, alpha=0.85, width=bar_width)
    for bar, val in zip(bars2, r2s):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=11, color=TEXT, fontweight='bold')
    ax2.set_title("Modell-R² pro Segment", pad=10)
    ax2.set_ylabel("R² (erklärte Varianz)")
    ax2.set_ylim(0, max(r2s)*1.4)
    ax2.grid(axis='y', alpha=0.4)
    ax2.text(0.5, 0.95, "R² < 2% — Modell erklärt nur einen Bruchteil\nder Position-Varianz",
             transform=ax2.transAxes, ha='center', va='top', fontsize=9, color=YELLOW,
             bbox=dict(boxstyle='round,pad=0.4', facecolor=BG2, alpha=0.9, edgecolor=YELLOW))

    # Panel 3: β path_length per Segment — Sig oder n.s.
    ax3 = axes[2]
    betas = [ymyl[g]["path_length_beta"] for g in groups]
    pvals = [ymyl[g]["path_length_pval"] for g in groups]
    bar_cols = [ACCENT if p < 0.05 else TEXT_DIM for p in pvals]
    bars3 = ax3.bar(short, betas, color=bar_cols, alpha=0.85, width=bar_width)
    ax3.axhline(0, color=TEXT_DIM, linewidth=1.0, linestyle='--')

    y_range = max(abs(b) for b in betas) if betas else 0.01
    for bar, b, p in zip(bars3, betas, pvals):
        sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "n.s."))
        offset = y_range * 0.15
        if b >= 0:
            ypos = b + offset; va = 'bottom'
        else:
            ypos = b - offset; va = 'top'
        ax3.text(bar.get_x()+bar.get_width()/2, ypos,
                 f'β={b:+.4f}\n{sig}', ha='center', va=va,
                 fontsize=10, color=TEXT, fontweight='bold')

    ax3.set_title("path_length β nach Confoundern", pad=10)
    ax3.set_ylabel("Regressionskoeffizient β")
    ax3.set_ylim(min(betas)*2 - y_range, max(betas)*2 + y_range)
    ax3.grid(axis='y', alpha=0.4)

    # H6-Annotation: korrigiert, kompakter und außerhalb der Balken
    ax3.text(0.03, 0.97,
             "Nur Non-YMYL signifikant\n(β positiv = kürzere URLs minimal besser)\nYMYL + Gambling: kein Effekt",
             transform=ax3.transAxes, ha='left', va='top', fontsize=8.5, color=YELLOW,
             bbox=dict(boxstyle='round,pad=0.4', facecolor=BG2, alpha=0.95, edgecolor=YELLOW, linewidth=1))

    # Fußnote
    fig.text(0.5, 0.04,
             f'Non-YMYL: n={ymyl["NonYMYL"]["n"]:,} | YMYL (Finanzen+Gesundheit+Gambling): n={ymyl["YMYL"]["n"]:,} | Gambling (isoliert): n={ymyl["Gambling"]["n"]:,}\n'
             f'Authority-Kontrolle: log(organic_keywords_count) via DataForSEO | HC3 robuste SE',
             ha='center', va='center', fontsize=9, color=TEXT_DIM)

    fig.suptitle("URL-Längen-Effekt nach YMYL-Segment (mit externem Authority-Signal)",
                 fontsize=14, fontweight='bold', y=0.97)
    save(fig, "09_ymyl_vs_non_ymyl_v2")


# ─── Chart 10 v2: seo-kreativ.de Benchmark ────────────────────────────────────
def chart10_v2(df):
    style()
    sk = df[df['is_seo_kreativ']==1].copy()
    all_df = df[df['is_seo_kreativ']!=1].copy()
    sk['log_organic_kw'] = np.log1p(sk['organic_keywords_count'])
    all_df['log_organic_kw'] = np.log1p(all_df['organic_keywords_count'])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    fig.subplots_adjust(top=0.86, bottom=0.20, left=0.07, right=0.97, wspace=0.35)

    class_order = ['kurz','optimal','mittel','lang','ueberlang']
    class_labels = ['Kurz\n(≤30)','Optimal\n(31–60)','Mittel\n(61–80)','Lang\n(81–100)','Überlang\n(>100)']

    # Panel 1: URL-Klassen-Verteilung
    ax = axes[0]
    sk_pct = sk['url_length_class'].value_counts(normalize=True).reindex(class_order, fill_value=0)*100
    all_pct = all_df['url_length_class'].value_counts(normalize=True).reindex(class_order, fill_value=0)*100
    x = np.arange(len(class_order))
    w = 0.36
    bars_sk = ax.bar(x-w/2, sk_pct.values, w, label='seo-kreativ.de', color=ACCENT, alpha=0.88)
    bars_all = ax.bar(x+w/2, all_pct.values, w, label='Wettbewerber', color=GREEN, alpha=0.75)
    for bar, val in zip(bars_sk, sk_pct.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.6, f'{val:.0f}%', ha='center', fontsize=10, color=ACCENT, fontweight='bold')
    for bar, val in zip(bars_all, all_pct.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.6, f'{val:.0f}%', ha='center', fontsize=10, color=GREEN, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(class_labels, fontsize=10)
    ax.set_ylabel("Anteil URLs (%)"); ax.set_title("URL-Längenklassen-Verteilung", pad=10)
    ax.set_ylim(0, max(sk_pct.max(), all_pct.max())*1.25)
    ax.legend(loc='upper right', fontsize=10); ax.grid(axis='y', alpha=0.4)

    # Panel 2: Authority-Decile-Vergleich (NEU)
    ax2 = axes[1]
    sk_auth = sk['log_organic_kw'].dropna()
    all_auth = all_df['log_organic_kw'].dropna()

    bins = np.linspace(0, max(sk_auth.max(), all_auth.max()), 30)
    ax2.hist(all_auth, bins=bins, alpha=0.6, color=GREEN, density=True, label=f'Wettbewerber (n={len(all_auth):,})')
    ax2.hist(sk_auth, bins=bins, alpha=0.7, color=ACCENT, density=True, label=f'seo-kreativ.de (n={len(sk_auth):,})')

    ax2.axvline(sk_auth.median(), color=ACCENT, linestyle='--', linewidth=2, label=f'seo-kreativ Median: {sk_auth.median():.2f}')
    ax2.axvline(all_auth.median(), color=GREEN, linestyle='--', linewidth=2, label=f'Wettbewerber Median: {all_auth.median():.2f}')

    ax2.set_xlabel("log(organic_keywords_count) - externes Authority-Signal")
    ax2.set_ylabel("Dichte")
    ax2.set_title("Authority-Verteilung: seo-kreativ.de vs. Wettbewerber", pad=10)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', alpha=0.4)

    fig.text(0.5, 0.05,
             f'seo-kreativ.de: 3.735 Datenpunkte | Median Pfadlänge 35 Z. | Median Position 6,0 | log(organic_kw) Median {sk_auth.median():.2f}\n'
             f'Wettbewerber-Sample: {len(all_df):,} Datenpunkte | Quelle: DataForSEO domain_overview, google.de, Mai 2026',
             ha='center', va='center', fontsize=9, color=TEXT_DIM)

    fig.suptitle("seo-kreativ.de Benchmark: URL-Profil und Authority im Wettbewerbsvergleich",
                 fontsize=14, fontweight='bold', y=0.97)
    save(fig, "10_seo_kreativ_benchmark_v2")


# ─── Chart 11 NEU: Spec-Curve — Authority-Robustheit ──────────────────────────
def chart11_spec_curve(rob):
    style()
    auth = rob["authority_specifications"]

    # Sortiere nach β
    items = [(k, v) for k, v in auth.items() if v and "beta" in v]
    items.sort(key=lambda x: x[1]["beta"])

    labels = [k.replace("OLD_domain_rank_decile (Vergleich)", "OLD n_appearances-Proxy") for k, _ in items]
    betas  = [v["beta"]   for _, v in items]
    ci_lo  = [v["ci_low"] for _, v in items]
    ci_hi  = [v["ci_high"] for _, v in items]
    pvals  = [v["pval"] for _, v in items]
    sig    = [p < 0.05 for p in pvals]

    fig, ax = plt.subplots(figsize=(15, 8.5), facecolor=BG)
    fig.subplots_adjust(top=0.85, bottom=0.18, left=0.30, right=0.96)

    y = np.arange(len(labels))
    cols = [ACCENT if s else TEXT_DIM for s in sig]

    # Konfidenzintervall-Linien
    for i, (lo, hi, c) in enumerate(zip(ci_lo, ci_hi, cols)):
        ax.plot([lo, hi], [i, i], color=c, linewidth=2, alpha=0.7)
        ax.plot([lo, lo], [i-0.2, i+0.2], color=c, linewidth=1.5)
        ax.plot([hi, hi], [i-0.2, i+0.2], color=c, linewidth=1.5)

    # Punktschätzer
    for i, (b, c, s) in enumerate(zip(betas, cols, sig)):
        ax.scatter([b], [i], color=c, s=120, edgecolor=TEXT, linewidth=1.2, zorder=3)

    # Nulllinie
    ax.axvline(0, color=ACCENT2, linewidth=1.2, linestyle='--', alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("path_length β mit 95% Konfidenzintervall  (β=0 = kein Effekt)")
    ax.set_title("Spec-Curve: URL-Längen-Effekt über 10 Authority-Spezifikationen\n"
                 "(jede Zeile = eine andere Art Authority zu kontrollieren)",
                 pad=15, fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.4)

    # Erweiterten X-Range schaffen damit Text + Legend Platz haben
    x_min = min(ci_lo) * 1.15
    x_max = max(ci_hi) * 1.5
    ax.set_xlim(x_min, x_max)

    # Annotation: Robustheits-Befund (oben rechts, klar getrennt)
    ax.text(x_max * 0.98, len(labels) - 1.5,
            "ALLE 10 Spezifikationen → β > 0\n→ Kürzere URLs ranken minimal besser\n→ Effekt klein, aber Vorzeichen robust",
            ha='right', va='top', fontsize=10, color=YELLOW,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=BG2, edgecolor=YELLOW, alpha=0.95))

    # Legend für signifikant vs n.s. (unten links damit nichts überlappt)
    sig_patch = mpatches.Patch(color=ACCENT, label='Signifikant (p < 0.05)')
    ns_patch  = mpatches.Patch(color=TEXT_DIM, label='Nicht signifikant')
    ax.legend(handles=[sig_patch, ns_patch], loc='lower left', fontsize=10, framealpha=0.95)

    fig.text(0.5, 0.02,
             f'N = 22.250 Top-10 SERP-Ergebnisse | HC3 robuste Standardfehler\n'
             f'Authority-Spezifikationen variieren in Transformation (log/raw/sqrt/decile) und Quelle (org. KW / ETV / Top3)',
             ha='center', va='center', fontsize=8.5, color=TEXT_DIM)

    save(fig, "11_spec_curve")


def main():
    print(f"\n{'='*70}\nPHASE 9: CHARTS v2\n{'='*70}")

    # Lade Analyse-Outputs
    results = json.loads((OUT / "model_results.json").read_text())
    rob = json.loads((OUT / "robustness_v2.json").read_text())
    df = pd.read_csv(DATA / "06_dataset_with_authority.csv", low_memory=False)

    chart06_v2(results)
    chart09_v2(results)
    chart10_v2(df)
    chart11_spec_curve(rob)

    print("\nFertig.")


if __name__ == "__main__":
    main()
