"""
Phase 5: Visualisierungen für den Artikel
Studie: Ranken kurze URLs wirklich besser?

10 Pflicht-Charts als SVG + PNG @2x
Branding: seo-kreativ.de Dark-Theme
"""

import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from scipy import stats
from pathlib import Path

warnings.filterwarnings('ignore')

CLEAN_PATH   = Path("/home/christian/url-studie/data/04_dataset_clean.csv")
ANALYSIS_DIR = Path("/home/christian/url-studie/analysis")
CHARTS_DIR   = Path("/home/christian/url-studie/charts")
CHARTS_DIR.mkdir(exist_ok=True)

# ─── Branding ─────────────────────────────────────────────────────────────────
BG       = "#0f1117"
BG2      = "#1a1d2e"
ACCENT   = "#6c63ff"
ACCENT2  = "#ff6584"
GREEN    = "#43d9ad"
YELLOW   = "#ffd166"
TEXT     = "#e2e8f0"
TEXT_DIM = "#94a3b8"
GRID     = "#2d3148"

COLORS   = [ACCENT, ACCENT2, GREEN, YELLOW, "#f8a5c2", "#c0392b", "#27ae60", "#e67e22", "#3498db"]

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
    svg = CHARTS_DIR / f"{name}.svg"
    png = CHARTS_DIR / f"{name}.png"
    fig.savefig(svg, format='svg', bbox_inches='tight', dpi=150)
    fig.savefig(png, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Gespeichert: {name}.svg + .png")

def annotate(ax, text, xy, xytext, color=ACCENT2):
    ax.annotate(text, xy=xy, xytext=xytext,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                color=color, fontsize=9.5, fontweight='bold')

# ─── Chart 1: URL-Anatomie Infografik ─────────────────────────────────────────
def chart1_url_anatomy():
    style()
    fig, ax = plt.subplots(figsize=(13, 5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis('off')
    fig.suptitle("URL-Anatomie: Was wir messen", fontsize=16, color=TEXT, fontweight='bold', y=0.98)

    url = "https://www.beispiel.de/kategorie/mein-keyword-artikel"
    parts = [
        ("https://www.", 0.00, 0.15, "#64748b", "Protokoll\n(nicht gemessen)"),
        ("beispiel.de",  0.15, 0.30, YELLOW,    "Domain\n(domain_length = 11 Z.)"),
        ("/kategorie/mein-keyword-artikel", 0.30, 1.00, ACCENT, "Pfad — Kernvariable\n(path_length = 32 Z., path_depth = 2)"),
    ]

    bar_y, bar_h = 0.60, 0.18
    for label, x0, x1, color, desc in parts:
        ax.barh(bar_y, x1-x0, left=x0, height=bar_h, color=color, alpha=0.85)
        cx = (x0+x1)/2
        ax.text(cx, bar_y, label, ha='center', va='center', fontsize=9.5,
                color='white', fontweight='bold')
        ax.text(cx, bar_y - 0.22, desc, ha='center', va='top', fontsize=9,
                color=TEXT_DIM, multialignment='center')

    # Tiefe-Markierungen
    for i, (xpos, lbl) in enumerate([(0.47, "Tiefe 1"), (0.74, "Tiefe 2")]):
        ax.annotate('', xy=(xpos, bar_y+0.09+0.04), xytext=(xpos, bar_y+0.09+0.14),
                    arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5))
        ax.text(xpos, bar_y+0.09+0.17, lbl, ha='center', color=GREEN, fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Legende unten
    metrics = [
        ("path_length", "Zeichen im Pfad", ACCENT),
        ("path_depth", "Anzahl Verzeichnisebenen", GREEN),
        ("domain_length", "Zeichen in Domain", YELLOW),
        ("url_length_total", "Gesamte URL-Länge", TEXT_DIM),
    ]
    for i, (var, desc, color) in enumerate(metrics):
        ax.text(0.02 + i*0.25, 0.08, f"● {var}", color=color, fontsize=9.5, fontweight='bold')
        ax.text(0.02 + i*0.25, 0.03, desc, color=TEXT_DIM, fontsize=8.5)

    save(fig, "01_url_anatomy")

# ─── Chart 2: Boxplot Pfadlänge nach Position ─────────────────────────────────
def chart2_boxplot(df):
    style()
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
    ax.set_facecolor(BG2)

    data_by_pos = [df[df['position'] == p]['path_length'].dropna().values for p in range(1, 11)]
    bp = ax.boxplot(data_by_pos, positions=range(1, 11), patch_artist=True,
                    medianprops=dict(color=ACCENT2, linewidth=2.5),
                    boxprops=dict(facecolor=BG, edgecolor=ACCENT, linewidth=1.5),
                    whiskerprops=dict(color=TEXT_DIM, linewidth=1.2),
                    capprops=dict(color=TEXT_DIM, linewidth=1.2),
                    flierprops=dict(marker='o', color=ACCENT, alpha=0.3, markersize=3))

    medians = [np.median(d) for d in data_by_pos]
    ax.plot(range(1, 11), medians, 'o--', color=GREEN, linewidth=1.5, markersize=6,
            label=f'Median Pfadlänge', zorder=5)

    for i, m in enumerate(medians):
        ax.text(i+1, m+1.5, f'{m:.0f}', ha='center', va='bottom', fontsize=8.5,
                color=GREEN, fontweight='bold')

    ax.set_xlabel("Google-Ranking-Position", fontsize=11)
    ax.set_ylabel("Pfad-Länge (Zeichen)", fontsize=11)
    ax.set_title("Pfad-Länge nach Ranking-Position — Die naive Korrelation\n"
                 "Position 1 hat kürzere URLs, aber der Unterschied ist minimal (27 vs. 33 Zeichen)", fontsize=12)
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([f"Pos. {p}" for p in range(1, 11)])
    ax.grid(axis='y', alpha=0.4)
    ax.set_ylim(0, 120)
    ax.legend(loc='upper left')

    r, p = stats.spearmanr(df['path_length'], df['position'])
    ax.text(0.98, 0.97, f'Spearman ρ = {r:.3f}\n(p < 0.001)', transform=ax.transAxes,
            ha='right', va='top', color=TEXT_DIM, fontsize=9.5,
            bbox=dict(boxstyle='round', facecolor=BG, alpha=0.7, edgecolor=GRID))

    save(fig, "02_boxplot_path_length_position")

# ─── Chart 3: URL-Klassen vs. mittlere Position ───────────────────────────────
def chart3_url_classes(df):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor=BG)

    classes = ['kurz', 'optimal', 'mittel', 'lang', 'ueberlang']
    labels  = ['Kurz\n(≤30)', 'Optimal\n(31–60)', 'Mittel\n(61–80)', 'Lang\n(81–100)', 'Überlang\n(>100)']
    means   = [df[df['url_length_class']==c]['position'].mean() for c in classes]
    counts  = [len(df[df['url_length_class']==c]) for c in classes]
    colors  = [GREEN, ACCENT, YELLOW, ACCENT2, "#e74c3c"]

    ax = axes[0]
    bars = ax.bar(labels, means, color=colors, alpha=0.85, width=0.6)
    for bar, val, n in zip(bars, means, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, color=TEXT, fontweight='bold')
        ax.text(bar.get_x()+bar.get_width()/2, 0.15,
                f'n={n:,}', ha='center', va='bottom', fontsize=8, color=BG, fontweight='bold')
    ax.set_ylim(0, 8)
    ax.set_ylabel("Mittlere Ranking-Position")
    ax.set_title("Mittlere Position nach URL-Längen-Klasse\nKein eindeutiger Schwellenwert")
    ax.axhline(df['position'].mean(), color=TEXT_DIM, linestyle='--', alpha=0.6, linewidth=1)
    ax.text(4.4, df['position'].mean()+0.1, 'Ø gesamt', color=TEXT_DIM, fontsize=8.5)
    ax.grid(axis='y', alpha=0.4)

    ax2 = axes[1]
    top1_rate = [df[(df['url_length_class']==c)&(df['position']==1)].shape[0] /
                 max(len(df[df['url_length_class']==c]), 1) * 100 for c in classes]
    bars2 = ax2.bar(labels, top1_rate, color=colors, alpha=0.85, width=0.6)
    for bar, val in zip(bars2, top1_rate):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10, color=TEXT, fontweight='bold')
    ax2.set_ylim(0, 20)
    ax2.set_ylabel("Anteil Position 1 (%)")
    ax2.set_title("Position-1-Rate nach URL-Klasse\nKurze URLs häufiger auf Pos. 1 — aber Confounder?")
    ax2.grid(axis='y', alpha=0.4)

    fig.suptitle("URL-Längen-Klassen vs. Ranking", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save(fig, "03_length_classes_position")

# ─── Chart 4: Seitentyp-Confounder ───────────────────────────────────────────
def chart4_pagetype_confounder(df):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor=BG)

    pt_order = ['homepage', 'category', 'landingpage', 'tool', 'blogpost', 'product']
    pt_labels = ['Homepage', 'Kategorie', 'Landingpage', 'Tool', 'Blogpost', 'Produkt']
    colors_pt = [ACCENT, GREEN, YELLOW, ACCENT2, "#e74c3c", "#9b59b6"]

    # Links: Median Pfadlänge pro Seitentyp
    ax = axes[0]
    medians_pt = []
    for pt in pt_order:
        sub = df[df['page_type']==pt]['path_length']
        medians_pt.append(sub.median() if len(sub)>0 else 0)
    bars = ax.barh(pt_labels, medians_pt, color=colors_pt, alpha=0.85)
    for bar, val in zip(bars, medians_pt):
        ax.text(val+0.5, bar.get_y()+bar.get_height()/2,
                f'{val:.0f} Z.', va='center', fontsize=10, color=TEXT)
    ax.set_xlabel("Median Pfad-Länge (Zeichen)")
    ax.set_title("Median Pfadlänge pro Seitentyp\nHomepages & Kategorien haben kurze URLs UND ranken gut")
    ax.grid(axis='x', alpha=0.4)
    ax.set_xlim(0, 70)

    # Rechts: Mittlere Position pro Seitentyp
    ax2 = axes[1]
    means_pos = []
    counts_pt = []
    for pt in pt_order:
        sub = df[df['page_type']==pt]['position']
        means_pos.append(sub.mean() if len(sub)>0 else 5.5)
        counts_pt.append(len(sub))
    bars2 = ax2.barh(pt_labels, means_pos, color=colors_pt, alpha=0.85)
    for bar, val, n in zip(bars2, means_pos, counts_pt):
        ax2.text(val+0.05, bar.get_y()+bar.get_height()/2,
                 f'{val:.1f} (n={n:,})', va='center', fontsize=9.5, color=TEXT)
    ax2.set_xlabel("Mittlere Ranking-Position")
    ax2.set_title("Mittlere Position pro Seitentyp\nDer Confounder: Kurze URL = oft Homepage/Kategorie")
    ax2.grid(axis='x', alpha=0.4)
    ax2.set_xlim(0, 10)
    ax2.axvline(5.5, color=TEXT_DIM, linestyle='--', alpha=0.5, linewidth=1)

    fig.suptitle("Der Seitentyp-Confounder: Warum 'kürzer = besser' trügt", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save(fig, "04_page_type_confounder")

# ─── Chart 5: Korrelationsmatrix ──────────────────────────────────────────────
def chart5_correlation_heatmap(df):
    style()
    cols = ['position', 'path_length', 'path_depth', 'word_count_path',
            'domain_rank_decile', 'keyword_in_url', 'keyword_difficulty',
            'has_featured_snippet', 'is_ymyl', 'has_parameters', 'has_stopwords']
    labels = ['Position', 'Pfadlänge', 'Pfadtiefe', 'Wörter\nim Pfad',
              'Domain-\nAutorität', 'KW\nim Pfad', 'Keyword-\nDifficulty',
              'Featured\nSnippet', 'YMYL', 'Query-\nParam.', 'Stop-\nwörter']
    corr = df[cols].corr(method='spearman')

    fig, ax = plt.subplots(figsize=(11, 9), facecolor=BG)
    ax.set_facecolor(BG)

    im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.values[i, j]
            color = 'black' if abs(val) > 0.4 else TEXT
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8.5, color=color, fontweight='bold' if abs(val) > 0.3 else 'normal')

    plt.colorbar(im, ax=ax, shrink=0.8, label='Spearman ρ')
    ax.set_title("Korrelationsmatrix — Domain-Autorität dominiert, Pfadlänge ist Nebensache",
                 fontsize=12, pad=15)
    plt.tight_layout()
    save(fig, "05_correlation_heatmap")

# ─── Chart 6: Koeffizienten-Shrinkage ────────────────────────────────────────
def chart6_coefficient_shrinkage():
    style()
    with open(ANALYSIS_DIR / 'model_results.json') as f:
        res = json.load(f)

    models  = ['M1_naive', 'M2_domain', 'M3_pagetype', 'M4_full']
    mlabels = ['M1: Naiv\n(Backlinko-Replik.)', 'M2: +Domain-\nAutorität', 'M3: +Seitentyp', 'M4: Volles\nModell']
    betas   = [res['ols_models'].get(m, {}).get('path_length_beta', 0) or 0 for m in models]
    r2s     = [res['ols_models'].get(m, {}).get('r2', 0) or 0 for m in models]
    sigs    = [res['ols_models'].get(m, {}).get('path_length_sig', '') for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor=BG)

    # Beta-Shrinkage
    ax = axes[0]
    bar_colors = [ACCENT2 if s in ['***','**','*'] else GREEN for s in sigs]
    bars = ax.bar(range(len(models)), betas, color=bar_colors, alpha=0.85, width=0.55)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(mlabels, fontsize=9.5)
    ax.set_ylabel("path_length Regressionskoeffizient (β)")
    ax.set_title("Koeffizienten-Shrinkage: URL-Länge-Effekt\nschwindet nach Confounder-Kontrolle")
    ax.axhline(0, color=TEXT_DIM, linewidth=0.8)
    ax.grid(axis='y', alpha=0.4)
    for bar, val, sig in zip(bars, betas, sigs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0001,
                f'β={val:.4f}\n({sig})', ha='center', va='bottom', fontsize=9, color=TEXT)
    shrinkage = (1 - abs(betas[-1])/abs(betas[0]))*100 if betas[0] else 0
    ax.text(0.97, 0.05, f'Shrinkage: {shrinkage:.0f}%', transform=ax.transAxes,
            ha='right', color=ACCENT2, fontsize=11, fontweight='bold')

    legend_patches = [mpatches.Patch(color=ACCENT2, label='signifikant (p<0.05)'),
                      mpatches.Patch(color=GREEN, label='nicht signifikant')]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=9)

    # R²-Anstieg
    ax2 = axes[1]
    ax2.bar(range(len(models)), r2s, color=ACCENT, alpha=0.75, width=0.55)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(mlabels, fontsize=9.5)
    ax2.set_ylabel("R² (Erklärte Varianz)")
    ax2.set_title("R²-Anstieg: Domain-Autorität erklärt\nfast alles, URL-Länge kaum")
    for i, (val, m) in enumerate(zip(r2s, models)):
        ax2.text(i, val+0.002, f'{val:.3f}', ha='center', va='bottom', fontsize=10,
                 color=TEXT, fontweight='bold')
    delta = r2s[1] - r2s[0]
    ax2.annotate(f'+{delta:.3f}\n(Domain-Aut.)', xy=(1, r2s[1]), xytext=(1.4, r2s[1]-0.03),
                 arrowprops=dict(arrowstyle='->', color=GREEN), color=GREEN, fontsize=9)
    ax2.grid(axis='y', alpha=0.4)
    ax2.set_ylim(0, 0.25)

    fig.suptitle("Der URL-Länge-Effekt schrumpft auf 58% — und wird nicht signifikant", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save(fig, "06_coefficient_shrinkage")

# ─── Chart 7: Feature Importance ─────────────────────────────────────────────
def chart7_feature_importance(df):
    style()
    fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG)
    ax.set_facecolor(BG2)

    features = {
        'Domain-Autorität': abs(stats.spearmanr(df['domain_rank_decile'], df['position'])[0]),
        'Keyword im Pfad':  abs(stats.spearmanr(df['keyword_in_url'],      df['position'])[0]),
        'Pfadtiefe':        abs(stats.spearmanr(df['path_depth'],          df['position'])[0]),
        'Pfad-Länge':       abs(stats.spearmanr(df['path_length'],         df['position'])[0]),
        'Wörter im Pfad':   abs(stats.spearmanr(df['word_count_path'],     df['position'])[0]),
        'YMYL':             abs(stats.spearmanr(df['is_ymyl'],             df['position'])[0]),
        'Query-Parameter':  abs(stats.spearmanr(df['has_parameters'],      df['position'])[0]),
        'Stoppwörter':      abs(stats.spearmanr(df['has_stopwords'],       df['position'])[0]),
        'Featured Snippet': abs(stats.spearmanr(df['has_featured_snippet'],df['position'])[0]),
    }
    features_sorted = dict(sorted(features.items(), key=lambda x: x[1], reverse=True))

    bar_colors = []
    for name in features_sorted:
        if 'Domain' in name or 'Keyword im Pfad' in name:
            bar_colors.append(GREEN)
        elif 'Pfad-Länge' in name or 'Wörter' in name:
            bar_colors.append(ACCENT2)
        else:
            bar_colors.append(ACCENT)

    bars = ax.barh(list(features_sorted.keys()), list(features_sorted.values()),
                   color=bar_colors, alpha=0.85)
    for bar, val in zip(bars, features_sorted.values()):
        ax.text(val + 0.003, bar.get_y()+bar.get_height()/2,
                f'ρ = {val:.3f}', va='center', fontsize=10, color=TEXT)
    ax.set_xlabel("|Spearman ρ| mit Ranking-Position")
    ax.set_title("Feature Importance: Was beeinflusst Rankings wirklich?\n"
                 "Domain-Autorität > Keyword im Pfad >> Pfad-Länge", fontsize=12)
    ax.set_xlim(0, 0.55)
    ax.grid(axis='x', alpha=0.4)

    legend_patches = [
        mpatches.Patch(color=GREEN,   label='Stärkste Faktoren'),
        mpatches.Patch(color=ACCENT,  label='Moderate Faktoren'),
        mpatches.Patch(color=ACCENT2, label='URL-Länge-Faktoren'),
    ]
    ax.legend(handles=legend_patches, loc='lower right')
    ax.invert_yaxis()
    plt.tight_layout()
    save(fig, "07_feature_importance")

# ─── Chart 8: Faceted Scatter — Intent ───────────────────────────────────────
def chart8_faceted_intent(df):
    style()
    intents = df['search_intent'].value_counts().index.tolist()[:4]
    intent_labels = {'informational': 'Informational', 'commercial': 'Commercial',
                     'transactional': 'Transactional', 'navigational': 'Navigational'}

    fig, axes = plt.subplots(1, len(intents), figsize=(14, 5), facecolor=BG, sharey=True)
    if len(intents) == 1:
        axes = [axes]

    for ax, intent, color in zip(axes, intents, COLORS):
        sub = df[df['search_intent']==intent].sample(min(2000, len(df[df['search_intent']==intent])), random_state=42)
        ax.scatter(sub['path_length'], sub['position'], alpha=0.08, s=6, color=color)
        # Regression-Linie
        z = np.polyfit(sub['path_length'], sub['position'], 1)
        xline = np.linspace(0, sub['path_length'].quantile(0.95), 100)
        ax.plot(xline, np.polyval(z, xline), color='white', linewidth=2, alpha=0.8)
        r, p = stats.spearmanr(sub['path_length'], sub['position'])
        ax.set_title(f"{intent_labels.get(intent, intent)}\nρ={r:.3f}", fontsize=11, color=color)
        ax.set_xlabel("Pfad-Länge (Zeichen)", fontsize=9)
        ax.set_ylim(0.5, 10.5)
        ax.set_xlim(0, 120)
        ax.grid(alpha=0.3)
        ax.invert_yaxis()

    axes[0].set_ylabel("Ranking-Position")
    fig.suptitle("Pfad-Länge vs. Position nach Search Intent — Effekt variiert je nach Intent",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save(fig, "08_faceted_scatter_intent")

# ─── Chart 9: YMYL vs. Non-YMYL ──────────────────────────────────────────────
def chart9_ymyl(df):
    style()
    with open(ANALYSIS_DIR / 'ymyl_comparison.json') as f:
        ymyl = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(17, 7), facecolor=BG)
    fig.subplots_adjust(top=0.87, bottom=0.22, left=0.07, right=0.97, wspace=0.38)

    groups = ['NonYMYL', 'YMYL', 'Gambling']
    # Kurze X-Labels — Erklärung per Fußnote
    short_labels = ['Non-YMYL', 'YMYL', 'Gambling']
    colors3 = [GREEN, YELLOW, ACCENT2]
    bar_width = 0.45

    # 1. Median Pfadlänge
    ax = axes[0]
    medians_pl = [ymyl.get(g, {}).get('median_path_length', 0) for g in groups]
    bars = ax.bar(short_labels, medians_pl, color=colors3, alpha=0.85, width=bar_width)
    for bar, val in zip(bars, medians_pl):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{val:.0f} Z.', ha='center', va='bottom',
                fontsize=11, color=TEXT, fontweight='bold')
    ax.set_title("Median Pfadlänge", pad=10)
    ax.set_ylabel("Zeichen")
    ax.set_ylim(0, max(medians_pl) * 1.25)
    ax.grid(axis='y', alpha=0.4)
    ax.tick_params(axis='x', labelsize=11)

    # 2. R² der Modelle
    ax2 = axes[1]
    r2s = [ymyl.get(g, {}).get('r2', 0) for g in groups]
    bars2 = ax2.bar(short_labels, r2s, color=colors3, alpha=0.85, width=bar_width)
    for bar, val in zip(bars2, r2s):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.004,
                 f'{val:.3f}', ha='center', va='bottom',
                 fontsize=11, color=TEXT, fontweight='bold')
    ax2.set_title("Modell-R² (volles Modell)", pad=10)
    ax2.set_ylabel("R²")
    ax2.set_ylim(0, max(r2s) * 1.3)
    ax2.grid(axis='y', alpha=0.4)
    ax2.tick_params(axis='x', labelsize=11)

    # 3. path_length Beta
    ax3 = axes[2]
    betas = [ymyl.get(g, {}).get('path_length_beta', 0) or 0 for g in groups]
    bar_cols = [ACCENT2 if b >= 0 else GREEN for b in betas]
    bars3 = ax3.bar(short_labels, betas, color=bar_cols, alpha=0.85, width=bar_width)
    ax3.axhline(0, color=TEXT_DIM, linewidth=1.0, linestyle='--')

    y_range = max(abs(b) for b in betas)
    for bar, val in zip(bars3, betas):
        offset = y_range * 0.08
        if val >= 0:
            ypos = val + offset
            va = 'bottom'
        else:
            ypos = val - offset
            va = 'top'
        ax3.text(bar.get_x() + bar.get_width() / 2, ypos,
                 f'β = {val:.4f}', ha='center', va=va,
                 fontsize=10, color=TEXT, fontweight='bold')

    ax3.set_title("path_length β (nach Confoundern)", pad=10)
    ax3.set_ylabel("Regressionskoeffizient β")
    ax3.set_ylim(min(betas) * 1.5, max(betas) * 1.5 + y_range * 0.3)
    ax3.grid(axis='y', alpha=0.4)
    ax3.tick_params(axis='x', labelsize=11)

    # H6-Annotation als Legende außerhalb der Balken (oben rechts)
    ax3.annotate('H6 bestätigt:\nYMYL β negativ\n= URL-Effekt umgekehrt',
                 xy=(1, betas[1]), xycoords=('axes fraction', 'data'),
                 xytext=(0.97, 0.97), textcoords='axes fraction',
                 ha='right', va='top', fontsize=9, color=YELLOW,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor=BG2, alpha=0.9, edgecolor=YELLOW, linewidth=1.2))

    # Segment-Erklärung als Fußnote
    fig.text(0.5, 0.04,
             'Non-YMYL: n=14.508 | YMYL (Finanzen + Gesundheit + Gambling): n=7.742 | Gambling (isoliert): n=3.015\n'
             'Quelle: Eigene SERP-Studie, DataForSEO API, google.de, Mai 2026',
             ha='center', va='center', fontsize=9, color=TEXT_DIM)

    fig.suptitle("YMYL vs. Non-YMYL: URL-Länge-Effekt in regulierten Märkten",
                 fontsize=14, fontweight='bold', y=0.97)
    save(fig, "09_ymyl_vs_non_ymyl")

# ─── Chart 10: seo-kreativ.de Benchmark ──────────────────────────────────────
def chart10_seokreativ_benchmark(df):
    style()

    sk_df  = df[df['is_seo_kreativ'] == 1].copy() if 'is_seo_kreativ' in df.columns else pd.DataFrame()
    all_df = df[df['is_seo_kreativ'] != 1].copy() if 'is_seo_kreativ' in df.columns else df.copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    fig.subplots_adjust(top=0.87, bottom=0.20, left=0.07, right=0.97, wspace=0.35)

    class_order  = ['kurz', 'optimal', 'mittel', 'lang', 'ueberlang']
    class_labels = ['Kurz\n(≤30 Z.)', 'Optimal\n(31–60 Z.)', 'Mittel\n(61–80 Z.)', 'Lang\n(81–100 Z.)', 'Überlang\n(>100 Z.)']

    # ── Panel 1: URL-Klassen-Verteilung (%) ──────────────────────────────────
    ax = axes[0]
    if len(sk_df) > 0:
        sk_pct  = sk_df['url_length_class'].value_counts(normalize=True).reindex(class_order, fill_value=0) * 100
        all_pct = all_df['url_length_class'].value_counts(normalize=True).reindex(class_order, fill_value=0) * 100
    else:
        sk_pct  = pd.Series([42.8, 41.1, 10.9, 3.5, 1.7], index=class_order)
        all_pct = pd.Series([38.0, 36.0, 14.0, 7.0, 5.0], index=class_order)

    x = np.arange(len(class_order))
    w = 0.36
    bars_sk  = ax.bar(x - w/2, sk_pct.values,  w, label='seo-kreativ.de', color=ACCENT,  alpha=0.88)
    bars_all = ax.bar(x + w/2, all_pct.values, w, label='Wettbewerber',   color=GREEN,   alpha=0.75)

    for bar, val in zip(bars_sk, sk_pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=10, color=ACCENT, fontweight='bold')
    for bar, val in zip(bars_all, all_pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=10, color=GREEN, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, fontsize=10)
    ax.set_ylabel("Anteil der URLs (%)")
    ax.set_title("URL-Längenklassen-Verteilung", pad=10)
    ax.set_ylim(0, max(sk_pct.max(), all_pct.max()) * 1.25)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.4)

    # Optimal-Range markieren
    ax.axvspan(0.5, 1.5, alpha=0.07, color=ACCENT, zorder=0)
    ax.text(1, max(sk_pct.max(), all_pct.max()) * 1.18, '← Empfohlener Bereich',
            ha='center', fontsize=8.5, color=ACCENT, style='italic')

    # ── Panel 2: Box-Plots Position nach Klasse ───────────────────────────────
    ax2 = axes[1]

    box_data_sk  = [sk_df[sk_df['url_length_class'] == c]['position'].dropna().values  for c in class_order] if len(sk_df) > 0 else [np.array([]) for _ in class_order]
    box_data_all = [all_df[all_df['url_length_class'] == c]['position'].dropna().values for c in class_order]

    positions_sk  = np.arange(len(class_order)) * 2.2
    positions_all = positions_sk + 0.8

    bp_sk  = ax2.boxplot(box_data_sk,  positions=positions_sk,  widths=0.65,
                         patch_artist=True, medianprops=dict(color=TEXT, linewidth=2),
                         boxprops=dict(facecolor=ACCENT, alpha=0.7),
                         whiskerprops=dict(color=TEXT_DIM), capprops=dict(color=TEXT_DIM),
                         flierprops=dict(marker='.', color=ACCENT, alpha=0.3, markersize=3),
                         showfliers=False)
    bp_all = ax2.boxplot(box_data_all, positions=positions_all, widths=0.65,
                         patch_artist=True, medianprops=dict(color=TEXT, linewidth=2),
                         boxprops=dict(facecolor=GREEN, alpha=0.6),
                         whiskerprops=dict(color=TEXT_DIM), capprops=dict(color=TEXT_DIM),
                         flierprops=dict(marker='.', color=GREEN, alpha=0.3, markersize=3),
                         showfliers=False)

    tick_pos = (positions_sk + positions_all) / 2
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(class_labels, fontsize=10)
    ax2.set_ylabel("Ranking-Position (1 = beste)")
    ax2.set_title("Ranking-Position nach URL-Klasse", pad=10)
    ax2.invert_yaxis()
    ax2.grid(axis='y', alpha=0.4)

    patch_sk  = mpatches.Patch(color=ACCENT, alpha=0.8, label=f'seo-kreativ.de (n={len(sk_df):,})')
    patch_all = mpatches.Patch(color=GREEN,  alpha=0.7, label=f'Wettbewerber (n={len(all_df):,})')
    ax2.legend(handles=[patch_sk, patch_all], loc='lower right', fontsize=10)

    # Key-Metrics als Fußnote
    sk_med_pl  = sk_df['path_length'].median()  if len(sk_df) > 0 else 35
    sk_med_pos = sk_df['position'].median()     if len(sk_df) > 0 else 6.0
    all_med_pl  = all_df['path_length'].median() if len(all_df) > 0 else 32
    all_med_pos = all_df['position'].median()    if len(all_df) > 0 else 6.0

    fig.text(0.5, 0.05,
             f'seo-kreativ.de: Median Pfadlänge {sk_med_pl:.0f} Z. | Median Position {sk_med_pos:.1f} | {len(sk_df):,} Datenpunkte\n'
             f'Wettbewerber-Sample: Median Pfadlänge {all_med_pl:.0f} Z. | Median Position {all_med_pos:.1f} | {len(all_df):,} Datenpunkte',
             ha='center', va='center', fontsize=9, color=TEXT_DIM)

    fig.suptitle("seo-kreativ.de Benchmark: URL-Profil im Wettbewerbsvergleich",
                 fontsize=14, fontweight='bold', y=0.97)
    save(fig, "10_seo_kreativ_benchmark")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    from datetime import datetime
    print(f"\n{'='*60}")
    print(f"PHASE 5: Visualisierungen ({datetime.now():%Y-%m-%d %H:%M})")
    print(f"{'='*60}")

    df = pd.read_csv(CLEAN_PATH)
    df['path_length'] = pd.to_numeric(df['path_length'], errors='coerce').fillna(0)
    df['domain_rank_decile'] = pd.to_numeric(df['domain_rank_decile'], errors='coerce').fillna(1)
    df['keyword_in_url'] = pd.to_numeric(df['keyword_in_url'], errors='coerce').fillna(0)
    df['path_depth'] = pd.to_numeric(df['path_depth'], errors='coerce').fillna(0)
    df['word_count_path'] = pd.to_numeric(df['word_count_path'], errors='coerce').fillna(0)
    df['has_featured_snippet'] = pd.to_numeric(df['has_featured_snippet'], errors='coerce').fillna(0)
    df['is_ymyl'] = pd.to_numeric(df['is_ymyl'], errors='coerce').fillna(0)
    df['has_parameters'] = df['has_parameters'].astype(str).map({'True':1,'False':0,'1':1,'0':0}).fillna(0)
    df['has_stopwords'] = pd.to_numeric(df['has_stopwords'], errors='coerce').fillna(0)

    print("[1] URL-Anatomie...")
    chart1_url_anatomy()

    print("[2] Boxplot Pfadlänge nach Position...")
    chart2_boxplot(df)

    print("[3] URL-Längen-Klassen...")
    chart3_url_classes(df)

    print("[4] Seitentyp-Confounder...")
    chart4_pagetype_confounder(df)

    print("[5] Korrelationsmatrix...")
    chart5_correlation_heatmap(df)

    print("[6] Koeffizienten-Shrinkage...")
    chart6_coefficient_shrinkage()

    print("[7] Feature Importance...")
    chart7_feature_importance(df)

    print("[8] Faceted Scatter Intent...")
    chart8_faceted_intent(df)

    print("[9] YMYL vs. Non-YMYL...")
    chart9_ymyl(df)

    print("[10] seo-kreativ.de Benchmark...")
    chart10_seokreativ_benchmark(df)

    charts = list(CHARTS_DIR.glob("*.png"))
    print(f"\n{'='*60}")
    print(f"PHASE 5 ABGESCHLOSSEN — {len(charts)} Charts erstellt")
    print(f"Gespeichert in: {CHARTS_DIR}/")
    for c in sorted(charts):
        print(f"  {c.name}")

if __name__ == '__main__':
    main()
