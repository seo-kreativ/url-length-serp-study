"""
Phase 3: Feature Engineering
Studie: Ranken kurze URLs wirklich besser?

Input:  data/02_serp_results.csv (27.291 Rows)
Output: data/03_dataset_engineered.csv (alle 29 Variablen)
        data/03_dataset_clean.csv (nach Bereinigung)
"""

import re, urllib.parse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

INPUT_PATH   = Path("/home/christian/url-studie/data/02_serp_results.csv")
ENG_PATH     = Path("/home/christian/url-studie/data/03_dataset_engineered.csv")
CLEAN_PATH   = Path("/home/christian/url-studie/data/04_dataset_clean.csv")

# ─── Deutsche Stoppwörter im URL-Pfad ────────────────────────────────────────
DE_STOPWORDS = {
    'und','der','die','das','fuer','mit','von','auf','bei','wie',
    'ein','eine','ist','im','in','an','am','zum','zur','den','dem',
    'des','aus','nach','uber','ueber','als','auch','sich','hat','war',
    'werden','wurde','wenn','oder','aber','noch','mehr','nur','schon'
}

# ─── Seitentyp-Heuristik ─────────────────────────────────────────────────────
def classify_page_type(path: str, domain: str) -> str:
    if not path or path in ('', '/', ''):
        return 'homepage'
    p = path.lower().strip('/')
    # Homepage
    if p == '':
        return 'homepage'
    # Datumsmuster → Blog
    if re.search(r'\d{4}/\d{2}', p):
        return 'blogpost'
    # Blog-Muster
    if any(seg in p for seg in ['/blog/', '/magazin/', '/ratgeber/', '/news/', '/artikel/',
                                  '/tipps/', '/guide/', '/post/', '/beitrag/']):
        return 'blogpost'
    if p.startswith(('blog/', 'magazin/', 'ratgeber/', 'news/', 'artikel/', 'tipps/')):
        return 'blogpost'
    # Tool/Rechner
    if any(seg in p for seg in ['/rechner/', '/tool/', '/calculator/', '/finder/']):
        return 'tool'
    if p.startswith(('rechner/', 'tool/', 'calculator/')):
        return 'tool'
    # Produkt
    if any(seg in p for seg in ['/produkt/', '/product/', '/p/', '/item/', '/artikel/']):
        return 'product'
    if re.search(r'/(p|item)/\w', p):
        return 'product'
    # Pfadtiefe 1, kurz → wahrscheinlich Kategorie oder Landingpage
    depth = len([s for s in p.split('/') if s])
    if depth == 1:
        slug = p.strip('/')
        word_count = len(slug.split('-'))
        if word_count <= 2:
            return 'category'
        return 'landingpage'
    if depth == 2:
        return 'landingpage'
    # Tief verschachtelt → Blog oder Kategorie
    return 'blogpost'

# ─── Brand-Domain-Heuristik via TLD + Struktur ───────────────────────────────
# Wird durch domain_rank_decile ersetzt — kein binäres Brand-Flag mehr

def compute_domain_rank_decile(domain_rank: float) -> int:
    """10 = stärkste Domains (Rank >900), 1 = schwächste."""
    if pd.isna(domain_rank) or domain_rank == 0:
        return 1
    rank = float(domain_rank)
    if rank >= 900: return 10
    if rank >= 800: return 9
    if rank >= 700: return 8
    if rank >= 600: return 7
    if rank >= 500: return 6
    if rank >= 400: return 5
    if rank >= 300: return 4
    if rank >= 200: return 3
    if rank >= 100: return 2
    return 1

def keyword_in_url(keyword: str, path: str) -> int:
    """0=kein Match, 1=Partial, 2=Exakt (alle Keyword-Wörter im Pfad)."""
    if not keyword or not path:
        return 0
    kw_words = set(re.sub(r'[^a-z0-9äöüß ]', '', keyword.lower()).split())
    path_words = set(re.split(r'[-/_ ]', path.lower()))
    if not kw_words:
        return 0
    if kw_words <= path_words:
        return 2
    if kw_words & path_words:
        return 1
    return 0

def classify_url_length(path_length) -> str:
    if pd.isna(path_length): return None
    pl = int(path_length)
    if pl <= 30:  return 'kurz'
    if pl <= 60:  return 'optimal'
    if pl <= 80:  return 'mittel'
    if pl <= 100: return 'lang'
    return 'ueberlang'

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"PHASE 3: Feature Engineering ({datetime.now():%Y-%m-%d %H:%M})")
    print(f"{'='*60}")

    df = pd.read_csv(INPUT_PATH)
    print(f"Eingabe: {len(df):,} Rows, {df['keyword'].nunique():,} Keywords")

    # ── Nur Top-10 Positionen behalten ────────────────────────────────────────
    n_before = len(df)
    df = df[df['position'] <= 10].copy()
    print(f"Filter Top-10: {n_before:,} → {len(df):,} Rows")

    # ── URL-Normalisierung (falls noch nicht in Phase 2 gemacht) ──────────────
    if 'path_normalized' not in df.columns:
        df['path_normalized'] = df['url'].apply(
            lambda u: urllib.parse.urlparse(str(u)).path.rstrip('/').lower() if pd.notna(u) else ''
        )

    # ── Fehlende URL-Features nachberechnen ───────────────────────────────────
    def safe_parse(url):
        try:
            p = urllib.parse.urlparse(str(url))
            path = p.path.rstrip('/').lower()
            domain = p.netloc.lower().replace('www.', '')
            return {
                'path_normalized':  path,
                'path_length':      len(path),
                'domain_length':    len(domain),
                'url_length_total': len(str(url).split('?')[0]),
                'path_depth':       len([s for s in path.split('/') if s]),
                'word_count_path':  len([w for w in re.split(r'[-/]', path) if w]),
                'has_date_pattern': bool(re.search(r'/\d{4}/', path)),
                'has_numbers':      bool(re.search(r'\d', path)),
                'has_parameters':   bool(p.query),
                'uses_hyphens':     '-' in path,
                'is_subdomain':     bool(p.netloc and
                                        not p.netloc.lower().startswith('www.') and
                                        p.netloc.count('.') > 1),
                'tld_type':         domain.split('.')[-1] if '.' in domain else 'other',
                'domain_clean':     domain,
            }
        except:
            return {}

    print("Berechne URL-Features...")
    url_features = df['url'].apply(lambda u: pd.Series(safe_parse(u)))
    for col in url_features.columns:
        if col not in df.columns or df[col].isna().all():
            df[col] = url_features[col]

    # ── Stoppwörter im Pfad ───────────────────────────────────────────────────
    print("Berechne Stoppwörter...")
    df['has_stopwords'] = df['path_normalized'].apply(
        lambda p: int(any(w in DE_STOPWORDS for w in re.split(r'[-/]', str(p).lower()) if w))
    )

    # ── Seitentyp-Klassifikation ──────────────────────────────────────────────
    print("Klassifiziere Seitentypen...")
    df['page_type'] = df.apply(
        lambda r: classify_page_type(str(r.get('path_normalized', '')), str(r.get('domain', ''))),
        axis=1
    )

    # ── Domain Rank Dezil ─────────────────────────────────────────────────────
    df['domain_rank_decile'] = df['domain_rank'].apply(compute_domain_rank_decile)

    # ── Keyword-im-URL ────────────────────────────────────────────────────────
    print("Berechne Keyword-Match...")
    df['keyword_in_url'] = df.apply(
        lambda r: keyword_in_url(str(r['keyword']), str(r.get('path_normalized', ''))),
        axis=1
    )

    # ── URL-Längen-Klasse ─────────────────────────────────────────────────────
    df['url_length_class'] = df['path_length'].apply(classify_url_length)

    # ── YMYL sicherstellen ────────────────────────────────────────────────────
    if 'is_ymyl' not in df.columns:
        ymyl_verticals = {'finanzen', 'gesundheit', 'gambling'}
        df['is_ymyl'] = df['vertical'].apply(
            lambda v: 1 if str(v).lower() in ymyl_verticals else 0
        )

    # ── Engineered Dataset speichern ──────────────────────────────────────────
    ENG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ENG_PATH, index=False, encoding='utf-8-sig')
    print(f"Engineered Dataset: {len(df):,} Rows → {ENG_PATH.name}")

    # ─── Phase 4: Datenbereinigung ────────────────────────────────────────────
    print("\n--- Datenbereinigung ---")

    # Duplikate: gleiche URL+Keyword-Kombination entfernen
    n = len(df)
    df_clean = df.drop_duplicates(subset=['keyword', 'url']).copy()
    print(f"Duplikate entfernt (URL+Keyword): {n:,} → {len(df_clean):,}")

    # Ausreißer: path_length > 300
    n = len(df_clean)
    df_clean = df_clean[df_clean['path_length'] <= 300]
    print(f"Ausreißer (path_length > 300): {n:,} → {len(df_clean):,}")

    # Missing domain_rank → 0 (bereits als Dezil behandelt)
    df_clean['domain_rank'] = df_clean['domain_rank'].fillna(0)

    # Nur Positionen 1-10
    df_clean = df_clean[df_clean['position'].between(1, 10)]
    print(f"Nach Positions-Filter (1-10): {len(df_clean):,} Rows")

    df_clean.to_csv(CLEAN_PATH, index=False, encoding='utf-8-sig')

    # ─── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 3 + 4 ABGESCHLOSSEN")
    print(f"{'='*60}")
    print(f"Sauberer Datensatz: {len(df_clean):,} Rows, {df_clean['keyword'].nunique():,} Keywords")

    print(f"\nSeitentyp-Verteilung:")
    print(df_clean['page_type'].value_counts().to_string())

    print(f"\nURL-Längen-Klassen:")
    print(df_clean['url_length_class'].value_counts().to_string())

    print(f"\nMedian Pfadlänge pro Seitentyp:")
    print(df_clean.groupby('page_type')['path_length'].median().sort_values().to_string())

    print(f"\nMedian Pfadlänge pro Position (1-10):")
    print(df_clean.groupby('position')['path_length'].median().to_string())

    print(f"\nYMYL-Verteilung:")
    print(df_clean.groupby('is_ymyl').size().to_string())

    print(f"\nDomain-Rank-Dezil Verteilung:")
    print(df_clean['domain_rank_decile'].value_counts().sort_index().to_string())

    print(f"\nKeyword-im-URL (0=kein, 1=partial, 2=exakt):")
    print(df_clean['keyword_in_url'].value_counts().sort_index().to_string())

    print(f"\nSpearman-Korrelation path_length ↔ position (bivariate):")
    from scipy import stats
    r, p = stats.spearmanr(df_clean['path_length'], df_clean['position'])
    print(f"  ρ = {r:.4f}, p = {p:.4e}")

if __name__ == '__main__':
    main()
