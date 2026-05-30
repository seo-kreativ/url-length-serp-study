"""
Phase 0: seo-kreativ.de GSC Keyword Export
Studie: Ranken kurze URLs wirklich besser?
Output: data/00_seo_kreativ_gsc_keywords.csv
"""

import sqlite3
import pandas as pd
import re
from pathlib import Path
from datetime import datetime

DB_PATH = "/mnt/c/Users/Christian/Desktop/complex-master/output/seo-kreativ.de/db/crawl.sqlite"
OUTPUT_PATH = Path("/home/christian/url-studie/data/00_seo_kreativ_gsc_keywords.csv")

# Navigational/Brand Queries ausschließen
BRAND_PATTERNS = [
    r'\bseo.?kreativ\b',
    r'\bseokreativ\b',
]

def is_single_word(query):
    return len(query.strip().split()) == 1

def is_brand_query(query):
    q_lower = query.lower()
    for pat in BRAND_PATTERNS:
        if re.search(pat, q_lower):
            return True
    return False

def parse_url_features(url):
    import urllib.parse
    try:
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.rstrip('/').lower()
        path_length = len(path)
        path_depth = len([s for s in path.split('/') if s])
        words_in_path = len([w for w in re.split(r'[-/]', path) if w])
        has_date = bool(re.search(r'/\d{4}/', path))
        has_numbers = bool(re.search(r'\d', path))
        has_parameters = bool(parsed.query)
        return path, path_length, path_depth, words_in_path, has_date, has_numbers, has_parameters
    except:
        return '', 0, 0, 0, False, False, False

def main():
    print(f"[{datetime.now():%H:%M:%S}] Phase 0: GSC Export starten...")

    con = sqlite3.connect(DB_PATH)

    date_info = pd.read_sql("""
        SELECT MIN(date) as min_date, MAX(date) as max_date, COUNT(DISTINCT date) as n_dates
        FROM gsc_performance
    """, con)
    print(f"Datumsbereich: {date_info['min_date'].iloc[0]} bis {date_info['max_date'].iloc[0]} ({date_info['n_dates'].iloc[0]} Tage)")

    print("Aggregiere GSC-Daten pro Query + URL...")
    df_raw = pd.read_sql("""
        SELECT
            query,
            url,
            SUM(impressions)   AS impressions_total,
            SUM(clicks)        AS clicks_total,
            SUM(impressions * position) / NULLIF(SUM(impressions), 0) AS avg_position,
            COUNT(DISTINCT date) AS days_with_data,
            MIN(date) AS first_seen,
            MAX(date) AS last_seen
        FROM gsc_performance
        GROUP BY query, url
    """, con)
    con.close()

    print(f"Roh: {len(df_raw):,} Query-URL-Kombinationen, {df_raw['query'].nunique():,} einzigartige Queries")

    min_date = pd.to_datetime(date_info['min_date'].iloc[0])
    max_date = pd.to_datetime(date_info['max_date'].iloc[0])
    n_months = max(1, (max_date - min_date).days / 30)
    df_raw['impressions_per_month'] = df_raw['impressions_total'] / n_months

    print(f"Beobachtungszeitraum: {n_months:.1f} Monate")

    # Pro Query: URL mit höchster Impressionszahl als primäre URL
    df_best = df_raw.sort_values('impressions_total', ascending=False)\
                    .groupby('query').first().reset_index()\
                    .rename(columns={'url': 'primary_url'})

    # Filter 1: ≥10 Impressionen/Monat
    n_before = len(df_best)
    df_filtered = df_best[df_best['impressions_per_month'] >= 10].copy()
    print(f"\nFilter ≥10 Imp/Monat: {n_before:,} → {len(df_filtered):,} Queries")

    # Filter 2: Keine Ein-Wort-Keywords
    n_before = len(df_filtered)
    df_filtered = df_filtered[~df_filtered['query'].apply(is_single_word)]
    print(f"Filter Ein-Wort: {n_before:,} → {len(df_filtered):,} Queries")

    # Filter 3: Keine Brand/Navigational Queries
    n_before = len(df_filtered)
    df_filtered = df_filtered[~df_filtered['query'].apply(is_brand_query)]
    print(f"Filter Brand: {n_before:,} → {len(df_filtered):,} Queries")

    # Index resetten VOR dem Feature-Engineering (verhindert concat-Misalignment)
    df_filtered = df_filtered.reset_index(drop=True)

    # URL-Features vorberechnen
    features = df_filtered['primary_url'].apply(
        lambda u: pd.Series(parse_url_features(u),
                            index=['path_normalized', 'sk_path_length', 'sk_path_depth',
                                   'sk_word_count_path', 'sk_has_date', 'sk_has_numbers', 'sk_has_parameters'])
    )
    df_filtered = pd.concat([df_filtered, features], axis=1)

    df_out = df_filtered[[
        'query', 'primary_url', 'path_normalized',
        'avg_position', 'impressions_total', 'impressions_per_month',
        'clicks_total', 'days_with_data', 'first_seen', 'last_seen',
        'sk_path_length', 'sk_path_depth', 'sk_word_count_path',
        'sk_has_date', 'sk_has_numbers', 'sk_has_parameters'
    ]].sort_values('impressions_per_month', ascending=False)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

    print(f"\n{'='*60}")
    print(f"PHASE 0 ABGESCHLOSSEN")
    print(f"{'='*60}")
    print(f"Keywords im Benchmark-Sample: {len(df_out)}")
    print(f"Gespeichert: {OUTPUT_PATH}")
    print(f"\nTop 15 Keywords nach Impressionen/Monat:")
    print(df_out[['query', 'avg_position', 'impressions_per_month', 'sk_path_length']]\
          .head(15).to_string(index=False))
    print(f"\nURL-Längen-Verteilung (seo-kreativ.de URLs):")
    bins = [0, 30, 60, 80, 100, 9999]
    labels = ['Kurz (<=30)', 'Optimal (31-60)', 'Mittel (61-80)', 'Lang (81-100)', 'Ueberlang (>100)']
    df_out['url_klasse'] = pd.cut(df_out['sk_path_length'], bins=bins, labels=labels)
    print(df_out['url_klasse'].value_counts().sort_index().to_string())
    print(f"\nMedian Pfad-Laenge seo-kreativ.de: {df_out['sk_path_length'].median():.0f} Zeichen")
    print(f"Median Avg-Position: {df_out['avg_position'].median():.1f}")

if __name__ == '__main__':
    main()
