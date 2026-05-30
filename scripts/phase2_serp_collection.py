"""
Phase 2: SERP-Datenerhebung
Studie: Ranken kurze URLs wirklich besser?

3.000 Keywords × Top 10 = 30.000 Datenpunkte
API: DataForSEO SERP Google Organic Live Advanced
Kosten: ~$0.0006 pro Keyword × 2866 = ~$1.72

Output: data/02_serp_raw_results.json   (Rohdaten)
        data/02_serp_results.csv        (normalisiert, ein Row pro URL)

Erhebungszeitraum: maximal 48h (laut Studiendesign)
collection_timestamp wird pro Request geloggt.
"""

import os, sys, time, json, base64, re
import requests
import pandas as pd
import urllib.parse
from pathlib import Path
from datetime import datetime, timezone

DATAFORSEO_LOGIN    = os.getenv("DATAFORSEO_LOGIN")
DATAFORSEO_PASSWORD = os.getenv("DATAFORSEO_PASSWORD")
LOCATION_CODE       = 2276
LANGUAGE_CODE       = "de"

KEYWORDS_PATH = Path("/home/christian/url-studie/data/01_keyword_sample_3000.csv")
RAW_JSON_PATH = Path("/home/christian/url-studie/data/02_serp_raw_results.json")
CSV_PATH      = Path("/home/christian/url-studie/data/02_serp_results.csv")

BATCH_SIZE    = 1    # live/advanced Endpoint: 1 Task pro Request
DELAY         = 0.4  # Sekunden zwischen Batches
MAX_RESULTS   = 10   # Top-10 Ergebnisse

# ─── DataForSEO ───────────────────────────────────────────────────────────────
def dfs_headers() -> dict:
    creds = base64.b64encode(f"{DATAFORSEO_LOGIN}:{DATAFORSEO_PASSWORD}".encode()).decode()
    return {"Authorization": f"Basic {creds}", "Content-Type": "application/json"}

def fetch_serp_batch(keywords: list[str]) -> list[dict]:
    """SERP Top-10 für einen Batch von Keywords."""
    payload = [{
        "keyword":       kw,
        "location_code": LOCATION_CODE,
        "language_code": LANGUAGE_CODE,
        "device":        "desktop",
        "os":            "windows",
        "depth":         MAX_RESULTS,
        "se_type":       "organic",
        "calculate_rectangles": False,
    } for kw in keywords]

    for attempt in range(3):
        try:
            r = requests.post(
                "https://api.dataforseo.com/v3/serp/google/organic/live/advanced",
                headers=dfs_headers(),
                json=payload,
                timeout=90
            )
            r.raise_for_status()
            return r.json().get("tasks", []) or []
        except requests.exceptions.Timeout:
            print(f"    Timeout (Versuch {attempt+1}/3), warte 10s...")
            time.sleep(10)
        except Exception as e:
            print(f"    API Fehler (Versuch {attempt+1}/3): {e}")
            time.sleep(5)
    return []

# ─── URL-Parsing ──────────────────────────────────────────────────────────────
def parse_url(url: str) -> dict:
    try:
        p = urllib.parse.urlparse(url)
        path = p.path.rstrip('/').lower()
        domain = p.netloc.lower().replace('www.', '')
        return {
            "path_normalized": path,
            "path_length":     len(path),
            "domain_length":   len(domain),
            "url_length_total": len(url.split('?')[0]),
            "path_depth":      len([s for s in path.split('/') if s]),
            "word_count_path": len([w for w in re.split(r'[-/]', path) if w]),
            "has_date_pattern": bool(re.search(r'/\d{4}/', path)),
            "has_numbers":      bool(re.search(r'\d', path)),
            "has_parameters":   bool(p.query),
            "uses_hyphens":     '-' in path,
            "has_stopwords":    any(w in path for w in ['und', 'der', 'die', 'das', 'fuer', 'mit', 'von', 'auf', 'bei', 'wie']),
            "is_subdomain":     bool(p.netloc and not p.netloc.lower().startswith('www.') and '.' in p.netloc.replace('www.', '')),
            "tld_type":         domain.split('.')[-1] if '.' in domain else 'other',
        }
    except:
        return {k: None for k in ['path_normalized','path_length','domain_length','url_length_total',
                                   'path_depth','word_count_path','has_date_pattern','has_numbers',
                                   'has_parameters','uses_hyphens','has_stopwords','is_subdomain','tld_type']}

def classify_url_length(path_length) -> str:
    if path_length is None: return None
    if path_length <= 30:   return "kurz"
    if path_length <= 60:   return "optimal"
    if path_length <= 80:   return "mittel"
    if path_length <= 100:  return "lang"
    return "ueberlang"

def keyword_in_url(keyword: str, path: str) -> int:
    """0=kein Match, 1=Partial Match, 2=Exakter Match (alle Wörter)"""
    if not keyword or not path: return 0
    kw_words = set(keyword.lower().split())
    path_words = set(re.split(r'[-/]', path.lower()))
    if kw_words <= path_words: return 2  # exakt
    if kw_words & path_words:  return 1  # partial
    return 0

# ─── SERP-Ergebnisse normalisieren ────────────────────────────────────────────
def normalize_serp_tasks(tasks: list[dict], kw_meta: dict, timestamp: str) -> list[dict]:
    """Extrahiert alle organischen Ergebnisse aus SERP-Tasks."""
    rows = []
    for task in tasks:
        if task.get("status_code") != 20000:
            continue
        result = (task.get("result") or [None])[0]
        if not result:
            continue

        keyword = task.get("data", {}).get("keyword", "")
        meta    = kw_meta.get(keyword, {})

        # SERP-Features zählen
        items = result.get("items", []) or []
        serp_features = [it for it in items if it.get("type") != "organic"]
        has_featured_snippet = any(it.get("type") == "featured_snippet" for it in items)

        organic_items = [it for it in items if it.get("type") == "organic"]

        for item in organic_items:
            pos  = item.get("rank_absolute", 99)
            url  = item.get("url", "")
            domain_info = item.get("domain", "")
            domain_rank = item.get("domain_parameters", {}).get("domain_rank", 0) if item.get("domain_parameters") else 0

            url_features = parse_url(url)
            kw_in_url    = keyword_in_url(keyword, url_features.get("path_normalized", ""))

            rows.append({
                "keyword":          keyword,
                "position":         pos,
                "url":              url,
                "domain":           domain_info,
                "domain_rank":      domain_rank or 0,
                "collection_timestamp": timestamp,
                # Keyword-Metadaten
                "vertical":         meta.get("vertical", ""),
                "is_ymyl":          meta.get("is_ymyl", 0),
                "search_volume":    meta.get("search_volume", 0),
                "keyword_difficulty": meta.get("keyword_difficulty", 0),
                "search_intent":    meta.get("search_intent", ""),
                "volume_class":     meta.get("volume_class", ""),
                "is_seo_kreativ":   meta.get("is_seo_kreativ", 0),
                # URL-Features
                **url_features,
                "url_length_class": classify_url_length(url_features.get("path_length")),
                "keyword_in_url":   kw_in_url,
                # SERP-Kontext
                "serp_feature_count":    len(serp_features),
                "has_featured_snippet":  int(has_featured_snippet),
                "result_type":           "organic",
            })

    return rows

# ─── Fortschritt laden (Resume) ───────────────────────────────────────────────
def load_done_keywords(csv_path: Path) -> set:
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, usecols=['keyword'])
            return set(df['keyword'].unique())
        except:
            return set()
    return set()

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"PHASE 2: SERP-Erhebung ({datetime.now():%Y-%m-%d %H:%M})")
    print(f"{'='*60}")

    df_kw = pd.read_csv(KEYWORDS_PATH)
    print(f"Keywords gesamt: {len(df_kw)}")

    # kw_meta als Lookup-Dict
    kw_meta = {
        row['keyword']: {
            'vertical':           row.get('vertical', ''),
            'is_ymyl':            row.get('is_ymyl', 0),
            'search_volume':      row.get('search_volume', 0),
            'keyword_difficulty': row.get('keyword_difficulty', 0),
            'search_intent':      row.get('search_intent', ''),
            'volume_class':       row.get('volume_class', ''),
            'is_seo_kreativ':     row.get('is_seo_kreativ', 0),
        }
        for _, row in df_kw.iterrows()
    }

    keywords = df_kw['keyword'].tolist()

    # Resume: bereits geholte Keywords überspringen
    done = load_done_keywords(CSV_PATH)
    remaining = [kw for kw in keywords if kw not in done]
    print(f"Bereits erhoben: {len(done)} | Verbleibend: {len(remaining)}")

    if not remaining:
        print("Alle Keywords bereits erhoben!")
        return

    all_rows  = []
    raw_tasks = []
    n_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    errors    = 0
    start_ts  = datetime.now(timezone.utc)

    print(f"Starte Erhebung: {len(remaining)} Keywords in {n_batches} Batches...")
    print(f"Geschätzte Dauer: {n_batches * DELAY / 60:.1f} Minuten")

    for batch_idx in range(n_batches):
        batch_kws = remaining[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
        timestamp = datetime.now(timezone.utc).isoformat()

        tasks = fetch_serp_batch(batch_kws)
        raw_tasks.extend(tasks or [])
        rows = normalize_serp_tasks(tasks, kw_meta, timestamp)
        all_rows.extend(rows)

        if (batch_idx + 1) % 50 == 0 or batch_idx == n_batches - 1:
            # Zwischenspeichern: immer an bestehende CSV anhängen
            df_new = pd.DataFrame(all_rows)
            if CSV_PATH.exists():
                df_existing = pd.read_csv(CSV_PATH)
                df_combined = pd.concat([df_existing, df_new]).drop_duplicates(subset=['keyword', 'position'])
            else:
                df_combined = df_new
            df_combined.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
            elapsed = (datetime.now(timezone.utc) - start_ts).seconds / 60
            print(f"  Batch {batch_idx+1}/{n_batches} | gesamt {len(df_combined)} Rows | {elapsed:.1f} min")
            all_rows = []  # Memory freigeben

        if len(tasks) == 0:
            errors += 1

        time.sleep(DELAY)

    # Raw JSON speichern (komprimiert)
    print(f"\nSpeichere Raw-JSON...")
    with open(RAW_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(raw_tasks[-1000:], f, ensure_ascii=False)  # Letzten 1000 Tasks als Sample

    # Finale Statistik
    df_final = pd.read_csv(CSV_PATH)
    elapsed_total = (datetime.now(timezone.utc) - start_ts).seconds / 60

    print(f"\n{'='*60}")
    print(f"PHASE 2 ABGESCHLOSSEN")
    print(f"{'='*60}")
    print(f"Keywords erhoben: {df_final['keyword'].nunique()}")
    print(f"Organische Ergebnisse: {len(df_final)}")
    print(f"API-Fehler: {errors}")
    print(f"Dauer: {elapsed_total:.1f} Minuten")
    print(f"Gespeichert: {CSV_PATH}")
    print(f"\nPositions-Verteilung:")
    print(df_final['position'].value_counts().sort_index().head(10).to_string())
    print(f"\nURL-Längen-Verteilung (Klassen):")
    print(df_final['url_length_class'].value_counts().to_string())
    print(f"\nMedian Pfad-Länge pro Position:")
    print(df_final.groupby('position')['path_length'].median().to_string())

if __name__ == '__main__':
    main()
