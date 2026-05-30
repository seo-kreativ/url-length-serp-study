"""
Phase 1: Keyword-Sourcing & Kuratierung
Studie: Ranken kurze URLs wirklich besser?

Stratifizierte Stichprobe: 9 Verticals × ~320 Keywords
Intent: Informational 30% / Commercial 25% / Transactional 25% / Local 20%
Volume: Low 25% / Medium 25% / High 25% / Very High 25%
YMYL: Finanzen, Gesundheit, Gambling = 1

Output: data/01_keyword_sample_3000.csv
Kosten: ~$1.80 (Labs API keyword_suggestions, ~15.000 Ergebnisse)
"""

import os, sys, time, json, base64, re
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ─── Config ──────────────────────────────────────────────────────────────────
DATAFORSEO_LOGIN    = os.getenv("DATAFORSEO_LOGIN")
DATAFORSEO_PASSWORD = os.getenv("DATAFORSEO_PASSWORD")
LOCATION_CODE       = 2276   # Deutschland
LANGUAGE_CODE       = "de"
OUTPUT_PATH         = Path("/home/christian/url-studie/data/01_keyword_sample_3000.csv")
GSC_PATH            = Path("/home/christian/url-studie/data/00_seo_kreativ_gsc_keywords.csv")

TARGET_TOTAL        = 3000
TARGET_PER_VERTICAL = 320

# ─── Seeds (9 Verticals) ─────────────────────────────────────────────────────
VERTICALS = {
    "ecommerce": {
        "ymyl": False,
        "seeds": [
            "schuhe online kaufen", "möbel günstig", "elektronik versandkostenfrei",
            "kleidung sale", "haushaltsgeräte kaufen", "sportbedarf online",
            "bücher kaufen", "spielzeug günstig", "gartengeräte kaufen", "drogerie online"
        ]
    },
    "finanzen": {
        "ymyl": True,
        "seeds": [
            "etf sparplan vergleich", "kfz versicherung wechseln", "tagesgeld zinsen vergleich",
            "kredit zinsen", "aktien kaufen anfänger", "riester rente",
            "hausratversicherung", "privatkredit vergleich", "geldanlage tipps", "depot eröffnen"
        ]
    },
    "gesundheit": {
        "ymyl": True,
        "seeds": [
            "kopfschmerzen ursachen", "hausarzt online termin", "erkältung symptome",
            "rückenübungen zuhause", "blutdruck senken natürlich", "vitamin d mangel symptome",
            "gesunde ernährung plan", "sport nach operation", "schmerzmittel wirkung vergleich", "psychotherapie kosten"
        ]
    },
    "technologie": {
        "ymyl": False,
        "seeds": [
            "vpn vergleich", "password manager test", "antivirus software kostenlos",
            "cloud storage vergleich", "windows 11 optimieren", "laptop test 2025",
            "browser vergleich sicherheit", "online backup lösung", "software kaufen günstig", "handy reparatur kosten"
        ]
    },
    "reise": {
        "ymyl": False,
        "seeds": [
            "hotel mallorca günstig", "wanderurlaub alpen", "flug günstig buchen tipps",
            "mietwagen vergleich urlaub", "campen deutschland plätze", "städtereise tipps europa",
            "reiserücktrittsversicherung vergleich", "lastminute urlaub angebote", "ferienhaus mieten ostsee", "backpacking europa route"
        ]
    },
    "bildung": {
        "ymyl": False,
        "seeds": [
            "python lernen anfänger", "bewerbung schreiben tipps", "online kurs vergleich plattform",
            "englisch lernen schnell", "präsentation tipps beruf", "lebenslauf vorlage kostenlos",
            "fortbildung kosten absetzen", "master studium berufsbegleitend", "zertifikate online kostenlos", "e-learning plattform vergleich"
        ]
    },
    "haus_garten": {
        "ymyl": False,
        "seeds": [
            "fliesen verlegen anleitung", "rasen mähen wann richtig", "küche renovieren kosten",
            "badezimmer sanieren tipps", "holz streichen außen anleitung", "dachdämmung kosten",
            "wasserhahn wechseln anleitung", "laminat verlegen selber", "garten gestalten ideen", "kompost anlegen anleitung"
        ]
    },
    "recht": {
        "ymyl": False,
        "seeds": [
            "mietvertrag kündigung frist", "abmahnung erhalten was tun", "scheidung kosten anwalt",
            "erbrecht ohne testament", "arbeitsvertrag prüfen lassen", "verbraucherschutz reklamation",
            "steuererklärung selber machen tipps", "datenschutz dsgvo unternehmen", "bußgeld einspruch einlegen", "anwalt erstberatung kostenlos"
        ]
    },
    "gambling": {
        "ymyl": True,
        "seeds": [
            "online casino deutschland legal", "sportwetten vergleich",
            "casino bonus ohne einzahlung", "slots online spielen",
            "poker online echtgeld", "glücksspiel lizenz deutschland",
            "verantwortungsvolles spielen", "live casino anbieter",
            "wettanbieter test vergleich", "freispiele ohne einzahlung",
            "spielautomaten online legal", "spielhalle online legal deutschland",
            "lotto online spielen seriös", "glücksspiel automaten erfahrungen"
        ]
    }
}

# ─── Ausschlusskriterien ──────────────────────────────────────────────────────
BRAND_PATTERNS = [r'\bseo.?kreativ\b', r'\bseokreativ\b']

def is_excluded(kw: str) -> bool:
    kw_lower = kw.lower().strip()
    if len(kw_lower.split()) == 1:          # Ein-Wort
        return True
    for pat in BRAND_PATTERNS:              # Brand
        if re.search(pat, kw_lower):
            return True
    return False

# ─── Volume-Klassifikation ────────────────────────────────────────────────────
def classify_volume(sv: float) -> str:
    if sv < 100:   return "very_low"
    if sv < 500:   return "low"
    if sv < 2000:  return "medium"
    if sv < 10000: return "high"
    return "very_high"

# ─── Intent-Mapping (DataForSEO liefert: informational, commercial, transactional, navigational) ──
def normalize_intent(intent_str: str | None) -> str:
    if not intent_str:
        return "informational"
    i = intent_str.lower()
    if "transactional" in i: return "transactional"
    if "commercial"    in i: return "commercial"
    if "navigational"  in i: return "navigational"
    return "informational"

# ─── DataForSEO API ───────────────────────────────────────────────────────────
def dfs_headers() -> dict:
    creds = base64.b64encode(f"{DATAFORSEO_LOGIN}:{DATAFORSEO_PASSWORD}".encode()).decode()
    return {"Authorization": f"Basic {creds}", "Content-Type": "application/json"}

def fetch_keyword_suggestions(seed: str, max_results: int = 200) -> list[dict]:
    """DataForSEO Labs: Keyword Suggestions für einen Seed."""
    payload = [{
        "keyword": seed,
        "language_code": LANGUAGE_CODE,
        "location_code": LOCATION_CODE,
        "limit": max_results,
        "include_seed_keyword": True,
    }]
    try:
        r = requests.post(
            "https://api.dataforseo.com/v3/dataforseo_labs/google/keyword_suggestions/live",
            headers=dfs_headers(),
            json=payload,
            timeout=60
        )
        r.raise_for_status()
        data = r.json()
        tasks = data.get("tasks", [])
        if not tasks or tasks[0].get("status_code") != 20000:
            err = tasks[0].get("status_message", "unknown") if tasks else "no tasks"
            print(f"  API Fehler für '{seed}': {err}")
            return []
        result = tasks[0].get("result") or [{}]
        items = (result[0] or {}).get("items") or []
        return items
    except Exception as e:
        print(f"  Exception für '{seed}': {e}")
        return []

def fetch_search_volume(keywords: list[str]) -> dict[str, dict]:
    """DataForSEO Labs: Suchvolumen + KD für bis zu 700 Keywords."""
    results = {}
    batch_size = 700
    for i in range(0, len(keywords), batch_size):
        batch = keywords[i:i+batch_size]
        payload = [{
            "keywords": batch,
            "language_code": LANGUAGE_CODE,
            "location_code": LOCATION_CODE,
        }]
        try:
            r = requests.post(
                "https://api.dataforseo.com/v3/dataforseo_labs/google/search_intent/live",
                headers=dfs_headers(),
                json=payload,
                timeout=60
            )
            # Falls search_intent fehlschlägt, nur Volume/KD holen
        except:
            pass

        try:
            r = requests.post(
                "https://api.dataforseo.com/v3/keywords_data/google_ads/search_volume/live",
                headers=dfs_headers(),
                json=[{"keywords": batch, "language_code": LANGUAGE_CODE, "location_code": LOCATION_CODE}],
                timeout=60
            )
            r.raise_for_status()
            data = r.json()
            items = data.get("tasks", [{}])[0].get("result", [])
            for item in (items or []):
                kw = item.get("keyword", "")
                results[kw] = {
                    "search_volume": item.get("search_volume", 0) or 0,
                    "competition": item.get("competition", 0) or 0,
                    "cpc": item.get("cpc", 0) or 0,
                }
        except Exception as e:
            print(f"  Volume-Fetch Fehler batch {i}: {e}")
        time.sleep(0.5)
    return results

# ─── Stratifiziertes Sampling ──────────────────────────────────────────────────
INTENT_TARGETS = {"informational": 0.30, "commercial": 0.25, "transactional": 0.25, "local": 0.20}
VOLUME_TARGETS = {"low": 0.25, "medium": 0.25, "high": 0.25, "very_high": 0.25}

def stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Stratifiziertes Sampling nach Intent × Volume.
    Füllt Quoten so weit wie möglich; fehlende Quoten werden proportional aufgefüllt.
    """
    if len(df) <= n:
        return df
    df = df.copy()
    selected = []
    # Versuche Quoten exakt zu erfüllen
    for intent, intent_frac in INTENT_TARGETS.items():
        intent_target = int(n * intent_frac)
        df_intent = df[df['search_intent'] == intent]
        for vol, vol_frac in VOLUME_TARGETS.items():
            vol_target = int(intent_target * vol_frac)
            subset = df_intent[df_intent['volume_class'] == vol]
            take = min(len(subset), vol_target)
            selected.append(subset.sample(take, random_state=42))
    sampled = pd.concat(selected).drop_duplicates(subset='keyword')
    # Auffüllen wenn Shortfall
    if len(sampled) < n:
        remaining = df[~df['keyword'].isin(sampled['keyword'])]
        shortfall = n - len(sampled)
        extra = remaining.sample(min(shortfall, len(remaining)), random_state=42)
        sampled = pd.concat([sampled, extra])
    return sampled.head(n)

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"PHASE 1: Keyword-Sourcing ({datetime.now():%Y-%m-%d %H:%M})")
    print(f"{'='*60}")

    # seo-kreativ.de Keywords laden (für Merge-Check)
    df_sk = pd.read_csv(GSC_PATH) if GSC_PATH.exists() else pd.DataFrame()
    sk_keywords = set(df_sk['query'].str.lower().tolist()) if len(df_sk) else set()
    n_sk = len(sk_keywords)
    target_general = TARGET_TOTAL - n_sk
    print(f"seo-kreativ.de Keywords: {n_sk}")
    print(f"Allgemeine Stichprobe Ziel: {target_general} Keywords")

    all_keywords = []

    for vertical_id, vertical_cfg in VERTICALS.items():
        seeds   = vertical_cfg["seeds"]
        is_ymyl = vertical_cfg["ymyl"]
        print(f"\n[{vertical_id.upper()}] {'(YMYL)' if is_ymyl else ''} — {len(seeds)} Seeds")

        vertical_pool = []
        for seed in seeds:
            print(f"  Seed: '{seed}'...", end=" ", flush=True)
            items = fetch_keyword_suggestions(seed, max_results=150)
            print(f"{len(items)} Keywords")

            for item in items:
                kw = item.get("keyword", "").strip().lower()
                if not kw or is_excluded(kw) or kw in sk_keywords:
                    continue
                sv = item.get("keyword_info", {}).get("search_volume", 0) or 0
                kd = item.get("keyword_info", {}).get("keyword_difficulty", 0) or 0
                intent_raw = ""
                # Intent aus search_intent_info
                si_info = item.get("search_intent_info", {})
                if si_info:
                    main_intent = si_info.get("main_intent", "")
                    intent_raw = main_intent
                intent = normalize_intent(intent_raw)

                vertical_pool.append({
                    "keyword":       kw,
                    "vertical":      vertical_id,
                    "is_ymyl":       int(is_ymyl),
                    "search_volume": sv,
                    "keyword_difficulty": kd,
                    "search_intent": intent,
                    "volume_class":  classify_volume(sv),
                    "seed_keyword":  seed,
                })
            time.sleep(0.3)  # Rate limiting

        # Deduplizierung innerhalb Vertical
        df_vertical = pd.DataFrame(vertical_pool).drop_duplicates(subset='keyword')
        print(f"  Pool nach Deduplizierung: {len(df_vertical)} Keywords")

        # Stratifiziertes Sampling
        if len(df_vertical) == 0:
            print(f"  WARNUNG: Kein Pool für {vertical_id} — übersprungen")
            continue
        sample = stratified_sample(df_vertical, TARGET_PER_VERTICAL)
        print(f"  Sample: {len(sample)} Keywords")
        if len(sample) > 0:
            print(f"  Intent: {sample['search_intent'].value_counts().to_dict()}")
            print(f"  Volume: {sample['volume_class'].value_counts().to_dict()}")

        all_keywords.append(sample)

    # Alle Verticals zusammenführen
    df_general = pd.concat(all_keywords).drop_duplicates(subset='keyword').reset_index(drop=True)
    print(f"\nAllgemeine Stichprobe gesamt: {len(df_general)} Keywords")

    # seo-kreativ.de Keywords hinzufügen
    if len(df_sk) > 0:
        df_sk_prepared = df_sk[['query']].rename(columns={'query': 'keyword'}).copy()
        df_sk_prepared['keyword'] = df_sk_prepared['keyword'].str.lower()
        df_sk_prepared['vertical']     = 'seo_kreativ'
        df_sk_prepared['is_ymyl']      = 0
        df_sk_prepared['search_volume'] = df_sk.get('impressions_per_month', pd.Series([0]*len(df_sk))).values
        df_sk_prepared['keyword_difficulty'] = 0
        df_sk_prepared['search_intent'] = 'informational'
        df_sk_prepared['volume_class']  = df_sk_prepared['search_volume'].apply(classify_volume)
        df_sk_prepared['seed_keyword']  = 'gsc_export'
        df_sk_prepared['is_seo_kreativ'] = 1
        df_general['is_seo_kreativ'] = 0
        df_all = pd.concat([df_general, df_sk_prepared]).drop_duplicates(subset='keyword').reset_index(drop=True)
    else:
        df_general['is_seo_kreativ'] = 0
        df_all = df_general

    total = len(df_all)
    print(f"\nGesamt-Sample: {total} Keywords")

    # Zusammenfassung
    print(f"\n{'='*60}")
    print("VERTEILUNG NACH VERTICAL:")
    print(df_all.groupby('vertical').size().to_string())
    print(f"\nVERTEILUNG NACH YMYL:")
    print(df_all.groupby('is_ymyl').size().to_string())
    print(f"\nVERTEILUNG NACH INTENT:")
    general = df_all[df_all['is_seo_kreativ'] == 0]
    print(general.groupby('search_intent').size().to_string())
    print(f"\nVERTEILUNG NACH VOLUME:")
    print(general.groupby('volume_class').size().to_string())

    # Speichern
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"\nGespeichert: {OUTPUT_PATH}")
    print(f"\nPHASE 1 ABGESCHLOSSEN — {total} Keywords bereit für Phase 2")

if __name__ == '__main__':
    main()
