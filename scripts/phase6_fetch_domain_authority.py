"""
Phase 6: Externe Domain Authority via DataForSEO domain_overview
Output: data/05_domain_authority.csv

Pro Domain:
  - organic_keywords_count  (Anzahl org. Keywords im DE-Markt)
  - organic_etv             (Estimated Traffic Value)
  - paid_keywords_count
  - paid_etv
  - timestamp

Re-runs sind idempotent: bereits abgerufene Domains werden uebersprungen.
"""

import os, sys, json, time, base64
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Credentials aus Umgebungsvariablen
# Account: https://app.dataforseo.com/
LOGIN = os.environ.get("DATAFORSEO_LOGIN")
PASSWORD = os.environ.get("DATAFORSEO_PASSWORD")
if not LOGIN or not PASSWORD:
    raise RuntimeError("Bitte DATAFORSEO_LOGIN und DATAFORSEO_PASSWORD als Umgebungsvariablen setzen.")
AUTH = base64.b64encode(f"{LOGIN}:{PASSWORD}".encode()).decode()
HEADERS = {"Authorization": f"Basic {AUTH}", "Content-Type": "application/json"}

ENDPOINT = "https://api.dataforseo.com/v3/dataforseo_labs/google/domain_rank_overview/live"

OUT_PATH = Path("/home/christian/url-studie/data/05_domain_authority.csv")
CLEAN_PATH = Path("/home/christian/url-studie/data/04_dataset_clean.csv")


def fetch_one(domain: str, retries: int = 3) -> dict:
    """Holt domain_overview fuer EINE Domain. Retry bei 429/5xx."""
    payload = [{
        "target": domain,
        "location_name": "Germany",
        "language_code": "de",
    }]
    for attempt in range(retries):
        try:
            r = requests.post(ENDPOINT, headers=HEADERS, json=payload, timeout=30)
            if r.status_code == 200:
                data = r.json()
                tasks = data.get("tasks", [])
                if tasks and tasks[0].get("status_code") == 20000:
                    result = tasks[0].get("result", [{}])[0] or {}
                    items = result.get("items", [{}])
                    item = items[0] if items else {}
                    metrics = item.get("metrics", {})
                    organic = metrics.get("organic", {}) or {}
                    paid = metrics.get("paid", {}) or {}
                    return {
                        "domain": domain,
                        "organic_keywords_count": organic.get("count"),
                        "organic_etv": organic.get("etv"),
                        "organic_pos_1": organic.get("pos_1"),
                        "organic_pos_2_3": organic.get("pos_2_3"),
                        "organic_pos_4_10": organic.get("pos_4_10"),
                        "paid_keywords_count": paid.get("count"),
                        "paid_etv": paid.get("etv"),
                        "fetched_at": datetime.utcnow().isoformat(),
                        "status": "ok",
                    }
                else:
                    code = tasks[0].get("status_code") if tasks else None
                    msg = tasks[0].get("status_message") if tasks else None
                    return {"domain": domain, "status": f"api_err_{code}", "error": str(msg)}
            elif r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            else:
                return {"domain": domain, "status": f"http_{r.status_code}"}
        except Exception as e:
            if attempt == retries - 1:
                return {"domain": domain, "status": "exception", "error": str(e)}
            time.sleep(2 ** attempt)
    return {"domain": domain, "status": "max_retries"}


def main():
    # Eindeutige Domains aus Clean-Dataset
    clean = pd.read_csv(CLEAN_PATH, low_memory=False)
    domains = sorted(clean["domain_clean"].dropna().unique().tolist())
    print(f"Total unique domains: {len(domains):,}")

    # Bereits gefetchte ueberspringen
    done = set()
    if OUT_PATH.exists():
        prev = pd.read_csv(OUT_PATH)
        done = set(prev[prev["status"] == "ok"]["domain"].tolist())
        print(f"Bereits abgerufen: {len(done):,}")

    todo = [d for d in domains if d not in done]
    print(f"Zu fetchen: {len(todo):,}")

    if not todo:
        print("Alles bereits fertig.")
        return

    results = []
    if OUT_PATH.exists():
        results = pd.read_csv(OUT_PATH).to_dict("records")

    # Parallel fetchen (max 8 worker, schont API)
    start = time.time()
    with ThreadPoolExecutor(max_workers=8) as exec_:
        futures = {exec_.submit(fetch_one, d): d for d in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)

            # Periodisch speichern (alle 100)
            if i % 100 == 0 or i == len(todo):
                pd.DataFrame(results).to_csv(OUT_PATH, index=False)
                elapsed = time.time() - start
                rate = i / elapsed
                eta_s = (len(todo) - i) / rate
                ok = sum(1 for r in results[-i:] if r.get("status") == "ok")
                print(f"[{i:>4}/{len(todo)}]  ok={ok}  rate={rate:.1f}/s  eta={eta_s/60:.1f}min")

    # Final stats
    final = pd.read_csv(OUT_PATH)
    n_ok = (final["status"] == "ok").sum()
    n_err = (final["status"] != "ok").sum()
    print(f"\nFertig. OK: {n_ok:,}  Errors: {n_err:,}")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
