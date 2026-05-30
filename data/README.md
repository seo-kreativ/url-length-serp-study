# Data

This directory intentionally does **not** contain the row-level SERP data.

## Why

The raw dataset consists of individual Google SERP rows (keyword, position, URL,
domain, …) retrieved through the [DataForSEO](https://dataforseo.com) API. Their
Terms of Service do not explicitly permit redistributing row-level API output, so
to stay on the safe side this repository ships only:

- **Aggregated, derived results** — see [`/analysis`](../analysis) (JSON: regression
  models, robustness checks, YMYL segmentation, methodology comparison)
- **All analysis code** — see [`/scripts`](../scripts) (phase0–phase10)
- **All visualizations** — see [`/charts`](../charts) (PNG + SVG, created by the author)

These aggregated outputs are released under CC BY 4.0 (see [`/LICENSE`](../LICENSE)).

## Reproducing the dataset

The full pipeline is reproducible with your own DataForSEO credentials:

```bash
export DATAFORSEO_LOGIN="your_login@example.com"
export DATAFORSEO_PASSWORD="your_password"

cd ../scripts
python phase1_keyword_sourcing.py      # stratified keyword sample (9 verticals)
python phase2_serp_collection.py       # SERP collection (~$3-5 in credits)
python phase3_feature_engineering.py   # URL anatomy + page-type features
python phase6_fetch_domain_authority.py  # external authority (~$8-15 in credits)
python phase7_reanalysis.py            # re-analysis with external authority
python phase8_robustness.py            # robustness checks
python phase9_charts_v2.py             # visualizations
python phase10_methodology_upgrade.py  # cluster SE, ordered logit, splines
```

Re-collected SERPs will differ slightly from the published snapshot because of
SERP volatility (the study snapshot is May 26–27, 2026).

## Raw data policy

For licensing and compliance reasons, no row-level raw data is published. The public
release contains only aggregated results and analysis code. Raw data is not shared on
request. The full dataset is reproducible with your own DataForSEO credentials via the
pipeline above.

(DE) Aus Lizenz- und Compliance-Gründen werden keine zeilenweisen Rohdaten
veröffentlicht. Der öffentliche Release enthält nur aggregierte Ergebnisse und
Analysecode. Rohdaten werden nicht auf Anfrage geteilt.
