# URL Length & Google Rankings — A 27,000-SERP Study

An independent analysis of whether shorter URLs rank better in google.de, based on 27,291 SERP
results across 2,862 keywords and 9 industries. The headline finding: after controlling for an
**external** domain-authority signal, the apparent URL-length advantage shrinks by ~36 % and stays
tiny — URL length is a micro-signal, not a primary ranking lever.

> Independent study, not affiliated with or endorsed by DataForSEO. Analysis and all
> visualizations created by the author. Aggregated, derived results only — see
> [Data availability](#data-availability).

Companion article (DE): <https://www.seo-kreativ.de/blog/url-laenge-ranking/>

---

## Reproducing the study

```bash
pip install -r requirements.txt

export DATAFORSEO_LOGIN="your_login@example.com"
export DATAFORSEO_PASSWORD="your_password"

cd scripts
python phase1_keyword_sourcing.py        # stratified keyword sample (9 verticals, n ≈ 2,862)
python phase2_serp_collection.py         # SERP collection via DataForSEO (~$3-5 in credits)
python phase3_feature_engineering.py     # URL anatomy + page-type features
python phase6_fetch_domain_authority.py  # external authority signal (~$8-15 in credits)
python phase7_reanalysis.py              # analysis with external authority
python phase8_robustness.py              # robustness: spec-curve, bootstrap, CV
python phase9_charts_v2.py               # visualizations
python phase10_methodology_upgrade.py    # cluster SE, ordered logit, cubic splines
```

`phase0`, `phase4`, `phase5` are earlier pipeline stages kept for transparency.

---

## Data availability

This repository ships:

- **Aggregated, derived results** — [`/analysis`](./analysis) (regression models, robustness
  checks, YMYL segmentation, methodology comparison, as JSON)
- **All analysis code** — [`/scripts`](./scripts) (`phase0`–`phase10`)
- **All visualizations** — [`/charts`](./charts) (PNG + SVG, created by the author)

It does **not** ship the row-level SERP data. Those rows (keyword, position, URL, domain) come
from the DataForSEO API. For licensing and compliance reasons, no row-level raw data is published —
and it is not shared on request either. The full dataset is reproducible with your own DataForSEO
credentials via the pipeline above. See [`/data/README.md`](./data/README.md) for details.

---

## Key findings (technical summary)

### Bivariate correlations (Spearman ρ vs. position)

| Variable | ρ | p | n |
|---|---|---|---|
| path_length | +0.0526 | < 0.001 | 22,250 |
| path_depth | +0.0296 | < 0.001 | 22,250 |
| keyword_in_url | +0.0037 | 0.579 | 22,250 |
| log(organic_keywords) (external authority) | -0.0623 | < 0.001 | 21,891 |
| log(organic_etv) | -0.1043 | < 0.001 | 21,891 |
| in-sample n_appearances proxy (for comparison) | -0.4217 | < 0.001 | 22,250 |

Note: the in-sample authority proxy (ρ = -0.42) is inflated by circularity (domains that rank more
are automatically scored "stronger"). The external signal shows only ρ ≈ -0.1, the realistic
magnitude.

### Coefficient shrinkage M1 → M4

| Model | Formula | β path_length | p | R² |
|---|---|---|---|---|
| M1 naive | position ~ path_length | +0.0056 | < 0.001 | 0.003 |
| M2 + authority | + log(organic_keywords) | +0.0071 | < 0.001 | 0.007 |
| M3 + page type | + C(page_type) | +0.0048 | < 0.001 | 0.008 |
| M4 full | + all confounders | +0.0036 | 0.036 | 0.012 |

Shrinkage M1 → M4: 35.9 %. The effect remains (barely) significant in the full model. With
log-authority on complete cases and cluster-robust SE at keyword level, β ≈ +0.0050 (p < 0.001).

### YMYL segmentation

| Segment | n | β path_length | p | Significant? |
|---|---|---|---|---|
| Non-YMYL | 14,508 | +0.0081 | < 0.01 | Yes (small positive effect) |
| YMYL (finance + health + gambling) | 7,742 | -0.0018 | 0.30 | No |
| Gambling (subsample) | 3,015 | +0.0035 | 0.40 | No |

The often-cited "YMYL exception" is not detectable in this dataset once external authority is
controlled for.

### Robustness

- 10 authority specifications (log / raw / sqrt / decile × organic_kw / organic_etv / organic_top3):
  all show β > 0, sign stable; 4 significant.
- Bootstrap 95 % CI: [+0.0015, +0.0085]. 5-fold CV: β stable across folds.
- Outlier trim (5 % each tail): β = +0.0072 — effect slightly stronger, not outlier-driven.

See [`/analysis/robustness_v2.json`](./analysis) and [`/charts/11_spec_curve.png`](./charts).

---

## Limitations

- **Observational, not causal.** Correlation, not proof of a ranking signal.
- **Low R² (~1.2 %).** Most position variance comes from unmeasured factors (content quality,
  backlinks per URL, user signals, page experience).
- **Two-day snapshot** (May 26–27, 2026); SERP volatility not captured.
- **Multicollinearity:** path_length VIF = 16.35 (with word_count_path VIF = 14.79).
- **keyword_difficulty** returned constant 0 from the API → effectively no contribution.
- **Authority metric** = DataForSEO organic_keywords_count; a good but imperfect proxy.

See [`/docs/methodology.md`](./docs/methodology.md) for the full methodology and limitations.

---

## License

- **Code** (`/scripts`): [MIT](./LICENSE)
- **Aggregated results, analysis outputs and charts** (`/analysis`, `/charts`):
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — reuse with attribution and a link
  to seo-kreativ.de.

Citation:

```
Ott, C. (2026). URL Length & Google Rankings — A 27,000-SERP Study.
https://www.seo-kreativ.de/blog/url-laenge-ranking/
```

---

## Contact

Questions, critique, methodology feedback welcome — open a GitHub Issue or reach out via
<https://www.seo-kreativ.de>.
