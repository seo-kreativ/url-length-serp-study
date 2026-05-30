# Methodology

Detailed methodology and limitations for the URL Length & Rankings Study.

## Research question

Do shorter URLs rank better in Google.de SERPs after controlling for domain authority and other
confounders?

## Data collection

### Keyword sample (phase 1)

2.862 keywords stratified across 9 verticals:
- E-commerce
- Finance
- Health
- Technology
- Travel
- Education
- Home & Garden
- Law
- Gambling

Plus 3.735 SERP rows from seo-kreativ.de as a self-benchmark (vertical = `seo_kreativ`).

Stratification was approximate (each vertical aiming for 300 keywords). Variation per vertical
reflects DataForSEO availability.

### SERP collection (phase 2)

DataForSEO Live SERP API. Endpoint:
```
https://api.dataforseo.com/v3/serp/google/organic/live/advanced
```

Settings:
- Location: Germany
- Language: German (de)
- Device: Desktop
- Depth: 10 (organic top 10)

Collection: May 26-27, 2026 (UTC).

### Feature engineering (phase 3)

For each SERP row, computed:
- URL anatomy: path length, path depth, word count, has dates/numbers/parameters/stopwords
- Domain characteristics: TLD type, is_subdomain, hyphen usage
- Page type classification: blogpost / category / landingpage / product (heuristic from path patterns)
- Keyword-in-URL flag (exact + partial match)
- SERP feature flags: has_featured_snippet, serp_feature_count

### External authority (phase 6)

For each of 3.921 unique domains, fetched from DataForSEO `domain_rank_overview` endpoint:
- `organic_keywords_count` - total organic keywords in DACH market
- `organic_etv` - estimated traffic value
- `organic_pos_1`, `organic_pos_2_3`, `organic_pos_4_10` - top position counts

Authority signals are external (independent of our keyword sample), avoiding the circularity
problem of in-sample frequency proxies.

## Analysis

### Statistical models

8 OLS regression models with HC3 robust standard errors. HC3 because heteroscedasticity is expected
when modeling discrete ranking positions (1-10) as a continuous outcome.

| Model | Formula |
|---|---|
| M1 naive | `position ~ path_length` |
| M2 + authority | `+ log(organic_keywords_count)` |
| M3 + page type | `+ C(page_type)` |
| M4 full | `+ path_depth + keyword_in_url + has_parameters + word_count_path + keyword_difficulty + C(search_intent) + has_featured_snippet` |
| M5 depth | replaces path_length with path_depth |
| M6 intent | `path_length * C(search_intent)` interaction |
| M7 ymyl | `path_length * is_ymyl` interaction |
| M8 class | `C(url_length_class)` instead of continuous |

### Robustness checks (phase 8)

1. **Authority specifications** (10 variants):
   - log(organic_keywords_count), log(organic_etv), log(organic_top3)
   - decile binning of log(organic_kw)
   - raw: organic_kw, organic_etv, organic_pos_1
   - sqrt(organic_kw)
   - no authority (control)
   - OLD n_appearances-proxy (comparison)

2. **VIF check** for multicollinearity.

3. **5-fold cross-validation** with K-Fold (random_state=42).

4. **Bootstrap** with 500 resamples, 95% percentile CI.

5. **Outlier sensitivity:** trim 5% from both ends of path_length distribution.

6. **YMYL robustness:** all 3 segments × 3 authority specifications.

### Significance

p < 0.05 = significant (single asterisk).
p < 0.01 = highly significant (two asterisks).
p < 0.001 = very highly significant (three asterisks).

Important caveat: at n = 22.250, even tiny effects achieve statistical significance. Practical
significance requires β to be meaningful in magnitude, not just statistically distinguishable
from zero.

## Limitations

### 1. Observational, not causal

The study shows correlations between URL properties and SERP positions. It does NOT prove that
Google uses URL length as a ranking signal directly. The observed effect could be:
- A direct ranking signal
- An indirect signal (via user behavior, CTR, click satisfaction)
- An artifact of an unobserved confounder (e.g. content quality correlated with both)

A randomized controlled trial (RCT) with URL length randomized at the URL level would be the
clean causal design. This is impossible to run in practice.

### 2. Low R² (1-2 %)

The full model explains only 1.2 % of position variance. This is methodologically honest but means
98 % of variance comes from factors not measured:
- Content quality (text depth, expertise, original research)
- Backlinks per URL (quantity and quality)
- On-page optimization (title tags, headings, internal linking)
- User signals (CTR, dwell time, pogo-sticking)
- Page experience (Core Web Vitals)

A higher R² would be suspicious - it would imply that simple URL features capture most of the
ranking system, which is not plausible.

### 3. Two-day snapshot

SERPs collected May 26-27, 2026. SERP volatility (daily fluctuations of 1-2 positions for many
queries) is not captured. Repeating collection in 3-4 waves over several weeks would improve
robustness. Budget-constrained for this iteration.

### 4. Multicollinearity

path_length VIF = 16.35, word_count_path VIF = 14.79. These are highly correlated by construction
(longer paths = more words). In the full model with both, the individual coefficient attribution
is fuzzy. Robustness across specifications with/without word_count_path shows the main result is
stable.

### 5. Authority metric is imperfect

DataForSEO organic_keywords_count is a good proxy but not a complete measure. Real domain
authority includes:
- Backlink quantity AND quality (referring domains, anchor diversity, link freshness)
- Brand mentions and search demand
- Click-through and engagement signals
- E-E-A-T signals (author expertise, source citations)

A future iteration could integrate Moz DA, Ahrefs DR, and Majestic TF for triangulation.

### 6. Self-benchmark inclusion

3.735 rows are from seo-kreativ.de keywords - selected because the domain ranks for them. This
creates a sample-bias for one specific domain. Excluding these doesn't materially change the
results (run `phase7_reanalysis.py` with `df[df.is_seo_kreativ != 1]` to verify), but the
inclusion is a deliberate transparency choice for the benchmark section.

### 7. Survivor bias in keyword sample

Keywords were sourced via DataForSEO `keyword_suggestions`, which biases toward queries that
already have substantial search volume. Long-tail and zero-volume queries are underrepresented.
The findings generalize to commercially relevant queries, not the entire query space.

### 8. German market only

All data is from google.de with German location/language. Generalizability to other markets
(US, UK, etc.) is not established. URL conventions and Google algorithm signals may differ.

## What this study can and cannot tell us

**It CAN tell us:**
- Whether short URLs correlate with better rankings in our sample
- Whether the correlation survives confounder adjustment
- How robust the finding is to different model specifications
- Rough magnitude of the partial association

**It CANNOT tell us:**
- Whether changing URL length on an existing site will improve rankings
- Whether Google "uses" URL length as a ranking signal in its scoring function
- What the optimal URL length is for any specific page
- How URL length interacts with non-measured factors (content quality, brand)

## Reproducibility checklist

To independently reproduce the v2 findings (starting from the provided data snapshot):

- [ ] Clone the repo
- [ ] `pip install -r requirements.txt`
- [ ] `cd scripts && python phase7_reanalysis.py` → should produce identical `analysis/model_results.json`
- [ ] `python phase8_robustness.py` → should produce identical `analysis/robustness_v2.json`
  (modulo bootstrap noise; the seed is fixed at 42)
- [ ] `python phase9_charts_v2.py` → should produce identical charts

If you re-collect SERPs via `phase1` and `phase2`, expect slightly different results because:
- SERP volatility (different snapshot date)
- DataForSEO keyword_suggestions may have evolved
- URL universe has changed (new pages added, old pages removed)
