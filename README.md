# AI-Based Rainwater Harvesting Recommendation System — Bengaluru

Give it a map pin and a few household details, and it tells you which rainwater
harvesting structure suits that specific site in Bengaluru, how big it needs to be,
what it will cost, how much of the household's water it will actually cover month
by month, and whether it pays for itself.

Environmental Science project (BAIP508), Department of AI & ML, BMS Institute of
Technology & Management. The original project brief is `READMEgovind.pdf`; this file
documents what was built against it.

> **Planning estimates, not an engineering design.** Every number here is for
> deciding whether and roughly what to build. A site visit, a soil percolation test
> and a licensed plumber's drawing are still required before anyone digs.

---

## What it does

1. **Estimates harvestable rainwater** for the location, from the hobli's measured
   rainfall normal, the roof area and the surface's runoff coefficient, net of
   first-flush and filter losses.
2. **Recommends a structure** — storage tank, recharge pit, borewell recharge, or a
   combination — or says plainly that the roof is too small to be worth it.
3. **Sizes and costs it**, with an itemised build-up and an indicative price range.
4. **Compares supply with demand** month by month, reporting what share of
   non-potable demand is met and how many months are covered in full.
5. **Reports the water saving**, the payback period, and how much goes back into the
   aquifer.
6. **Checks the Karnataka byelaw** — whether harvesting is mandatory for that plot
   size, and whether the recommended design would comply.
7. **Tests the design against a dry year**, using the real monthly pattern of the
   eight driest years on record rather than a flat percentage.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
python -m ml.train --regenerate # build the training set and the classifier
uvicorn app.main:app --reload
```

Then open <http://127.0.0.1:8000/> for the map UI, or
<http://127.0.0.1:8000/docs> for the generated API reference.

The app runs without a trained model — it falls back to the rule engine and says so
in `/api/health`. Tested on Python 3.12.

### Tests

```bash
python -m pytest -q      # 261 tests
```

`tests/test_guidelines.py` is the one to read first: each test there pins a number
to an outside authority (IS 15797, a CGWB design manual, the Karnataka byelaw,
KSNDMC's published district normal) so that a change which drifts away from
accepted practice fails a test instead of quietly shipping.

---

## How a recommendation is produced

```
inputs ─> harvest potential ─> structure decision ─> sizing ─> cost
                                      │                          │
                                      └──> water balance ────────┴──> saving & payback
```

The core relation, as in the brief and IS 15797:

```
harvest (litres) = rainfall (mm) × area (m²) × runoff coefficient
```

`app/core/engine.py:recommend()` is the single entry point. It is a pure function of
a `SiteContext` and a set of `HouseholdInputs` — no HTTP, no file IO — so the whole
model is testable without a server.

### The rules and the classifier

A deterministic rule engine (`app/core/recommender.py`) scores storage against
recharge from roof area, infiltration rate, groundwater depth and category, demand
coverage, and whether a borewell already exists. A scikit-learn classifier
(`app/core/ml.py`, trained by `ml/train.py`) learns the same decision from 16
features.

`engine` selects how they interact:

| mode | behaviour |
|---|---|
| `auto` *(default)* | Rules decide. The classifier runs alongside and any disagreement is reported in the `ml` block rather than hidden. |
| `rules` | Classifier not consulted. |
| `ml` | Classifier's label is used — but only if the site can physically take the structures it implies; otherwise the rules' choice stands. |

Sizing, costing and the water balance are **always** deterministic. The model only
ever picks the structure *type*, never a dimension or a price.

### How the model is tuned

`ml/train.py` searches four families — decision tree, random forest, extra trees
and histogram gradient boosting — over their hyperparameters by randomised
cross-validation, then calibrates the winner. Three choices are worth knowing:

- **Selection is on macro F1, not accuracy.** The labels are badly imbalanced:
  `storage_tank_and_recharge_pit` is over half the samples, while
  `borewell_recharge` is about 2.5%. Accuracy is dominated by the majority class
  and barely moves when a rare label is predicted badly. Macro F1 weights every
  structure type equally.
- **Probabilities are calibrated.** The API reports the model's probability to the
  user as a confidence, so that number has to mean what it says — a random forest's
  vote share does not. The winner is refitted on part of the training split,
  isotonic calibration is fitted on a held-out slice, and the result is kept only
  if it lowers expected calibration error on the untouched test set. If it does
  not, the uncalibrated model ships and the metrics say so.
- **The test split is scored once.** All search and selection happens inside
  cross-validation on the training split, so the reported figures are not the ones
  that were optimised.

`ml/artifacts/metrics.json` records the chosen hyperparameters, every family's
scores, the confusion matrix, the calibration before and after, and a learning
curve showing whether more samples would still help.

The full search takes about 15 minutes (4 families × 24 configurations × 5 folds).
`--fast` skips it and uses known-good fixed hyperparameters; `--n-iter` controls the
search width.

#### Current results

Random forest won on cross-validated macro F1, with histogram gradient boosting a
close second:

| Family | CV macro F1 | Holdout accuracy |
|---|---|---|
| **RandomForestClassifier** | **0.9514** | **0.9608** |
| HistGradientBoostingClassifier | 0.9501 | 0.9592 |
| DecisionTreeClassifier | 0.9440 | 0.9558 |
| ExtraTreesClassifier | 0.9023 | 0.9217 |

The shipped model scores **0.9596 accuracy, 0.9461 balanced accuracy, 0.9489 macro
F1** on the untouched test split. Tuning lifted accuracy from 0.9337, and the rare
`borewell_recharge` class from 0.86 to 0.89 F1 — which is the point of selecting on
macro F1 rather than accuracy.

Isotonic calibration cut expected calibration error from **0.0235 to 0.0028** and log
loss from 0.322 to 0.271, so the confidence the API reports is now worth reading as
a probability rather than a ranking.

The learning curve still gains +0.0058 at the last doubling, so more samples would
help a little. It is left at 12,000 deliberately: since the labels come from the
rule engine, more samples buy a closer imitation of the rules, not better advice.
Raise it with `--samples` if you want the imitation tighter.

**On the classifier's accuracy.** It scores ~96% against a held-out
split, which sounds better than it is: the training labels come from the rule
engine itself, with 3% label noise injected, so the model is largely learning to
imitate the rules. Tuning makes it a better *imitator* — it does not make the
underlying advice more correct. It is an honest demonstration of the ML layer the
brief asked for, not independent evidence that the recommendations are right.
Validation against real surveyed installations is the missing piece, and no such
dataset was available. `metrics.json` carries this caveat in the file itself.

---

## Data

| What | Source | Grade |
|---|---|---|
| Hobli rainfall normals, and their split across the three seasons | **KSNDMC** telemetric rain gauges, via [OpenCity](https://data.opencity.in/dataset/karnataka-annual-rainfall-districts-taluks-and-hoblis) (public domain) | measured |
| Month-by-month shape *within* each season | **IMD** published 1991–2020 Bengaluru monthly normals, via [OpenCity](https://data.opencity.in/dataset/bengaluru-rainfall) | measured (city-wide) |
| Dry-year monthly pattern | **IMD** Bengaluru monthly records 1901–2025; the eight driest years since 1991 | measured |
| Soil classification | **ISRIC SoilGrids 2.0** WRB class at each hobli centroid | measured, low confidence |
| Soil infiltration rate | Texture-based typical rates, cross-checked against SoilGrids | reference estimate |
| Depth to groundwater, CGWB category | CGWB / India-WRIS 2022 assessment | reference estimate |
| Rainy-day counts | Scaled from the IMD City ratio (56 days / 961.6 mm) | derived |
| BBMP zones and population density | OpenCity ward data, Census 2011 | reference estimate |
| Unit costs | Bengaluru market rates, 2025 base year | indicative |
| Basemap and boundaries | OpenStreetMap | — |

Provenance travels with the data rather than living only in this table. Every
`/api/location` and `/api/recommend` response carries `rainfall_data_grade` and
`data_grade` separately, because they genuinely differ — the rainfall is measured
while the soil and groundwater figures are not — plus plain-language `notes` saying
so. The UI prints them on the assessment sheet.

### Refreshing the data

Run in this order — each step feeds the next. Every script takes `--check` to
report and validate without writing.

```bash
python scripts/ingest_city_monthly.py   # IMD monthly normals + dry-year pattern
python scripts/ingest_ksndmc.py         # KSNDMC hobli normals -> rainfall tables
python scripts/ingest_soilgrids.py      # ISRIC soil classification + cross-check
python -m ml.train --regenerate         # retrain, since the features moved
```

All raw sources are vendored in `data/raw/`, so ingestion is reproducible offline
and every shipped number can be audited against the file it came from.
`ingest_soilgrids.py --fetch` re-queries the ISRIC API.

**`--regenerate` matters.** `ml.train` reuses a cached `ml/data/training_data.csv`
if one exists. After changing anything under `data/`, retrain with `--regenerate`
or the model will be fitted to the old site data.

### What is genuinely measured, and what is not

Each of the 22 hoblis carries its **own** KSNDMC annual normal and its **own**
measured split across Karnataka's three rainfall seasons (pre-monsoon January–May,
south-west June–September, north-east October–December). Those seasonal shares
really do vary across the city: reliance on the south-west monsoon ranges from 53.0%
in Hoskote to 58.8% in the Bengaluru East kasaba.

The shape *inside* a season is one shared curve, taken from IMD's published
1991–2020 Bengaluru monthly normals — the current WMO reference period. So a
hobli's season totals are its own, but the split between, say, August and
September is city-wide. KSNDMC publishes per-hobli monthly normals only in annual
report PDFs that are not machine-readable; parsing those is the next fidelity step.

### The dry-year outlook

`dry_year_outlook` in every report answers a question the averages cannot: what
does the system you are about to build do when the monsoon is weak? The tank is
**not** resized — the normal-year design is held fixed and only the rainfall
changes.

The scenario is the mean monthly pattern of the eight driest years since 1991,
drawn from 124 years of IMD records. It is a monthly pattern rather than a flat
percentage because a dry year in Bengaluru is not an even shortfall: those years
average 66% of normal overall, but **September collapses to 39%** while June holds
at 102%. September is the wettest month of a normal year and the one a tank relies
on to fill before the dry season, so a uniform 66% scaling would hide exactly the
risk this exists to show.

For context, across the full record annual totals run P10 = 678 mm, P50 = 950 mm,
P90 = 1227 mm. The driest year since 1991 was 1994, at 587 mm.

### Why soil classification does not set the infiltration rate

`data/soil_wrb.csv` holds a real ISRIC SoilGrids classification for every hobli,
and it is reported alongside the assumed infiltration rate — but it deliberately
does not set it. Driving the rate from the classification was tried and made the
model worse.

Across all 22 hoblis SoilGrids returns only two modal classes (Luvisols for 15,
Lixisols for 7) at 32–55% confidence. Both are argic, clay-enriched groups of
similar permeability, so a probability-weighted rate collapses into a 12.3–18.6
mm/hr band — a spread of 6 mm/hr where the texture estimates span 23. Since the
rule engine screens recharge pits at 5 mm/hr, every hobli would clear the
threshold and infiltration would stop discriminating anything at all; Varthur and
Sarjapura, which are clay and currently get appropriately cautious sizing, would
be treated as freely draining.

So it is used as a **cross-check** instead. Where SoilGrids puts meaningful
probability on Vertisols — shrink-swell clay, which is what actually kills a
percolation pit — but the assumed rate says the ground drains freely, the hobli is
flagged and the report raises a caveat. Six hoblis currently trip it: Bidarahalli,
Begur, Attibele, Jigani, Devanahalli and Hoskote.

Two hoblis — the Bengaluru East and Bengaluru South kasabas — have no gauge of their
own and use their taluk's normal instead. They are graded `ksndmc_taluk_normal` so
the coarser source is visible wherever the number appears.

Hobli normals are the mean of the gauges inside them, since KSNDMC names gauges
rather than hoblis (`Kengeri_1`…`Kengeri_4`) and its spellings drift from the
project's (`Beguru`/`Begur`, `Varturu`/`Varthur`, `K R Pura`/`Krishnarajapuram`).
`GAUGE_ALIASES` in the ingestion script maps these explicitly rather than
fuzzy-matching.

---

## API

| Route | Purpose |
|---|---|
| `GET /api/health` | Liveness; how many hoblis and zones loaded, whether a model is trained |
| `GET /api/parameters` | Form defaults, allowed values and key constants, for the UI |
| `GET /api/hoblis` | The covered hoblis with centroids, for the map |
| `GET /api/zones` | BBMP zones with population density |
| `GET /api/location` | Site lookup for a pin: hobli, soil, groundwater, zone |
| `GET /api/rainfall` | Monthly rainfall for a pin or a hobli |
| `GET /api/model` | Classifier metrics, tuned hyperparameters and feature importances |
| `POST /api/recommend` | The full recommendation report, including `dry_year_outlook` |

```bash
curl -X POST http://127.0.0.1:8000/api/recommend \
  -H 'content-type: application/json' \
  -d '{"lat":12.9352,"lon":77.6245,"roof_area_m2":120,"roof_type":"rcc_flat",
       "open_area_m2":40,"household_size":4,"water_source":"cauvery_and_borewell",
       "has_borewell":true,"plot_area_m2":223}'
```

Accepts either `lat`/`lon` or a `hobli` name.

---

## Layout

```
app/
  main.py          FastAPI app, static hosting, error handlers
  api.py           routes
  schemas.py       request validation
  config.py        paths; every constant comes from data/parameters.json
  datasets.py      CSV loading, nearest-centroid location lookup, provenance
  core/
    calculator.py  harvest potential, effective rainfall, monthly series
    demand.py      household demand, potable vs non-potable split
    recommender.py the rule engine: which structure, and why
    sizing.py      tank volume, pit dimensions, borewell recharge sizing
    balance.py     month-by-month supply vs demand simulation
    costing.py     itemised cost build-up, savings and payback
    ml.py          classifier loading and prediction
    engine.py      orchestrates all of the above into one report
ml/
  generate_dataset.py  synthesises labelled training samples from the rules
  train.py             model selection, metrics, exported tree rules
  artifacts/           trained model + metrics.json (committed)
data/
  hoblis.csv           22 hoblis: centroid, rainfall, soil, groundwater
  rainfall_monthly.csv 12 rows per hobli
  soil_wrb.csv         ISRIC soil classification + infiltration cross-check
  bbmp_zones.csv       8 BBMP zones
  costs.csv            unit rates
  parameters.json      design constants, monthly normals, dry-year pattern
  raw/                 vendored KSNDMC, IMD and ISRIC source files
scripts/
  ingest_city_monthly.py  IMD monthly normals and the dry-year pattern
  ingest_ksndmc.py        KSNDMC hobli rainfall normals
  ingest_soilgrids.py     ISRIC soil classification
  build_reference_data.py derive monthly from annual (fallback only)
web/                   Leaflet map UI and printable assessment sheet
tests/                  261 tests
```

No constant is hard-coded in `app/core`. Runoff coefficients, tariff slabs, byelaw
thresholds, design storm intensity and storage design days all live in
`data/parameters.json` so they can be reviewed and changed without touching code.
`RWH_DATA_DIR` overrides the data directory.

---

## Roadmap status

All seven items from the brief are implemented.

- [x] Load and clean KSNDMC rainfall data by hobli
- [x] Rainfall and harvest-potential calculator
- [x] Map input and zone lookup
- [x] Recommendation model (structure type and size)
- [x] Cost estimate and demand-supply comparison
- [x] Web UI and result report
- [x] Validation against standard RWH design guidelines

---

## Limitations

These are the ones that would change a decision, not a boilerplate list.

- **Bengaluru only**, and only 22 hoblis across Bengaluru Urban plus four adjoining
  Bengaluru Rural taluks.
- **Location lookup is nearest-centroid, not point-in-polygon.** The bundled tables
  carry centroids rather than boundaries, so a pin near a hobli border can match the
  neighbour. Every response reports which hobli matched and how far the pin was from
  its centroid, and warns past 6 km.
- **The water balance runs monthly, not daily.** Treating a month's rain as
  available all month flatters small tanks; a daily simulation would show more of a
  wet month overflowing. Reported reliability is therefore optimistic.
- **Soil infiltration and groundwater are hobli-level reference estimates.**
  Infiltration varies enormously within a hobli and drives whether a recharge pit
  is recommended at all. The ISRIC cross-check flags where it is most doubtful, but
  a flag is not a measurement — only a percolation test is.
- **Cloudbursts are not modelled.** The dry-year outlook now covers the weak-monsoon
  case from the real record, but a month's rain falling in a day is not represented
  anywhere, and that is what overwhelms a first-flush diverter and a pit together.
- **The classifier imitates the rule engine**, as described above. Its accuracy is
  not evidence the advice is right.
- **Costs are indicative** 2025 Bengaluru rates and move with material prices and
  site access.
- **Taluk labels are the project's own.** KSNDMC now places Yelahanka, Jala and
  Hesaraghatta in a separate Yelahanka taluk; `data/hoblis.csv` still lists them
  under Bengaluru North. Gauge mapping is unaffected — only the printed label.

---

## Sustainability goals

Supports SDG 6 (Clean Water and Sanitation), SDG 11 (Sustainable Cities and
Communities) and SDG 13 (Climate Action).

## Related work

Builds on GIS and multi-criteria (AHP) studies of RWH site suitability, ML-based RWH
potential mapping, rooftop potential studies, and Bengaluru groundwater research
(Sekhar et al., *Sustainability*, 2018).

## Licence and attribution

Project code is coursework. KSNDMC rainfall data via OpenCity is public domain;
OpenStreetMap basemap tiles and boundaries are © OpenStreetMap contributors (ODbL).
Check each source's licence and current availability before reuse.
