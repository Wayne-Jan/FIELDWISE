# Inputs of FieldWise

This page defines what FieldWise reads when it is fitted and applied to your own fields: the tables, their units, which
parts need yield maps of earlier seasons, and the timing rules that keep every fit to information available at the
decision. `fieldwise/validate.py` checks prepared inputs against these rules.

Times are UTC. Instants are milliseconds since 1970-01-01 (`*_ms`); days are UTC day numbers since 1970-01-01. Yield is in
t/ha.

## 1. Four roles of the data

| Role | Content | Used by |
|---|---|---|
| History | Earlier field-seasons with yield maps: Base predictions, labels and all inputs | Every fit: residual bank, references, RPT, G, K, readouts, strengths, alpha |
| Target years and cutoffs | Years to serve and the cutoff of each year and phase (section 3) | Every fit: which history is legal |
| Queries | Cases to correct: Base predictions and inputs, no labels | Prediction |
| Evaluation labels | Yield maps of the queries | Evaluation only; never passed to a fit |

Prediction needs no labels. Fitting needs labelled history: the residual of a pixel is label minus Base prediction, and
every component learns from residuals of earlier years. In a rolling evaluation the bank may hold the labels of every
year: the fits serving year T select their history by time (section 3), so labels of year T and later never enter them,
and labels that became available only after T's cutoff stay out as well. In deployment the queries have no labels.

A **case** is one field-season at one decision stage: `tau1`–`tau4` in season and `harvest`.

## 2. Decision schedule

`fieldwise/schedule.py` builds the times from sowing and harvest dates and daily air temperature (manuscript Section
3.3.1 and Appendix A.3; rebuilt from the published rules and identical to the study's case table for all 9,843 cases).

- **Thermal clock.** Growing degree days accumulate from the sowing day: GDD = sum of max(T_mean − T_base, 0), where
  T_mean is the mean of the 24 hourly 2 m temperatures of a UTC day. A day with a missing hour is invalid. Threshold k is
  reached on the first day whose cumulative GDD is at least the threshold, provided every day from sowing to that day is
  valid; otherwise it is unknown.
- **Thresholds.** Maize and soybean: base 10 °C, 400, 800, 1,200, 1,600 °C·day. Wheat: base 0 °C, 600, 1,200, 1,800,
  2,400. Rapeseed: base 0 °C, 700, 1,400, 2,100, 2,800. Other crops need their own (`schedule.CROPS`).
- **Window.** From sowing to 00:00 of the day after the threshold day (in season) or after the harvest day (harvest).
- **Decision.** Window end + 6 days (`DECISION_DELAY_DAYS`, the assumed availability delay of ERA5-Land, the latest of
  the inputs). Every input of a case must be available by its decision.
- **Labels.** Available at 00:00 of the harvest date + 30 days (`LABEL_DELAY_DAYS`).
- **Exclusions.** In-season cases whose threshold is unknown or whose decision falls on or after the harvest date; all
  cases of a field-season whose sowing date has no usable clock (for example, derived from the harvest date) or that has
  no eligible in-season case.

Not part of the schedule and left to you: removing duplicate records, field-seasons with conflicting sowing dates and
conflicting versions of a yield map (the study removed 11 such field-seasons).

## 3. Cutoffs and legal history

Every target year T has a **cutoff** per phase: the time by which the fits serving T are made. Following the manuscript,
it is the earliest decision of the phase in year T; the four in-season stages share one cutoff and harvest time has its
own. Build the table from all queries of the year, labelled or not, with `selection.cutoff_table(queries)`; it has columns
`year`, `phase` (`in_season` or `harvest`) and `cutoff_ms`. Pass `selection.phase_cutoffs(table, phase)` as the `cutoff`
argument of every fitting function. Without it the functions use the earliest decision among the cases they are given,
which is only correct when those cases include every query of the phase, and undefined for a year without cases.

Fit and serve one crop at a time. The cutoffs, the legal history and the references of a crop come from that crop only,
and the functions take no crop argument. If your tables carry a `crop` column, tables holding several crops are rejected
(`selection.cutoff_table`, the fitting functions, `retrieval.select_references`, `rpt.RPT` and `validate.cases`).

| What | Rule |
|---|---|
| Legal history of T | Cases of earlier years whose decision and labels are both available by the cutoff of T |
| References of a query | Same stage, earlier year, decision and labels available by the query's own decision |
| Input statistics | Computed only from inputs available by the cutoff: the daily-weather standardisation uses the days available by the cutoff, the RPT initial scale the embeddings of cases decided by the cutoff, and the retrieval standardisation the candidates decided by the earliest query decision of the stage and year |
| Within the history | Field-seasons weigh equally within a year and years equally |

The manuscript code computed the retrieval standardisation from all earlier-year candidates of the stage; this leaves
every reference list and weight unchanged.

## 4. Case table

One row per case.

| Column | Type | Meaning |
|---|---|---|
| `case_id` | string | Unique case identifier |
| `year` | int | Harvest year of the field-season |
| `field` | any | Field identifier, unique within a year (one field-season per field and year) |
| `stage` | string | `tau1`–`tau4` or `harvest` |
| `decision_ms` | float | Decision time |
| `label_ms` | float | Time the yield map becomes available; may be missing for queries without labels |
| `window_end_ms` | float | Optional: end of the observation window; not later than `decision_ms` |
| `input_ms` | float | Optional: time by which every input of the case is available; not later than `decision_ms` |
| `crop` | string | Optional: one crop per table (section 3) |

Every decision, in season and at harvest, must precede the availability of the case's own labels.

## 5. Residual bank

`fieldwise.bank.Bank`: a case table and pixel arrays in case order.

| Array | Meaning |
|---|---|
| `row` | Case index of each pixel (pixels grouped by case) |
| `pixel` | Service-pixel identifier; `pixel - pix0` of the case is its position within the field |
| `pred` | Base prediction (t/ha) on every service pixel; `bank.ensemble` averages several Base seeds |
| `target` | Label (t/ha); used only where `gt_valid` |
| `gt_valid` | Pixel with a valid yield observation |

Service pixels are the 10 m pixels of the yield-map grid inside the field. The Base predictions of year V must be
out-of-sample: the Base model serving V is trained only on data legal at V's cutoff. The Base can be any model; the code
for the two Base models of the manuscript is in `fieldwise/base/`.

## 6. Daily weather

An array `daily [field-seasons, days, 10]` in the order of `weather.DAILY`, one row per UTC day starting 30 days before
sowing, with the UTC day number of its first row (`fs_first_day`). Each case points to its field-season (`fs_row`) and
its number of usable days (`n_days`): the last usable day is the last day before the window end. A day is available 6 days
after it ends, so the last usable day must end at least 6 days before the decision (with the schedule of section 2, the
decision is exactly 6 days after the window end).

| Variable | Unit | From hourly ERA5-Land (`weather.daily_quantities`) |
|---|---|---|
| `tmean`, `tmax`, `tmin` | °C | Mean, maximum, minimum of the 24 hourly 2 m temperatures |
| `vpd_max` | kPa | Maximum hourly vapour-pressure deficit (Tetens) |
| `precip` | mm | Sum of hourly precipitation |
| `ssrd` | MJ m⁻² | Sum of hourly downward solar radiation |
| `total_evap`, `pot_evap` | mm | Sums of hourly total and potential evaporation (ERA5 sign: evaporation is negative) |
| `theta_rz` | m³ m⁻³ | Mean root-zone soil water, 0.07 θ1 + 0.21 θ2 + 0.72 θ3 |
| `theta_l4` | m³ m⁻³ | Mean soil water of layer 4 |

A day with a missing hour is missing. G projects the 240 days up to the last usable day on eight B-spline lag kernels.

## 7. Named environment summaries

Twelve per case, from sowing to the last usable day.

| Column | Definition | Unit |
|---|---|---|
| `E1_total`, `E1_14d` | Heat exposure: sum over hours of max(T − threshold, 0) / 24, season and last 14 days; threshold 30 °C (32 °C soybean) | °C·day |
| `E2_total`, `E2_14d` | Hours with vapour-pressure deficit above 2 kPa, season and last 14 days | h |
| `E4_cum_precip` | Precipitation since sowing | mm |
| `E3_theta_last7` | Mean root-zone soil water of the finite days among the last seven | m³ m⁻³ |
| `E3_theta_min` | Minimum daily root-zone soil water since sowing | m³ m⁻³ |
| `lst_day_14` | Mean daytime land-surface temperature, last 14 days | °C |
| `dT_day_14`, `dT_day_seas` | Mean daytime land-surface temperature minus daily maximum air temperature, last 14 days and season | °C |
| `dT_pos_30` | Fraction of days with daytime land-surface temperature above maximum air temperature, last 30 days | 0–1 |
| `lst_night_14` | Mean nighttime land-surface temperature, last 14 days | °C |

The first seven come from `weather.daily_exposure_series` and `weather.exposures`. Land-surface temperature is MODIS
MOD11A1 (daily, 1 km, day and night quality control, available 2 days after observation); the package has no reader for
it, so provide the five columns yourself (missing values are allowed and standardised to zero). G also uses the stage
indicators, the Base mean, standard deviation and log pixel count of the case.

## 8. Retrieval descriptors

One row per case, numeric columns grouped by prefix (`retrieval.prefix_groups`). The manuscript uses four groups:

| Group | Prefixes | Content |
|---|---|---|
| Sentinel-2 | `q_s2_fixedgdd4__`, `q_s2_asof__` | Field-mean NDVI, NDRE and NDMI over the available season and each reached GDD interval: mean, standard deviation, minimum, maximum, observation count, valid ratio; per interval also endpoint slope and trapezoidal integral; peak acquisition index, mean within-field variability, valid pixel–time fraction |
| Weather | `q_weather_fixedgdd4__` | Daily mean, maximum, minimum temperature, diurnal range, precipitation and GDD over the season and reached intervals; precipitation and GDD totals, days above 30 °C maximum, days below 5 °C minimum, days with at most 0.1 mm, longest dry spell; per interval valid-day counts and ratios |
| Soil | `q_soil__` | Spatial mean and standard deviation of CEC, coarse fragments, clay, nitrogen, pH, sand, silt and organic carbon at 0–5, 5–15 and 15–30 cm, and pooled topsoil summaries (SoilGrids) |
| Terrain | `q_topo__` | Spatial mean and standard deviation of aspect, curvature, elevation, slope, topographic wetness index (SRTM) |

GDD intervals are the stretches between successive thresholds (`__stage0__` from sowing to threshold 1, `__stage1__` to
threshold 2, ...). At stage `tau<j>` only intervals 0 to j−1 have been reached; columns of later intervals must be missing.
Sentinel-2 summaries use scene classes 4, 5 and 7 and valid, non-negative bands, from acquisitions within the window.
`fieldwise/tutorial/descriptors.py` builds them from the Base inputs of a collected season; any descriptors that follow the
window rule can be used instead, grouped by prefix.

## 9. Pixel embeddings for RPT

A float16 array `[rows, 128]` with an index table (`case_id`, `row_start`, `n_pixels`): one row per service pixel of each
case, in bank pixel order, contiguous. The manuscript uses frozen TESSERA embeddings of the case's observation window
(`fieldwise/fm/`); a pixel without valid inputs has a missing (non-finite) embedding and receives no within-field
correction. RPT centres embeddings within each case. Reference embeddings and query embeddings must come from the same
encoder. Smoothing over 15 m uses neighbour lists per case (`geometry` of `rpt.RPT`): `rpt.neighbours(lon, lat)` builds
them from pixel-centre longitude and latitude in degrees, converted to local metres; with projected coordinates, build
the lists directly in metres.

## 10. Field-mean embeddings for readouts

An array `[cases, windows, models, D]` of field-mean embeddings (`readout.field_means`), windows in this order: the
calendar year before sowing, the first decision window (`tau1`), and the case's own window (or the full season at
harvest). A missing window is NaN. Each foundation model has its own readout; FieldWise averages them.

## 11. Checks

| Function | Checks |
|---|---|
| `validate.cases` | Columns, unique cases, known stages, one crop, decisions before label availability, windows and inputs by the decision |
| `validate.cutoffs` | A cutoff for every target year and phase, not later than the earliest query decision |
| `validate.late_history` | Earlier-year cases decided or labelled after a cutoff (left out by the fits; for information) |
| `validate.bank` | Pixel rows, finite predictions, finite labels where marked valid |
| `validate.daily` | Usable daily weather ends with the window and is published (6 days) by the decision |
| `validate.interval_features` | GDD-interval descriptors masked after the decision |
| `validate.embeddings` | Contiguous embedding rows, one per service pixel |

`validate.assert_valid(...)` raises on errors and returns the warnings.

## 12. Tutorial files

What you provide to `fieldwise.collect` and `fieldwise.tutorial`:

| Input | Format |
|---|---|
| Field boundaries on a 10 m grid | `FIELDS.json` (`x0`, `y1`, `w`, `h`, `crs`) and `PIXELS.parquet` (`field`, `row`, `col`, `lon`, `lat`) |
| Season dates | `FIELDS.parquet` (`field`, `n_pix`, `sow_day`, `harvest_day` as UTC day numbers) |
| Crop | maize, soybean or wheat settings (`docs/DATA_COLLECTION.md`, decision times) |
| Yield maps of earlier seasons (optional) | npz with `target` (t/ha) and `gt_valid` per pixel, in the pixel order of the seasons given (as `out/run/yield_history.npz`); pass with `--labels <file>` to `fieldwise.tutorial.run` |

What each step writes:

| Step | Output |
|---|---|
| `fieldwise.collect` | Decision times and cases (`<season>/collect/base_inputs/CASES.parquet`), inputs from public products |
| `fieldwise.collect.encode` | `<season>/encode/`: TESSERA per-pixel embeddings, field means of the four foundation models |
| `fieldwise.tutorial.run` | `out/run/model/`: Base model, RPT projection, G, K, readouts, strengths and alpha of the new season; `out/run/rpt_database/`: per-pixel residuals, embeddings, field descriptors and coordinates of the earlier seasons; `out/run/pixels.parquet`, `cases.parquet`, `RUN.json`: out-of-sample predictions of every season, per-year choices and the evaluation; synthetic yields when `--labels` is not given |
| `fieldwise.tutorial.infer` | `out/new/pixels.parquet`: Base, within-field correction S15, Correction only, FieldWise, lon, lat per pixel; `out/new/cases.parquet`, `references.parquet`: G, K, readouts, offsets, reference fields and weights per field |

The Base architecture and the four foundation models are fixed in this release. Fit one crop at a time.
