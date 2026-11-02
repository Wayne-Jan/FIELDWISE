<p align="center"><img src="docs/images/logo.png" alt="FieldWise" width="420"></p>

**FieldWise corrects pixel-level crop yield predictions for a new season before any of its yield is known, using
residual patterns and field-mean errors learned from earlier seasons.**

Reference code for *FieldWise: Explanation fidelity and decision value in residual correction of subfield crop yield
predictions* (Chan and Yang, under review) · [Quick start](#quick-start) · [Use your own data](#use-your-own-data) ·
[Citation](#citation-and-data)

The tutorial runs on **real public inputs; only the yields are synthetic**. Yield maps are rarely public, so the tutorial
generates them from the real inputs (`fieldwise/tutorial/synth.py`). Results on synthetic yields show how the code works,
not how well FieldWise works.

## How it works

![Fig. 1 of the manuscript](docs/images/figure1.png)

Fig. 1 of the manuscript (Chan and Yang, under review): FieldWise workflow for a soybean field-season at the fourth in-season
decision time. (A) Base predictions and historical residuals. (B) Information available at the decision time. (C) Residual
pattern transfer (RPT) and 15 m smoothing. (D) Environmental and calibration offsets, foundation-model residual readouts, and
offset selection. (E) Corrected prediction and observed yield. Clock symbols mark operations using earlier seasons. Query
yield is shown only for evaluation. The example field-season is from YieldSAT (Miranda et al., 2026).

FieldWise adds two corrections to the Base prediction: a **within-field correction** β·S15, transferred from the residual
maps of similar earlier fields (RPT), and a **field-mean correction** u, chosen by a decision layer from an environmental
offset G, a calibration offset K and residual readouts of four Earth-observation foundation models (TESSERA, Galileo,
OlmoEarth, Presto). Every part is fitted on earlier seasons only.

| Panel | Step | Code | Output |
|---|---|---|---|
| A | Base predictions and historical residuals | `fieldwise/base/`, `fieldwise/tutorial/run.py` | `out/run/model/base/`, `out/run/rpt_database/bank_*` |
| B | Decision times and what is available at each | `fieldwise/schedule.py`, `fieldwise/collect/build_base_inputs.py` | `<season>/collect/base_inputs/CASES.parquet` |
| C | Reference fields, pixel match, within-field correction S15 | `fieldwise/retrieval.py`, `fieldwise/rpt.py`, `fieldwise/tutorial/descriptors.py` | `out/new/references.parquet`, `out/new/pixels.parquet` |
| D | G, K, readouts and the decision layer | `environment.py`, `calibration.py`, `readout.py`, `strength.py`, `decision.py` in `fieldwise/` | `out/new/cases.parquet` |
| E | Base + β·S15 + u = FieldWise | `fieldwise/tutorial/infer.py` | `out/new/pixels.parquet` |

`fieldwise/explain/` holds the explanation tools of the manuscript (error layers, spatial structure, decision value,
readout increments, concept erasure).

## Quick start

Needs Docker; runs on CPU. The demo inputs are seven seasons (2019/20–2025/26) of 10 image-screened areas near Renli,
Tainan, collected and encoded from public products; the last season is the new season.

```bash
git clone https://github.com/Wayne-Jan/FIELDWISE.git && cd FIELDWISE
docker build -t fieldwise .
docker run --rm fieldwise python -m unittest discover -s tests     # optional: unit and leakage tests

# demo inputs, 44 MB
curl -LO https://github.com/Wayne-Jan/FIELDWISE/releases/download/demo-data-v1/fieldwise_tainan_demo_v1.tar.gz
tar -xzf fieldwise_tainan_demo_v1.tar.gz
mkdir -p out

RUN="docker run --rm --init -v $PWD/fieldwise_tainan_demo_v1:/data:ro -v $PWD/out:/out fieldwise"
SEASONS="/data/2019 /data/2020 /data/2021 /data/2022 /data/2023 /data/2024 /data/2025"

# 1. fit: synthetic yields, Base, RPT database and corrections (the 2025/26 yields are held out)
$RUN python -m fieldwise.tutorial.run --seasons $SEASONS --target 2026 --out /out/run --seeds 42 --epochs 3
# 2. serve the new season from the saved model and database, without its yields and without refitting
$RUN python -m fieldwise.tutorial.infer --model /out/run/model --database /out/run/rpt_database --season /data/2025 --out /out/new
# 3. draw the maps
$RUN python docs/tutorial_figures.py --run /out/run --infer /out/new --season /data/2025 --stage tau2 --out /out/figures
```

The quick start uses one seed and 3 epochs to stay short; the manuscript setting is five seeds and 55 epochs
(`--seeds 42 43 44 45 46 --epochs 55`).

**What you get**

| File | Content |
|---|---|
| `out/run/model/` | Base model, RPT projection, G, K, readouts, strengths and alpha of the new season |
| `out/run/rpt_database/` | Residual database of the earlier seasons: per-pixel residuals, embeddings, field descriptors, coordinates |
| `out/run/pixels.parquet`, `out/run/cases.parquet`, `out/run/RUN.json` | Out-of-sample predictions of every season, per-year choices and the evaluation |
| `out/new/pixels.parquet` | New season per pixel: Base, within-field correction S15, Correction only, FieldWise, lon, lat |
| `out/new/cases.parquet`, `out/new/references.parquet` | New season per field: G, K, readouts, offsets; reference fields and weights |
| `out/figures/inference.png` | The maps below, for the 10 demo areas |

![The new season predicted without labels](docs/images/inference.png)

The new season predicted from the saved model and database, without labels, on a Sentinel-2 image of the season. This
figure was drawn from the full tutorial collection of the same area (about 120 areas per season); the quick start draws
it for its 10 areas. The synthetic yield is shown for evaluation only.

## Build from public data

To rebuild the inputs or move to another place, collect and encode each season, then fit and serve as in the quick start.
One folder per season, oldest first.

**Requirements.** An Earth Engine project (`export FIELDWISE_EE_PROJECT=<project>`, authenticated with
`earthengine authenticate`; see [Earth Engine access](https://developers.google.com/earth-engine/guides/access)); the
collection image (`docker build -f Dockerfile.collect -t fieldwise-collect .`); for encoding, the encoding image
(`docker build -f Dockerfile.encode -t fieldwise-encode .`), which downloads the four foundation models from their official
sources at the versions used for FieldWise (see [Foundation models](#foundation-models)), and preferably an NVIDIA GPU.
Sentinel-1 RTC is read from the public Microsoft Planetary Computer catalogue.

**1. Fields and dates** (see [Use your own data](#use-your-own-data)). Without farm records, the Tainan example derives
season-start and season-end proxies (SOS20 / EOS20 of Sentinel-2 NDVI) and autumn-winter growth-cycle areas from imagery;
these are retrospective proxies, not sowing or harvest dates, and the crop is not verified:

```bash
python -m fieldwise.collect.extract_s2_season raw/s2_season <x0> <y1> <w> <h> <Y>-02-01 <Y+1>-08-01
python -m fieldwise.collect.season_proxies raw/s2_season <parcels.gpkg> <Y> <cx> <cy> 10 3000 fields
```

**2. Collect** the public products (sowing year `Y`):

```bash
python -m fieldwise.collect.init_fields fields <season>/collect <Y>
for step in base soil terrain build-base lst build-environment fm rtc-pre rtc; do
  python -m fieldwise.collect $step --root <season>/collect --config <season>/collect/collection.json
done
```

**3. Encode** with the four foundation models in the encoding image: TESSERA per-pixel embeddings (the RPT input) and field
means of the four models (the readout input). Without a GPU, add `cpu` after the output folder.

```bash
docker run --rm --init --gpus all -v $PWD/<season>:/season fieldwise-encode \
  python -m fieldwise.collect.encode /season/collect /opt/fm/paths.json /season/encode
```

**4. Fit:** synthetic yields (or yours, `--labels`), the Base of every year trained on earlier seasons only, the RPT
database of out-of-sample residuals, and the corrections, as in quick-start step 1. The new season's yields are held out of
every fit; `tests/test_leakage.py` checks that changing or removing them leaves its fits, references and predictions
unchanged.

**5. Serve** the new season as in quick-start step 2.

[`docs/DATA_COLLECTION.md`](docs/DATA_COLLECTION.md) lists every product, band and availability delay;
[`docs/INPUTS.md`](docs/INPUTS.md) defines the prepared tables, units and the timing rules that keep every fit to
information available at the decision.

## Foundation models

FieldWise uses four frozen Earth-observation foundation models. TESSERA's per-pixel embeddings give the pixel correspondence
of RPT; the field means of all four feed the residual readouts. What each model reads in FieldWise, before its official
normalisation (`fieldwise/fm/views.py`):

| | [Presto](https://github.com/nasaharvest/presto) | [Galileo](https://github.com/nasaharvest/galileo) | [OlmoEarth](https://github.com/allenai/olmoearth_pretrain) | [TESSERA](https://github.com/ucam-eo/tessera) |
|---|---|---|---|---|
| Checkpoint | default | nano | v1.2 nano | v2 student nano |
| Time steps | 30 days | 30 days | 30 days | every clear acquisition, with day of year |
| **Remote sensing** | | | | |
| Sentinel-2 | L1C quality mosaic (Cloud Score+), NDVI | L1C quality mosaic (Cloud Score+), NDVI | L2A, 12 bands, first clear acquisition per pixel | L2A, 10 bands, clear acquisitions (SCL) |
| Sentinel-1 | GRD VV, VH, one orbit pass | GRD VV, VH, one orbit pass | GRD VV, VH | RTC gamma0 VV, VH, ascending and descending apart |
| Landsat | – | – | masked | – |
| **Crop and surface** | | | | |
| Dynamic World | label per step | class probabilities | – | – |
| **Land** | | | | |
| Terrain (SRTM) | elevation, slope | elevation, slope | – | – |
| WorldCereal 2021 | – | classification | – | – |
| **Environment** | | | | |
| ERA5-Land | temperature, precipitation | temperature, precipitation | – | – |
| TerraClimate | – | aet, def, soil | – | – |
| **Human activity** | | | | |
| VIIRS night lights | – | avg_rad | – | – |
| LandScan | – | population | – | – |
| **Location** | latitude, longitude | latitude, longitude | – | – |
| **Used in FieldWise** | field-mean readout | field-mean readout | field-mean readout | pixel correspondence (RPT), field-mean readout |
| **Paper** | Tseng et al. (2023) | Tseng et al. (2025) | Herzog et al. (2026) | Feng et al. (2026) |

The inputs follow each model's official recipe; the few substitutions (Cloud Score+ as the quality-mosaic key, hourly
ERA5-Land aggregated per step, the field's clear fraction for the OlmoEarth scene order, a partial last step) are listed at
the top of `fieldwise/fm/views.py`. Products, bands and availability delays: [`docs/DATA_COLLECTION.md`](docs/DATA_COLLECTION.md).

`Dockerfile.encode` downloads these versions and checks their SHA-256; `fieldwise.fm.encode.Encoder` refuses other weights
unless `check=False`. The image's model locations are in `/opt/fm/paths.json`; for your own installation, write a JSON file
with the fields of `ModelPaths` (`fieldwise/fm/encode.py`).

| Model | Code | Weights | SHA-256 of the weights | Licence |
|---|---|---|---|---|
| Presto | [nasaharvest/presto](https://github.com/nasaharvest/presto) `ba88a3f` | `data/default_model.pt` in the repository | `559c9d76f525…` | MIT |
| Galileo | [nasaharvest/galileo](https://github.com/nasaharvest/galileo) `0f0b5b9` | [nasaharvest/galileo](https://huggingface.co/nasaharvest/galileo) `f039dd5`, `models/nano/encoder.pt` | `eab6acacf527…` | MIT |
| OlmoEarth | [allenai/olmoearth_pretrain](https://github.com/allenai/olmoearth_pretrain) `db13d68` | [allenai/OlmoEarth-v1_2-Nano](https://huggingface.co/allenai/OlmoEarth-v1_2-Nano) `e1f693a`, `weights.pth` | `2773fca48c23…` | [OlmoEarth Artifact License](https://huggingface.co/allenai/OlmoEarth-v1_2-Nano/blob/main/LICENSE.txt) |
| TESSERA | [ucam-eo/tessera](https://github.com/ucam-eo/tessera) `e368539`, `tessera_infer_v2/student` | [geotessera/TESSERA-V-2.0-2B-N](https://huggingface.co/geotessera/TESSERA-V-2.0-2B-N) `bd08c90`, `ckpt/student_nano.pt` | `bc9929e3643d…` | code MIT, weights CC0 |

Galileo's `config/normalization.json` must sit next to `single_file_galileo.py` (the image does this). The OlmoEarth
weights and their outputs may not be used for military and defence applications, human surveillance and policing, or
extractive activities such as mining and deforestation (Section 2 of its licence).

## Use your own data

| You provide | Format |
|---|---|
| Field boundaries on a 10 m grid | `FIELDS.json` (`x0`, `y1`, `w`, `h`, `crs`) and `PIXELS.parquet` (`field`, `row`, `col`, `lon`, `lat`) |
| Season dates | `FIELDS.parquet` (`field`, `n_pix`, `sow_day`, `harvest_day` as UTC day numbers) |
| Crop | maize, soybean or wheat settings (`docs/DATA_COLLECTION.md`, decision times) |
| Yield maps of earlier seasons (optional) | npz with `target` (t/ha) and `gt_valid` per pixel, in the pixel order of the seasons given (as `out/run/yield_history.npz`); pass with `--labels <file>` to `fieldwise.tutorial.run` |

| The code produces | |
|---|---|
| Decision times and cases, inputs from public products | `fieldwise.collect` |
| Foundation-model embeddings | `fieldwise.collect.encode` |
| Synthetic yields (only without `--labels`), Base models, RPT database, fitted corrections | `fieldwise.tutorial.run` |
| New-season predictions | `fieldwise.tutorial.infer` |

The Base architecture and the four foundation models are fixed in this release. Fit one crop at a time.

## AI use

The code and documentation in this repository were written with the assistance of AI coding tools, Claude Code
(Anthropic) and Codex (OpenAI). The authors designed the method, reviewed and tested the code, and take
responsibility for its content.

## Citation and data

The paper, *FieldWise: Explanation fidelity and decision value in residual correction of subfield crop yield
predictions* (Chan and Yang), is under review; its citation will be added upon publication. To cite the software, see
`CITATION.cff`.

- Code: MIT (`LICENSE`).
- Demo inputs: derived from public products under their providers' terms; notices and licences in the package's
  `NOTICE.md`. The tutorial's inputs are real public data; only the yields are synthetic.
- The manuscript's results were obtained on YieldSAT (Miranda et al., 2026), available from DFKI at
  https://yieldsat.github.io/. No YieldSAT data or models trained on it are distributed here; Fig. 1 is the
  manuscript's figure of its example field-season.
- Foundation models: TESSERA (Feng et al., 2026), Galileo (Tseng et al., 2025), OlmoEarth (Herzog et al., 2026) and
  Presto (Tseng et al., 2023). Their code and weights are not included; `Dockerfile.encode` downloads them from the
  official sources, under their own licences (see [Foundation models](#foundation-models)). Please cite them when you
  use the encoders.

Feng, Z., Atzberger, C., Jaffer, S., Knezevic, J., Sormunen, S., Young, R., Lisaius, M.C., Immitzer, M., Jackson, T.,
Ball, J., Coomes, D.A., Madhavapeddy, A., Blake, A., Keshav, S., 2026. TESSERA: Temporal embeddings of surface spectra
for Earth representation and analysis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pp. 34818-34831.

Herzog, H., Bastani, F., Zhang, Y., Tseng, G., Redmon, J., Sablon, H., Park, R., Morrison, J., Buraczynski, A.,
Farley, K., Hansen, J., Howe, A., Johnson, P.A., Otterlee, M., Schmitt, T., Pitelka, H., Daspit, S., Ratner, R.,
Wilhelm, C., Wood, S., Jacobi, M., Kerner, H., Shelhamer, E., Farhadi, A., Krishna, R., Beukema, P., 2026. OlmoEarth:
Stable latent image modeling for multimodal Earth observation. In: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), pp. 34806-34817.

Miranda, M., Pathak, D., Helber, P., Bischke, B., Najjar, H., Mena, F., Sanchez, C., Pai, A., Arenas, D.,
Valdenegro-Toro, M., Charfuelan, M., Nuske, M., Dengel, A., 2026. YieldSAT: A multimodal benchmark dataset for
high-resolution crop yield prediction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pp. 22920-22930.

Tseng, G., Cartuyvels, R., Zvonkov, I., Purohit, M., Rolnick, D., Kerner, H., 2023. Lightweight, pre-trained
transformers for remote sensing timeseries. In: NeurIPS 2023 Workshop on Tackling Climate Change with Machine Learning.
https://www.climatechange.ai/papers/neurips2023/58

Tseng, G., Fuller, A., Reil, M., Herzog, H., Beukema, P., Bastani, F., Green, J.R., Shelhamer, E., Kerner, H.,
Rolnick, D., 2025. Galileo: Learning global & local features of many remote sensing modalities. In: Proceedings of the
42nd International Conference on Machine Learning, PMLR 267, pp. 60280-60300.
