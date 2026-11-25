<p align="center"><img src="docs/images/logo.png" alt="FieldWise" width="420"></p>

**FieldWise corrects pixel-level crop yield predictions for a new season before harvest—without retraining the base
model—using within-field residual patterns and field-wide offsets learned from earlier seasons whose harvest labels are
already available by the decision time.**

This repository provides (1) a direct Python API (`fieldwise.correct_pixels`) for applying FieldWise to **any
pixel-level dataset**, (2) the full Base training and Earth-observation pipeline (supporting both **truncated-season**
and **whole-season** training), and (3) the **explainability and decision-value diagnostics** (`fieldwise.explain`)
accompanying the FieldWise manuscript (in preparation).

![Fig. 1 of the manuscript](docs/images/figure1.png)

*Fig. 1 of the manuscript: Conceptual overview of FieldWise two-scale post-hoc correction of a preharvest yield map
(wheat, Argentina, 2023, τ2). The base forecast uses satellite and weather observations up to the cutoff; reference
fields contribute residuals only if their harvest labels were available by the forecast date.*

## How it works

![Fig. 4 of the manuscript](docs/images/figure4.png)

*Fig. 4 of the manuscript: FieldWise workflow on an illustrative wheat field-season from YieldSAT (Miranda et al., 2026)
at the third in-season decision τ3 (Argentina, 2023-11-10, Transformer Base, truncated-season training). Clock icons mark
components fitted on earlier seasons.*

FieldWise adds two scale-separated corrections to any pixel-level Base prediction:
- a **within-field correction** β·S15, transferred from the residual maps of similar earlier fields via pixel-to-pixel
  matching and 15 m smoothing (residual pattern transfer, RPT), and
- a **field-wide offset** u, chosen by a shrinkage–reweighting decision layer from an environmental offset G, a linear
  calibration offset Cal (K in the equations) and residual readouts of four Earth-observation foundation models.

Every component is fitted only on earlier seasons whose harvest labels are already available by the decision time.

| Panel | Step | Code |
|---|---|---|
| A | Base model (Transformer or 3D-LSTM; truncated-season or whole-season training) and earlier-season residuals | `fieldwise/base/` (`truncation.py`, `train.py`) |
| B | Timing and label-availability schedule: observations end at the cutoff; earlier-season yield labels are assumed available 30 days after harvest | `fieldwise/schedule.py`, `fieldwise/selection.py` |
| C | Within-field correction (RPT): reference-field retrieval, learned pixel matching, 15 m smoothing (S15) | `fieldwise/retrieval.py`, `fieldwise/rpt.py` |
| D | Field-wide offset: environmental (G) and calibration (Cal) offsets, foundation-model readouts and shrinkage–reweighting decision layer | `environment.py`, `calibration.py`, `readout.py`, `decision.py` |
| E | Final pixel-level prediction: `Base + β·S15 + u` (vs. `Correction only = Base + β·S15 + c`) | `fieldwise/pipeline.py`, `fieldwise/tutorial/infer.py` |

## Use directly with your own pixel-level data

If you already have pixel-level predictions from your own Base model, historical pixel-level yield maps, and pixel-level
remote-sensing features or foundation-model embeddings, you can run FieldWise directly—no Earth Engine download or Base
retraining required:

```python
import pandas as pd
import numpy as np
import fieldwise

# pixels: one row per pixel with required columns:
#   case_id, year, field, stage, pred, target (target may be NaN for the unlabelled target year)
# optional columns:
#   gt_valid, decision_ms, label_ms, and pixel coordinates (lon, lat) or (x, y) for 15 m smoothing
pixels = pd.read_parquet("my_pixels.parquet")

# emb: [N_pixels, D] array aligned with pixels (foundation-model embeddings or pixel-level spectral/temporal features)
emb = np.load("my_pixel_embeddings.npy")

# Runs rolling temporal retrieval -> RPT -> calibration -> forward strength selection
# (pass G=... and rhat=... to include environmental offsets and foundation-model readouts)
res = fieldwise.correct_pixels(pixels, emb, k=50, radius=15.0, device="cuda")

# Output DataFrame in the exact input row order with:
#   pred_base, s15, within_field, field_offset, pred_correction_only, pred_fieldwise
print(res.pixels[["case_id", "pred_base", "within_field", "field_offset", "pred_fieldwise"]].head())
print("Selected strengths per year:", res.choices)
```

> **Temporal-legality guarantee:** Target year `T` only ever uses labelled reference fields from years `< T` whose
> harvest labels are available by the decision time (enforced by `fieldwise/selection.py` and verified by
> `tests/test_pipeline.py` and `tests/test_leakage.py`).

## Explainability and decision diagnostics

`fieldwise/explain/` provides building blocks for the scale decomposition, explanation fidelity tests and
decision-value diagnostics in manuscript Sections 4.4–4.7 and Figs. 7–12 (descriptor matrices are supplied by the
caller):

| Manuscript analysis | What it tests | Module / functions |
|---|---|---|
| Spatial-scale error layers & sequential increments (Fig. 7, Table 1) | Additive decomposition of MSE into within-field, remaining between-field and group-shared bias layers; sequential and component contributions | `fieldwise/explain/errors.py` (`Layers`, `correction_states`, `correction_contributions`) |
| Spatial fidelity: edge contrast & slope (Fig. 8) | Boundary-distance contrast error (10–20 m vs. ≥ 40 m) and within-field response slope with cluster bootstrap | `fieldwise/explain/spatial.py` (`boundary_distance`, `edge_contrast`, `response_slope`, `cluster_weights`) |
| Within-field mechanism & reconstruction (Fig. 9) | Pixel-matching shuffle ablation, LEACE concept erasure with matched random controls, and ridge reconstruction from caller-supplied descriptors (R² and retained benefit) | `fieldwise/explain/spatial.py` (`shuffle_residuals`, `reconstruct_ridge`, `retained_benefit`), `fieldwise/explain/erasure.py` (`fit_erasers`, `leace`) |
| Direct field-mean correction (Fig. 10) | Direct application of field-mean residual estimates before the decision layer, predictive skill and supplementary readout increments | `fieldwise/explain/decisions.py` (`readout_skill`, `loss_change`), `fieldwise/explain/increments.py` (`fit_blocks`) |
| Field-specific residual pairing (Fig. 11) | Within-country–year–stage permutation of field-mean residual estimates to test own-field pairing value | `fieldwise/explain/decisions.py` (`permuted_readouts`, `permutation_groups`, `permutation_losses`) |
| Decision-layer accuracy & severe-harm risk (Fig. 12, Table 2) | Revising the field-wide offset vs. *Correction only*, severe-harm rate (> 1.1 × Base MSE) and fixed-fraction baselines | `fieldwise/explain/decisions.py` (`layer_offsets`, `fraction_losses`, `loss_change`) |
| Per-pixel attribution & supplementary erasure | Exact additive breakdown of served predictions and supplementary field-wide readout erasure | `fieldwise/explain/inference.py`, `fieldwise/explain/erasure.py` (`readout_erasure`) |

## Quick start (end-to-end tutorial)

Needs Docker and an NVIDIA GPU with the NVIDIA Container Toolkit. The demo inputs are seven seasons (2019/20–2025/26) of
10 areas near Tainan, Taiwan; the last season is the new one. The three steps take about 3.5 minutes on one NVIDIA TITAN RTX.

```bash
git clone https://github.com/Wayne-Jan/FIELDWISE.git && cd FIELDWISE
docker build -t fieldwise .
docker run --rm --init fieldwise python -m unittest discover -s tests     # unit, pipeline and leakage tests

# demo inputs, 44 MB
curl -LO https://github.com/Wayne-Jan/FIELDWISE/releases/download/demo-data-v1/fieldwise_tainan_demo_v1.tar.gz
tar -xzf fieldwise_tainan_demo_v1.tar.gz
mkdir -p out

RUN="docker run --rm --init --gpus all -v $PWD/fieldwise_tainan_demo_v1:/data:ro -v $PWD/out:/out fieldwise"
SEASONS="/data/2019 /data/2020 /data/2021 /data/2022 /data/2023 /data/2024 /data/2025"

# 1. fit: Base (--strategy truncated by default, matching the manuscript; or whole), RPT database and corrections
$RUN python -m fieldwise.tutorial.run --seasons $SEASONS --target 2026 --out /out/run --strategy truncated --seeds 42 --epochs 3
# 2. serve the new season from the saved model and database, without its yields and without refitting
$RUN python -m fieldwise.tutorial.infer --model /out/run/model --database /out/run/rpt_database --season /data/2025 --out /out/new
# 3. draw the maps
$RUN python docs/tutorial_figures.py --run /out/run --infer /out/new --season /data/2025 --stage tau2 --out /out/figures
```

![The new season predicted without labels](docs/images/inference.png)

The new season predicted without its yields (shown here for the full tutorial collection of about 120 areas; the quick
start draws its 10). `out/run` holds the fitted model and the RPT database, `out/new` the new season per pixel and field
([files](docs/INPUTS.md#12-tutorial-files)). The manuscript setting is five seeds and 55 epochs
(`--seeds 42 43 44 45 46 --epochs 55`). Pass your own pixel-level yield maps directly via `--labels my_yields.npz`.

## Build your own region from Earth Engine

One folder per season, oldest first. Build the collection and encoding images, then open the collection image with your
[Earth Engine](https://developers.google.com/earth-engine/guides/access) project:

```bash
docker build -f Dockerfile.collect -t fieldwise-collect .
docker build -f Dockerfile.encode -t fieldwise-encode .
export FIELDWISE_EE_PROJECT=<your project>     # after earthengine authenticate
docker run --rm --init -it -e FIELDWISE_EE_PROJECT -e PYTHONPATH=/workspace/FIELDWISE \
  -v ~/.config/earthengine:/root/.config/earthengine:ro -v $PWD/work:/work -w /work fieldwise-collect bash
```

1. **Fields and dates.** Your field boundaries and season dates ([format](docs/INPUTS.md#12-tutorial-files)), or, as in
   the Tainan demo, areas and season dates estimated from Sentinel-2 NDVI:
   ```bash
   python -m fieldwise.collect.extract_s2_season raw/s2_season <x0> <y1> <w> <h> <Y>-02-01 <Y+1>-08-01
   python -m fieldwise.collect.season_proxies raw/s2_season <parcels.gpkg> <Y> <cx> <cy> 10 3000 fields
   ```
2. **Collect** the public products of sowing year `Y` ([products](docs/DATA_COLLECTION.md)):
   ```bash
   python -m fieldwise.collect.init_fields fields <season>/collect <Y>
   for step in base soil terrain build-base lst build-environment fm rtc-pre rtc; do
     python -m fieldwise.collect $step --root <season>/collect --config <season>/collect/collection.json
   done
   ```
3. **Encode** with the four foundation models in the encoding image, from the host; a GPU is recommended
   ([models](docs/FOUNDATION_MODELS.md)):
   ```bash
   docker run --rm --init --gpus all -v $PWD/work/<season>:/season fieldwise-encode \
     python -m fieldwise.collect.encode /season/collect /opt/fm/paths.json /season/encode
   ```
4. **Fit and serve** as in the quick start. Pass your pixel-level yield maps with `--labels`; without them the tutorial
   generates synthetic yields. `tests/test_leakage.py` checks that the new season's yields never enter a fit.

Table formats and the timing rules that keep every fit to information available at the decision:
[docs/INPUTS.md](docs/INPUTS.md).

## Citation, data and licences

- **Paper:** *FieldWise: Two-scale post-hoc residual correction, explanation fidelity, and decision value in preharvest
  subfield yield forecasting* (manuscript in preparation; full citation will be added upon publication). To cite the
  software, see `CITATION.cff`.
- **Data:** the manuscript's experiments use YieldSAT (Miranda et al., 2026), available from DFKI at
  https://yieldsat.github.io/.
- **Foundation models:** please cite TESSERA (Feng et al., 2026), Galileo (Tseng et al., 2025), OlmoEarth (Herzog et al.,
  2026) and Presto (Tseng et al., 2023) when you use the encoders; each keeps its own licence
  ([versions and licences](docs/FOUNDATION_MODELS.md#versions-and-licences)).
- **Licences:** code MIT (`LICENSE`); demo inputs under their providers' terms (`NOTICE.md` in the package).

<details>
<summary>References</summary>

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

</details>

## AI use

The code and documentation in this repository were written with the assistance of AI coding tools, Claude Code
(Anthropic) and Codex (OpenAI). The authors designed the method, reviewed and tested the code, and take
responsibility for its content.
