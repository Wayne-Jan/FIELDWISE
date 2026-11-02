# Foundation models

FieldWise uses four frozen Earth-observation foundation models. TESSERA's per-pixel embeddings give the pixel correspondence
of RPT; the field means of all four feed the residual readouts. Full references are in the [README](../README.md).

## Inputs

What each model reads in FieldWise, before its official normalisation (`fieldwise/fm/views.py`):

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
the top of `fieldwise/fm/views.py`. Products, bands and availability delays: [`DATA_COLLECTION.md`](DATA_COLLECTION.md).

## Versions and licences

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
