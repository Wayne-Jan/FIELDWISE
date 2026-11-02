# Collecting input data

FieldWise needs, for every field, a 10 m pixel grid, sowing and harvest dates, and public Earth-observation, weather,
soil and terrain products. Yield maps are needed only to fit or evaluate models and to enable within-field correction
(RPT) with your own reference fields. This page lists every product, where it comes from and how it is used;
`fieldwise.collect` downloads them (README, Build your own). `docs/INPUTS.md` defines the prepared inputs, their
units and timing rules, and `fieldwise/validate.py` checks them.

## 1. Fields and service pixels

- **Field boundaries.** Your own boundaries, or public parcels such as OpenStreetMap `landuse=farmland` polygons (ODbL: keep
  the attribution and share derived boundaries under the same licence).
- **Service pixels.** A 10 m grid in the local UTM zone; a pixel belongs to a field when its centre falls inside the field.
  Every product is sampled at these pixel centres.
- **Sowing and harvest dates.** From farm records when available. Otherwise they can be estimated from Sentinel-2 NDVI
  phenology (start and end of season; fieldwise/collect/season_proxies.py); record that they are estimates.

## 2. Decision times

Growing degree days (GDD) are accumulated from sowing as the sum of max(T_mean - T_base, 0), where T_mean is the mean of the
24 hourly ERA5-Land temperatures of each UTC day at the field. A decision stage is reached on the first day the cumulative
GDD reaches its threshold.

| Crop | T_base | Thresholds tau1-tau4 (degree C day) |
|---|---|---|
| Maize, soybean | 10 C | 400, 800, 1,200, 1,600 |
| Wheat | 0 C | 600, 1,200, 1,800, 2,400 |

The observation window runs from 00:00 UTC on the sowing date to E = 00:00 UTC on the day after the threshold is reached;
the decision is made at C = E + 6 days. A product observation is usable when it falls inside the window and is available
(observation time plus the delay below) by C. Yield labels count as available 30 days after harvest.

## 3. Products

Values are kept as delivered (native units, native pixel of each image; no resampling): each service pixel takes the value
of the native cell under its centre. Fill values are kept as missing, together with each image's source identity.

| Product | Source | Bands | Used by | Availability delay |
|---|---|---|---|---|
| Sentinel-2 L2A | GEE `COPERNICUS/S2_SR_HARMONIZED` | B1-B12 (no B10), SCL | Base, reference retrieval, TESSERA, OlmoEarth | 24 h |
| Sentinel-2 L1C | GEE `COPERNICUS/S2_HARMONIZED` | 13 spectral bands | Galileo, Presto (quality mosaic) | 24 h |
| Cloud Score+ | GEE `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` | cs, cs_cdf | Galileo, Presto (mosaic key) | 24 h |
| Sentinel-1 GRD | GEE `COPERNICUS/S1_GRD` (IW, VV + VH) | VV, VH (dB), angle | Galileo, Presto, OlmoEarth | 48 h |
| Sentinel-1 RTC | Microsoft Planetary Computer `sentinel-1-rtc` | VV, VH (gamma0, linear power) | TESSERA | 48 h |
| Dynamic World | GEE `GOOGLE/DYNAMICWORLD/V1` | label, 9 class probabilities | Galileo, Presto | 48 h |
| ERA5-Land hourly | GEE `ECMWF/ERA5_LAND/HOURLY` | 19 bands (temperature, dew point, precipitation, radiation, wind, pressure, evaporation, soil water and temperature) | Base, decision times, reference retrieval, environmental offset G, Galileo, Presto | 6 days |
| MODIS LST | GEE `MODIS/061/MOD11A1` | LST_Day_1km, LST_Night_1km, QC_Day, QC_Night | Environmental offset G | 2 days |
| SoilGrids 2.0 | ISRIC files `https://files.isric.org/soilgrids/latest/data/` | cec, cfvo, clay, nitrogen, phh2o, sand, silt, soc; 0-5, 5-15, 15-30 cm mean | Base, reference retrieval | static |
| SRTM | GEE `USGS/SRTMGL1_003` | elevation | Base, reference retrieval, Galileo, Presto (with derived slope, aspect, curvature, TWI) | static |
| TerraClimate | GEE `IDAHO_EPSCOR/TERRACLIMATE` | aet, def, soil | Galileo | months completed by E |
| VIIRS DNB monthly | GEE `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG` | avg_rad, cf_cvg | Galileo | months completed by E |
| LandScan Global | GEE `projects/sat-io/open-datasets/ORNL/LANDSCAN_GLOBAL` | b1 | Galileo | latest year starting by E |
| WorldCereal 2021 | GEE `ESA/WorldCereal/2021/MODELS/v100` | classification (five products) | Galileo | the 2021 release for every season |

Notes:
- **Sentinel-2 L2A.** Collect every acquisition of every MGRS tile covering the field (one image per acquisition and tile).
  A step is valid when all 12 bands are present; SCL is kept for the scene classes.
- **Sentinel-1 RTC.** TESSERA expects radiometrically terrain-corrected gamma0; GRD sigma0 is not a substitute.
- **ERA5-Land.** The field value of an hour is the bilinear combination of the four 0.1 degree cells around the field
  centre, renormalised over the cells with a valid value; one-hour accumulations belong to the hour t - 1 h.
- **MODIS LST.** Each pixel maps to the MODIS sinusoidal 1 km cell under its centre; an observation is valid with mandatory
  QC <= 1 and LST error class <= 1, and a field day is kept when valid cells cover at least half of the service pixels.
- **SoilGrids.** Read the official ISRIC files: the copy in the GEE community catalogue differed by about a factor of ten
  for nitrogen at the Taiwan test site.
- **Terrain.** Slope (Horn), aspect, Laplacian curvature and the topographic wetness index (pysheds D8 with pit and flat
  filling) are computed on a 30 m grid from SRTM with a 20 km margin; each 10 m pixel takes the cell under its centre.

## 4. Your own yield maps

Yield maps let you fit models for your region and turn on within-field correction with your own reference fields.
- Clean combine-harvester points (remove start and end of passes, overlaps and outliers) and aggregate them to the 10 m
  service grid; keep a valid mask for pixels with enough observations.
- Store yield in t/ha with the field, season, sowing and harvest dates.
- A season's map can serve as a reference only after its labels are available (harvest + 30 days) and only for later
  seasons.

YieldSAT (Miranda et al., 2026) provides such maps for Argentina, Brazil, Uruguay and Germany under its own licence; request
access at https://yieldsat.github.io/.

## 5. Attribution

Contains modified Copernicus Sentinel data (ESA); Sentinel-1 RTC processed by Catalyst and hosted by Microsoft Planetary
Computer; Dynamic World contains modified Copernicus data; ERA5-Land is provided by the Copernicus Climate Change Service
through ECMWF; LandScan is produced by Oak Ridge National Laboratory. Use every product under its provider's terms.
