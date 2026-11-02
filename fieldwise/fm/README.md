# Foundation-model inputs

`views.py` builds the input of each foundation model for one field and one observation window; `encode.py` runs the
frozen encoders. `Dockerfile.encode` installs the official code and weights at the versions used for FieldWise
(docs/FOUNDATION_MODELS.md); other installations pass their locations with `encode.ModelPaths`.

| Model | Products | Time steps |
|---|---|---|
| TESSERA v2 (student) | Sentinel-2 L2A (10 bands, clear acquisitions), Sentinel-1 RTC gamma0 (ascending and descending) | every acquisition, day of year |
| Galileo | Sentinel-1 GRD, Sentinel-2 L1C quality mosaic (Cloud Score+), NDVI, ERA5-Land, TerraClimate, VIIRS, SRTM, Dynamic World, WorldCereal 2021, LandScan, location | 30 days |
| Presto | Sentinel-1 GRD, Sentinel-2 L1C quality mosaic, ERA5-Land, SRTM, Dynamic World labels, location | 30 days |
| OlmoEarth v1.2 | Sentinel-2 L2A (12 bands), Sentinel-1 GRD | 30 days |

## Reader interface

`views.Field(D, fid, window)` reads through an object `D` with these methods. A window is `(start_ms, end_ms, decision_ms)`:
observations in `start <= t < end` that are available (time plus latency) by `decision`.

| Method | Returns (dict) |
|---|---|
| `pixels(fid)` | `lon`, `lat` of the service pixels |
| `s2(fid, window, positions)` | Sentinel-2 L2A: `values` [images, pixels, 13] (12 bands + SCL, NaN where absent), `found`, `valid`, `acquisition_ms`, `boa_add_offset`, `bands` |
| `s1_rtc(fid, window, latency_hours, positions)` | Sentinel-1 RTC: `values` [images, pixels, 2] linear gamma0 (VV, VH), `found`, `valid`, `acquisition_ms`, `orbit`, `keys` |
| `image(product, fid, window, latency_hours, positions)` | `s1_grd`, `s2_l1c`, `cloud_score_plus`, `dynamic_world`, `viirs_dnb_monthly`, `landscan_global`, `worldcereal_2021`: `values` [images, pixels, bands], `found`, `valid`, `keys`, `acquisition_ms`, `interval_start_ms`, `interval_end_ms`, `bands` |
| `weather('era5_land', fid, window)` | hourly ERA5-Land rows of the grid cells around the field: `values` [rows, 19], `valid`, `time_ms`, `cell_offsets`, `weights`, `bands` |
| `product('terraclimate', fid)` | monthly `values` [months, bands] (physical units), `valid`, `time_start_ms`, `time_end_ms`, `bands` |
| `static(fid)` | `terrain_elevation_m`, `terrain_slope_rise` per pixel |

`fieldwise/collect/reader.py` implements this interface on products extracted from Google Earth Engine and Microsoft
Planetary Computer.
