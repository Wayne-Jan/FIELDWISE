"""Daily weather quantities and seasonal exposure summaries from hourly ERA5-Land field values.

Hourly input X: [field-seasons, days, 24, bands], one UTC day per row; instantaneous bands sit at the hour of t and
one-hour accumulations at the hour of t - 1 h; NaN where no grid cell has a valid value.
"""
import numpy as np

DAILY = ('tmean', 'tmax', 'tmin', 'vpd_max', 'precip', 'ssrd', 'total_evap', 'pot_evap', 'theta_rz', 'theta_l4')
EXPOSURES = ('E1_total', 'E1_14d', 'E2_total', 'E2_14d', 'E4_cum_precip', 'E3_theta_last7', 'E3_theta_min')
HEAT_THRESHOLD = {'corn': 30., 'soybean': 32., 'wheat': 30.}


def saturation_vapour_pressure(t):
    """Tetens, kPa, t in degrees C."""
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def _band(X, bands, name):
    return np.asarray(X[..., bands.index(name)], dtype=np.float64)


def root_zone(X, bands):
    """Root-zone soil water: .07 layer 1 + .21 layer 2 + .72 layer 3."""
    return (.07 * _band(X, bands, 'volumetric_soil_water_layer_1') + .21 * _band(X, bands, 'volumetric_soil_water_layer_2')
            + .72 * _band(X, bands, 'volumetric_soil_water_layer_3'))


def daily_quantities(X, bands):
    """[field-seasons, days, 10] daily values in the order of DAILY; a day with a missing hour is missing."""
    T = _band(X, bands, 'temperature_2m') - 273.15
    Td = _band(X, bands, 'dewpoint_temperature_2m') - 273.15
    vpd = saturation_vapour_pressure(T) - saturation_vapour_pressure(Td)
    return np.stack([T.mean(-1), T.max(-1), T.min(-1), vpd.max(-1),
                     _band(X, bands, 'total_precipitation_hourly').sum(-1) * 1000,
                     _band(X, bands, 'surface_solar_radiation_downwards_hourly').sum(-1) / 1e6,
                     _band(X, bands, 'total_evaporation_hourly').sum(-1) * 1000,
                     _band(X, bands, 'potential_evaporation_hourly').sum(-1) * 1000,
                     root_zone(X, bands).mean(-1), _band(X, bands, 'volumetric_soil_water_layer_4').mean(-1)], -1)


def daily_exposure_series(X, bands, crop):
    """[field-seasons, days, 4]: heat degree-days, hours with VPD > 2 kPa, mean root-zone soil water and precipitation (mm).

    Missing hours are skipped (a day without valid hours has 0 heat, 0 VPD hours and 0 precipitation, and missing soil
    water).
    """
    T = _band(X, bands, 'temperature_2m') - 273.15
    Td = _band(X, bands, 'dewpoint_temperature_2m') - 273.15
    heat = np.where(np.isfinite(T), np.maximum(T - HEAT_THRESHOLD[crop], 0) / 24, np.nan)
    vpd2 = np.where(np.isfinite(T) & np.isfinite(Td),
                    (saturation_vapour_pressure(T) - saturation_vapour_pressure(Td) > 2).astype(float), np.nan)
    rz = root_zone(X, bands)
    with np.errstate(invalid='ignore'):
        theta = np.where(np.isfinite(rz).any(-1), np.nanmean(np.where(np.isfinite(rz), rz, np.nan), -1), np.nan)
    precip = np.nansum(_band(X, bands, 'total_precipitation_hourly'), -1) * 1000
    return np.stack([np.nansum(heat, -1), np.nansum(vpd2, -1), theta, precip], -1)


def exposures(series, seed_index, usable_index):
    """Seven exposure summaries for the cases of one field-season, in the order of EXPOSURES.

    series: [days, 4] from daily_exposure_series(); seed_index: row of the seeding day; usable_index: row of the last usable
    day of every case. Sums run from the seeding day; 14-day windows end on the last usable day; the soil water of the last
    seven days is the mean over the finite days among the last seven rows.
    """
    s = series[seed_index:]
    hi = np.asarray(usable_index) - seed_index + 1
    lo14 = np.maximum(hi - 14, 0)
    lo7 = np.maximum(hi - 7, 0)
    cum = lambda x: np.concatenate([[0.], np.cumsum(np.nan_to_num(x))])  # noqa: E731
    rz = s[:, 2]
    fr = np.isfinite(rz)
    Sr, Nr = cum(np.where(fr, rz, 0)), cum(fr)
    rmin = np.concatenate([[np.nan], np.fmin.accumulate(rz)])
    H, V, P = cum(s[:, 0]), cum(s[:, 1]), cum(s[:, 3])
    n7 = Nr[hi] - Nr[lo7]
    with np.errstate(invalid='ignore', divide='ignore'):
        last7 = np.where(n7 > 0, (Sr[hi] - Sr[lo7]) / n7, np.nan)
    return np.column_stack([H[hi], H[hi] - H[lo14], V[hi], V[hi] - V[lo14], P[hi], last7, rmin[hi]])
