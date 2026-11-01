"""Public maize data workflow: build_terrain. Run through python -m fieldwise.collect; see docs/DATA_COLLECTION.md."""
import json
import os
import sys
from pathlib import Path

from .settings import config, project, weather_cells, month_edges
import numpy as np
from pyproj import Transformer
from scipy import ndimage

RAW, FD, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
G = json.loads((FD / 'FIELDS.json').read_text())['grid']
z = np.load(RAW / 'srtm/SRTM.npz')
dem_ll, x0ll, y1ll, step = z['elevation'].astype(np.float64), float(z['x0']), float(z['y1']), float(z['step'])
assert (dem_ll > -32768).all(), 'SRTM voids in the domain'
L = 30.0
fwd = Transformer.from_crs('EPSG:4326', G['crs'], always_xy=True)
inv = Transformer.from_crs(G['crs'], 'EPSG:4326', always_xy=True)
h_ll, w_ll = dem_ll.shape
# 30 m UTM domain inside the SRTM extent (corners transformed, shrunk by 2 cells)
cx, cy = fwd.transform(np.array([x0ll, x0ll + w_ll * step, x0ll, x0ll + w_ll * step]), np.array([y1ll, y1ll, y1ll - h_ll * step, y1ll - h_ll * step]))
X0 = np.ceil(max(cx[0], cx[2]) / L) * L + 2 * L
X1 = np.floor(min(cx[1], cx[3]) / L) * L - 2 * L
Y1 = np.floor(min(cy[0], cy[1]) / L) * L - 2 * L
Y0 = np.ceil(max(cy[2], cy[3]) / L) * L + 2 * L
nx, ny = int((X1 - X0) / L), int((Y1 - Y0) / L)
xc = X0 + (np.arange(nx) + 0.5) * L
yc = Y1 - (np.arange(ny) + 0.5) * L
XX, YY = np.meshgrid(xc, yc)
lon, lat = inv.transform(XX, YY)
# point-bilinear on the 1 arcsecond grid (sample centres at x0ll + (j + 0.5) step)
fj = (lon - x0ll) / step - 0.5
fi = (y1ll - lat) / step - 0.5
j0, i0 = np.floor(fj).astype(int), np.floor(fi).astype(int)
tj, ti = fj - j0, fi - i0
assert j0.min() >= 0 and i0.min() >= 0 and j0.max() + 1 < w_ll and i0.max() + 1 < h_ll
Z = ((1 - ti) * (1 - tj) * dem_ll[i0, j0] + (1 - ti) * tj * dem_ll[i0, j0 + 1] + ti * (1 - tj) * dem_ll[i0 + 1, j0]
     + ti * tj * dem_ll[i0 + 1, j0 + 1])

P = np.pad(Z, 1, mode='edge')                                  # 'nearest'
a, b, c = P[:-2, :-2], P[:-2, 1:-1], P[:-2, 2:]
d, e, f = P[1:-1, :-2], P[1:-1, 1:-1], P[1:-1, 2:]
g, hh, i = P[2:, :-2], P[2:, 1:-1], P[2:, 2:]
dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * L)            # east
dzdy = ((a + 2 * b + c) - (g + 2 * hh + i)) / (8 * L)           # north (row 0 is north)
rise = np.hypot(dzdx, dzdy)
asp_def = rise >= 0.001
aspect = np.where(asp_def, np.degrees(np.arctan2(-dzdx, -dzdy)) % 360, np.nan)
D_ = ((d + f) / 2 - e) / L ** 2
E_ = ((b + hh) / 2 - e) / L ** 2
lap = 2 * (D_ + E_)
G_ = (f - d) / (2 * L)
H_ = (b - hh) / (2 * L)
curv_def = asp_def & ((G_ ** 2 + H_ ** 2) > 0)

if not hasattr(np, 'in1d'):                                     # pysheds 0.5 calls np.in1d (removed in numpy 2; np.isin is equivalent)
    np.in1d = np.isin
from pysheds.grid import Grid                                   # noqa: E402
from pysheds.view import Raster, ViewFinder                     # noqa: E402
from affine import Affine                                       # noqa: E402

vf = ViewFinder(affine=Affine(L, 0, X0, 0, -L, Y1), shape=Z.shape, nodata=np.nan)
r = Raster(Z.copy(), viewfinder=vf)
grid = Grid(viewfinder=vf)
filled = grid.fill_depressions(grid.fill_pits(r))
flat = grid.resolve_flats(filled)
fdir = grid.flowdir(flat)
acc = np.asarray(grid.accumulation(fdir), np.float64)           # includes the cell itself
twi = np.log(acc * L / np.maximum(rise, 0.001))
edge = np.zeros(Z.shape)
edge[:2, :] = edge[-2:, :] = edge[:, :2] = edge[:, -2:] = 1
contact = np.asarray(grid.accumulation(fdir, weights=Raster(edge, viewfinder=vf)), np.float64) - edge > 0

# 10 m service pixels -> the 30 m cell holding the pixel centre
fid = np.load(FD / 'field_id.npy')
rr, cc = np.nonzero(fid >= 0)
px = G['x0'] + (cc + 0.5) * 10
py = G['y1'] - (rr + 0.5) * 10
ci = ((Y1 - py) // L).astype(int)
cj = ((px - X0) // L).astype(int)
assert ci.min() >= 0 and cj.min() >= 0 and ci.max() < ny and cj.max() < nx
lay = dict(aspect_deg=aspect, laplacian_per_m=np.where(curv_def, lap, np.nan), elevation_m=Z, slope_deg=np.degrees(np.arctan(rise)),
           twi=twi, boundary_contact=contact.astype(float))
out = {k: v[ci, cj].astype(np.float32) for k, v in lay.items()}
out.update(row=rr, col=cc, field=fid[rr, cc])
tmp = OUT / 'TERRAIN.tmp.npz'
np.savez_compressed(tmp, **out)
os.replace(tmp, OUT / 'TERRAIN.npz')
summ = dict(domain_cells=[ny, nx], domain_km=[ny * L / 1000, nx * L / 1000],
            **{k: dict(min=float(np.nanmin(v)), median=float(np.nanmedian(v)), max=float(np.nanmax(v)), nan_share=float(np.isnan(v).mean()))
               for k, v in out.items() if k not in ('row', 'col', 'field')})
tmp = OUT / 'TERRAIN.tmp.json'
tmp.write_text(json.dumps(summ, indent=1))
os.replace(tmp, OUT / 'TERRAIN.json')
print(json.dumps(summ, indent=1))

