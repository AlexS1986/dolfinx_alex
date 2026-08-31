#!/usr/bin/env python3
"""
Small microstructure npz for the LOCAL SMOKE TEST of 071.

Two modes:

  crop      cut a small window out of an existing micro_<tag>.npz produced by
            070/preprocess_ebsd_to_grid.py - real EBSD data, just small
              python3 make_test_micro.py crop --src micro_long.npz \
                     --window 0 0 400 200 --out micro_test.npz

  synth     build a synthetic columnar-grain patch with the same file layout,
            so the smoke test runs with NO EBSD input at all (useful in CI and
            before the preprocessing has been done)
              python3 make_test_micro.py synth --Lx 400 --Ly 200 --out micro_test.npz

The synthetic patch imitates the qualitative WAAM situation: fine equiaxed
grains on the left (BCC, "17-4PH"), coarse columnar grains on the right (FCC,
"316L"), a transition band in between - but it is NOT measured data and must
never be used for a reported result. `meta.synthetic` is set to true so the
run metadata carries that flag.
"""
import argparse
import json
import os

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(here, p)


def do_crop(a):
    d = np.load(_abs(a.src))
    meta = json.loads(str(d['meta']))
    step = float(meta['step_um'])
    x_um = d['x_um'] if 'x_um' in d else (np.arange(d['phase'].shape[1]) + 0.5) * step
    y_um = d['y_um'] if 'y_um' in d else (np.arange(d['phase'].shape[0]) + 0.5) * step
    x0, y0, x1, y1 = a.window
    # window is given RELATIVE to the ROI (0,0 = top-left cell of the npz)
    xr = x_um - x_um[0] + 0.5 * step
    yr = y_um - y_um[0] + 0.5 * step
    ci = np.where((xr >= x0) & (xr < x1))[0]
    cj = np.where((yr >= y0) & (yr < y1))[0]
    if ci.size == 0 or cj.size == 0:
        raise SystemExit(f'empty window {a.window} (ROI is '
                         f'{xr[-1]:.0f} x {yr[-1]:.0f} um)')
    sl = np.ix_(cj, ci)
    meta['cropped_from'] = os.path.basename(a.src)
    meta['crop_window_um'] = list(a.window)
    out = dict(euler_deg=d['euler_deg'][cj][:, ci], phase=d['phase'][sl],
               grain_id=d['grain_id'][sl], zone=d['zone'][sl],
               x_um=x_um[ci], y_um=y_um[cj], meta=json.dumps(meta))
    return out, step


def do_synth(a):
    step = a.step
    nx = max(int(round(a.Lx / step)), 4)
    ny = max(int(round(a.Ly / step)), 4)
    rng = np.random.default_rng(a.seed)

    x = (np.arange(nx) + 0.5) * step
    y = (np.arange(ny) + 0.5) * step
    X_, Y_ = np.meshgrid(x, y)

    # three regions along x: fine | transition | coarse columnar
    b0, b1 = a.Lx / 3.0, 2.0 * a.Lx / 3.0
    zone = np.where(X_ < b0, 0, np.where(X_ < b1, 1, 2)).astype(np.int8)

    # grain size grows from left to right
    gsize = np.where(X_ < b0, a.grain_fine,
                     np.where(X_ < b1, 2.0 * a.grain_fine, a.grain_coarse))
    gx = np.floor(X_ / gsize).astype(np.int64)
    gy = np.floor(Y_ / np.where(zone == 2, a.grain_coarse * 3.0, gsize)).astype(np.int64)
    key = gx * 100003 + gy * 7919 + zone.astype(np.int64) * 104729
    uniq, gid = np.unique(key, return_inverse=True)
    gid = (gid + 1).astype(np.int32).reshape(ny, nx)

    n_g = len(uniq)
    eul_g = rng.uniform(0.0, 360.0, (n_g, 3))
    eul_g[:, 1] = np.rad2deg(np.arccos(rng.uniform(-1.0, 1.0, n_g)))   # uniform Phi
    euler = eul_g[gid - 1]
    phase_g = np.where(rng.random(n_g) < 0.5, 1, 2).astype(np.int8)
    phase = phase_g[gid - 1]
    # left region mostly BCC, right region FCC (as in the real material)
    phase = np.where(zone == 0, np.where(rng.random((ny, nx)) < 0.7, 2, phase), phase)
    phase = np.where(zone == 2, 1, phase).astype(np.int8)

    meta = dict(step_um=step, synthetic=True,
                roi_um=[0.0, 0.0, nx * step, ny * step],
                zones=[b0, b1], seed=a.seed,
                frame='map (x right, y DOWN); row 0 = top',
                euler='Bunge deg, TSL map frame', phase='1=fcc 2=bcc',
                zone='0=fine/BCC 1=transition 2=coarse columnar/FCC',
                note='SYNTHETIC smoke-test microstructure - not measured data')
    out = dict(euler_deg=euler, phase=phase, grain_id=gid, zone=zone,
               x_um=x, y_um=y, meta=json.dumps(meta))
    return out, step


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = ap.add_subparsers(dest='mode', required=True)

    c = sub.add_parser('crop')
    c.add_argument('--src', required=True)
    c.add_argument('--window', nargs=4, type=float, required=True,
                   metavar=('X0', 'Y0', 'X1', 'Y1'),
                   help='window in um, relative to the ROI origin')
    c.add_argument('--out', default='micro_test.npz')

    s = sub.add_parser('synth')
    s.add_argument('--Lx', type=float, default=400.0)
    s.add_argument('--Ly', type=float, default=200.0)
    s.add_argument('--step', type=float, default=3.371)
    s.add_argument('--grain-fine', type=float, default=12.0)
    s.add_argument('--grain-coarse', type=float, default=60.0)
    s.add_argument('--seed', type=int, default=42)
    s.add_argument('--out', default='micro_test.npz')

    a = ap.parse_args()
    out, step = (do_crop(a) if a.mode == 'crop' else do_synth(a))
    path = _abs(a.out)
    np.savez_compressed(path, **out)
    ny, nx = out['phase'].shape
    ng = len(np.unique(out['grain_id']))
    print(f'wrote {path}: {ny} x {nx} Zellen = {ny*nx}, '
          f'{nx*step:.0f} x {ny*step:.0f} um, {ng} Koerner, step {step:.3f} um')


if __name__ == '__main__':
    main()
