#!/usr/bin/env python3
"""
EBSD (TSL grain export + IPF-Z BMP) -> regular-grid microstructure field for
the 2D plane-stress FE model of the WAAM N=1 transition region.

The grain export (WAAM_N=1_A12D_Uebergangsbereich.txt) is per-GRAIN (centroid,
mean Bunge angles, ellipse fit, phase); the BMP is the IPF-Z rendering of the
same map (3.371 um/px). Per-pixel grain shapes are reconstructed by assigning
every pixel to the grain minimising

    cost = max(mahalanobis_ellipse - 1, 0) + LAM * |RGB_bmp - RGB_ipf(grain)|/85

over k-nearest-centroid candidates plus the N_BIG largest grains (their
centroids can be far from their own pixels - chevron growth shapes).
RGB_ipf is computed from the grain's mean orientation (cubic IPF-Z, sqrt
gamma), which reproduces the BMP rendering (validated: mean |dRGB| ~ 17).

Output <tag>.npz:
    euler_deg (ny,nx,3) Bunge angles [deg], TSL MAP frame (y down, z into plane)
    phase     (ny,nx)   1=FCC, 2=BCC
    grain_id  (ny,nx)   original grain id from the export
    zone      (ny,nx)   0=17-4PH side, 1=transition, 2=316L side (from --zones)
    x_um,y_um (nx),(ny) cell-center coordinates in map frame (y down!)
    meta      json string (roi, step, zones, provenance)
Row 0 = TOP of the map. The FE reader flips rows to a y-up frame and applies
the 180deg-about-x orientation flip (see plane_stress_crystal.FLIP_X180).

Run anywhere with numpy/pandas/scipy/PIL (no dolfinx needed).
"""
import argparse, json, os
import numpy as np, pandas as pd
from PIL import Image
from scipy.spatial import cKDTree

from plane_stress_crystal import bunge_to_g

S_UM_PER_PX = 3.371          # map scale (4695x3699 um on 1393x1097 px)

COLUMNS = ['grain_id','phase','phi1_deg','PHI_deg','phi2_deg','phi1_rad','PHI_rad','phi2_rad',
 'h','k','l','u','v','w','h_f','k_f','l_f','u_f','v_f','w_f','x_um','y_um','IQ','CI','fit_deg',
 'video_signal','R','G','B','edge_grain','n_points','area_um2','diameter_um','ASTM','aspect_ratio',
 'major_um','minor_um','ellipse_deg','ellipticity','circularity','feret_max','feret_min','ori_spread','neigh_misori']


def cubic_ops():
    import itertools
    ops = []
    for p in itertools.permutations(range(3)):
        P = np.zeros((3, 3)); P[[0, 1, 2], list(p)] = 1
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    M = P * np.array([sx, sy, sz])[:, None]
                    if abs(np.linalg.det(M) - 1) < 1e-9:
                        ops.append(M)
    return np.array(ops)


def ipf_z_colors(phi1, Phi, phi2):
    """IPF-Z colors (TSL-like, sqrt gamma) for arrays of Bunge angles [rad]."""
    OPS = cubic_ops()
    n = len(phi1)
    g = np.empty((n, 3, 3))
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c, sn = np.cos(Phi), np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    g[:, 0, 0] = c1*c2 - s1*s2*c; g[:, 0, 1] = s1*c2 + c1*s2*c; g[:, 0, 2] = s2*sn
    g[:, 1, 0] = -c1*s2 - s1*c2*c; g[:, 1, 1] = -s1*s2 + c1*c2*c; g[:, 1, 2] = c2*sn
    g[:, 2, 0] = s1*sn; g[:, 2, 1] = -c1*sn; g[:, 2, 2] = c
    d = g[:, :, 2]                                   # crystal direction of sample z
    v = np.abs(np.einsum('oij,nj->noi', OPS, d))
    vs = np.sort(v, axis=2)[:, 0, :]                 # (y<=x<=z), same for all ops
    y, x, z = vs[:, 0], vs[:, 1], vs[:, 2]
    R, G, B = z - x, x - y, y
    m = np.maximum.reduce([R, G, B, np.full_like(R, 1e-12)])
    rgb = np.stack([R, G, B], axis=1) / m[:, None]
    return np.sqrt(np.clip(rgb, 0, 1)) * 255


def load_grains(txt):
    df = pd.read_csv(txt, comment='#', sep=r'\s+', header=None, names=COLUMNS)
    return df[(df.grain_id > 0) & (df.phase > 0)].reset_index(drop=True)


def assign_pixels(g, im, k=30, n_big=120, lam=3.0, chunk=150000):
    """Per-pixel grain assignment (returns index into g's rows)."""
    H, W, _ = im.shape
    ipf = ipf_z_colors(g.phi1_rad.values, g.PHI_rad.values, g.phi2_rad.values)
    s = S_UM_PER_PX
    cx, cy = g.x_um.values / s, g.y_um.values / s
    a = np.maximum(g.major_um.values / s, 1.0)
    b = np.maximum(g.minor_um.values / s, 0.7)
    th = np.deg2rad(g.ellipse_deg.values)
    ct, st = np.cos(-th), np.sin(-th)          # y-down image: angle sign flips
    tree = cKDTree(np.stack([cx, cy], axis=1))
    big = np.argsort(-g.area_um2.values)[:n_big]
    ys, xs = np.mgrid[0:H, 0:W]
    P = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)
    pix = im.reshape(-1, 3).astype(float)
    out = np.empty(len(P), dtype=np.int64)
    for i0 in range(0, len(P), chunk):
        p, c = P[i0:i0+chunk], pix[i0:i0+chunk]
        _, jj = tree.query(p, k=k, workers=-1)
        jj = np.concatenate([jj, np.broadcast_to(big, (len(p), len(big)))], axis=1)
        dx, dy = p[:, 0, None] - cx[jj], p[:, 1, None] - cy[jj]
        u, v = dx*ct[jj] - dy*st[jj], dx*st[jj] + dy*ct[jj]
        pen = np.maximum(np.hypot(u / a[jj], v / b[jj]) - 1.0, 0.0)
        col = np.abs(ipf[jj] - c[:, None, :]).mean(axis=2)
        out[i0:i0+chunk] = jj[np.arange(len(p)), np.argmin(pen + lam*col/85.0, axis=1)]
    return out.reshape(H, W), ipf


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--txt', required=True)
    ap.add_argument('--bmp', required=True)
    ap.add_argument('--roi', nargs=4, type=float, metavar=('X0','Y0','X1','Y1'),
                    default=None, help='ROI in um, MAP frame (y down, origin top-left)')
    ap.add_argument('--step', type=float, default=S_UM_PER_PX,
                    help='grid cell size in um (default = 1 px = 3.371 um)')
    ap.add_argument('--zones', nargs=2, type=float, metavar=('X_PH_END','X_TRANS_END'),
                    default=None, help='zone boundaries in um along x: '
                    '17-4PH | transition | 316L (default: no zoning -> all 1)')
    ap.add_argument('--tag', default='transition2d')
    ap.add_argument('--outdir', default='.')
    args = ap.parse_args()

    g = load_grains(args.txt)
    im = np.asarray(Image.open(args.bmp)).astype(np.int16)
    H, W, _ = im.shape
    print(f'{len(g)} grains, map {W}x{H} px = {W*S_UM_PER_PX:.0f}x{H*S_UM_PER_PX:.0f} um')

    gi, ipf = assign_pixels(g, im)
    rec = ipf[gi.ravel()].reshape(im.shape)
    msk = ~(((im > 245).all(axis=2)) | ((im < 10).all(axis=2)))
    print(f'reconstruction mean|dRGB| (excl. annotation): '
          f'{np.abs(rec - im)[msk].mean():.1f}')

    x0, y0, x1, y1 = args.roi if args.roi else (0, 0, W*S_UM_PER_PX, H*S_UM_PER_PX)
    xc = np.arange(x0 + args.step/2, x1, args.step)
    yc = np.arange(y0 + args.step/2, y1, args.step)
    px = np.clip((xc / S_UM_PER_PX).astype(int), 0, W-1)
    py = np.clip((yc / S_UM_PER_PX).astype(int), 0, H-1)
    sub = gi[np.ix_(py, px)]                       # (ny, nx) grain row-index

    euler = np.stack([g.phi1_deg.values[sub], g.PHI_deg.values[sub],
                      g.phi2_deg.values[sub]], axis=-1)
    phase = g.phase.values[sub].astype(np.int8)
    gid = g.grain_id.values[sub].astype(np.int32)
    zone = np.ones_like(phase)
    if args.zones:
        zone = np.where(xc[None, :] < args.zones[0], 0,
                        np.where(xc[None, :] < args.zones[1], 1, 2)
                        ).astype(np.int8) * np.ones((len(yc), 1), np.int8)

    meta = dict(source_txt=os.path.basename(args.txt), source_bmp=os.path.basename(args.bmp),
                roi_um=[x0, y0, x1, y1], step_um=args.step, um_per_px=S_UM_PER_PX,
                zones=args.zones, frame='map (x right, y DOWN, z into plane); row 0 = top',
                euler='Bunge deg, TSL map frame', phase='1=fcc 2=bcc',
                zone='0=17-4PH 1=transition 2=316L')
    out = os.path.join(args.outdir, f'micro_{args.tag}.npz')
    np.savez_compressed(out, euler_deg=euler, phase=phase, grain_id=gid,
                        zone=zone, x_um=xc, y_um=yc, meta=json.dumps(meta))
    print(f'wrote {out}  grid {len(yc)}x{len(xc)} cells '
          f'({len(yc)*len(xc)} cells, step {args.step} um)')

    # preview: reconstructed IPF in ROI + phase map
    prev = ipf[sub.ravel()].reshape(len(yc), len(xc), 3).astype(np.uint8)
    Image.fromarray(prev).save(os.path.join(args.outdir, f'micro_{args.tag}_ipf.png'))
    ph = np.zeros((len(yc), len(xc), 3), np.uint8)
    ph[phase == 1] = (220, 60, 60); ph[phase == 2] = (60, 90, 220)
    Image.fromarray(ph).save(os.path.join(args.outdir, f'micro_{args.tag}_phase.png'))
    print('previews written')


if __name__ == '__main__':
    main()
