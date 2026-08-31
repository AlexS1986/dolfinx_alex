#!/usr/bin/env python3
"""
Höhenstudie: mehrere übereinanderliegende Auswerte-Streifen (Bänder) und die
volle Probenhöhe, alle aus DEMSELBEN EBSD-Scan.

Der Referenzfall `roi` (Bericht, Kapitel 2) ist ein einzelnes horizontales Band
in der Kartenmitte. Diese Studie legt weitere Bänder gleicher Höhe darüber und
darunter und zusätzlich ein Fenster über die **volle Kartenhöhe** (horizontal
wie der Referenzfall beschnitten, das weit rechts liegende homogene 316L wird
nur teilweise erfasst).

Der teure Schritt der Vorverarbeitung ist die Pixel->Korn-Zuordnung, und die
läuft ohnehin über die GANZE Karte (preprocess_ebsd_to_grid.assign_pixels).
Sie wird hier einmal berechnet, in `_fullmap_assign.npz` zwischengespeichert
und dann für alle Fälle nur noch ausgeschnitten.

Zonengrenzen: für den Referenzfall stammen sie aus der eingezeichneten
Markierung (666,6 / 1494,5 µm). Für die anderen Bänder gibt es keine
Markierung; sie werden daher aus der Mikrostruktur selbst bestimmt
(spaltenweiser BCC-Anteil und Korngröße, s. `zone_boundaries`) — dieselbe
Vorschrift wird zur Kontrolle auch auf den Referenzfall angewandt.

Aufruf:
    python3 study_rois.py --txt ... --bmp ...            # alle Fälle schreiben
    python3 study_rois.py --txt ... --bmp ... --only full
"""
import argparse, json, os

import numpy as np
import pandas as pd
from PIL import Image

from preprocess_ebsd_to_grid import (S_UM_PER_PX, load_grains, assign_pixels)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '_fullmap_assign.npz')

# Referenz (Bericht Kapitel 2): x 30..2811, y 1339..2225, Zonen 666.6 / 1494.5
X0, X1 = 30.0, 2811.0
BAND_H = 886.0                       # Höhe des Referenzbandes [µm]
REF_ZONES = (666.6, 1494.5)


def cases(H_um):
    """Fallliste: 4 gestapelte Bänder + volle Höhe + Referenzband."""
    out = [dict(tag='roi', y0=1339.0, y1=2225.0, zones=REF_ZONES,
                label='Referenzband (Bericht Kap. 2)')]
    for k in range(4):
        y0 = k * BAND_H
        out.append(dict(tag=f'band{k+1}', y0=y0, y1=min(y0 + BAND_H, H_um),
                        zones=None, label=f'Band {k+1} (y = {y0:.0f}…{y0+BAND_H:.0f} µm)'))
    out.append(dict(tag='full', y0=0.0, y1=H_um, zones=None,
                    label='volle Kartenhöhe'))
    return out


def fullmap_assignment(txt, bmp, rebuild=False):
    """(gi, ipf, grains, im.shape) — Pixel->Kornindex für die ganze Karte."""
    g = load_grains(txt)
    im = np.asarray(Image.open(bmp)).astype(np.int16)
    if os.path.isfile(CACHE) and not rebuild:
        d = np.load(CACHE)
        if d['shape'][0] == im.shape[0] and d['shape'][1] == im.shape[1]:
            print(f'Zuordnung aus Cache: {CACHE}')
            return d['gi'], d['ipf'], g, im
    print(f'{len(g)} Körner, Karte {im.shape[1]}x{im.shape[0]} px — '
          f'Pixelzuordnung läuft (einmalig)…')
    gi, ipf = assign_pixels(g, im)
    rec = ipf[gi.ravel()].reshape(im.shape)
    msk = ~(((im > 245).all(axis=2)) | ((im < 10).all(axis=2)))
    err = float(np.abs(rec - im)[msk].mean())
    print(f'mittlerer Farbfehler (ohne Beschriftung): {err:.1f} von 255')
    np.savez_compressed(CACHE, gi=gi.astype(np.int32), ipf=ipf,
                        shape=np.array(im.shape[:2]), dRGB=err)
    return gi, ipf, g, im


D_COARSE = 500.0        # ab diesem Äquivalenzdurchmesser gilt ein Korn als grob
WIN = 21                # Glättungsfenster [Spalten] ~ 71 µm


def zone_boundaries(phase, deq, xc):
    """Zonengrenzen aus der Mikrostruktur selbst — zwei unabhängige Signale.

    Links (17-4PH) ist BCC-reich und feinkörnig, rechts (316L) BCC-frei und
    extrem grobkörnig, dazwischen der Übergang.

      Grenze 0|1: der BCC-Anteil je Spalte fällt auf die **Hälfte** seines
                  linken Plateauwerts (Halbwertsstelle des Phasenübergangs).
      Grenze 1|2: der Flächenanteil **grober** Körner (d_eq > 500 µm) je Spalte
                  übersteigt 80 % — der Beginn des kolumnaren 316L-Gefüges.

    Beide Schwellen sind am Referenzband kalibriert, für das die Grenzen aus
    der eingezeichneten Markierung bekannt sind (666,6 / 1494,5 µm).

    Rückgabe: (x01, x12, f_bcc_glatt, f_grob_glatt)
    """
    k = np.ones(WIN) / WIN
    fb = np.convolve((phase == 2).mean(axis=0), k, mode='same')
    fc = np.convolve((deq > D_COARSE).mean(axis=0), k, mode='same')

    i_pl = int(np.argmax(fb))
    thr = 0.5 * fb[i_pl]
    below = np.nonzero((fb < thr) & (np.arange(len(fb)) > i_pl))[0]
    x01 = float(xc[below[0]]) if len(below) else float(xc[0])

    coarse = np.nonzero((fc > 0.8) & (xc > x01))[0]
    if len(coarse) == 0:                       # Rückfall: BCC praktisch weg
        gone = np.nonzero((fb < 0.02 * fb[i_pl]) & (xc > x01))[0]
        x12 = float(xc[gone[0]]) if len(gone) else float(xc[-1])
    else:
        x12 = float(xc[coarse[0]])
    return x01, x12, fb, fc


def write_case(c, gi, ipf, g, im, step, outdir, zones_override=None):
    H, W, _ = im.shape
    xc = np.arange(X0 + step / 2, X1, step)
    yc = np.arange(c['y0'] + step / 2, c['y1'], step)
    px = np.clip((xc / S_UM_PER_PX).astype(int), 0, W - 1)
    py = np.clip((yc / S_UM_PER_PX).astype(int), 0, H - 1)
    sub = gi[np.ix_(py, px)]

    euler = np.stack([g.phi1_deg.values[sub], g.PHI_deg.values[sub],
                      g.phi2_deg.values[sub]], axis=-1)
    phase = g.phase.values[sub].astype(np.int8)
    gid = g.grain_id.values[sub].astype(np.int32)

    area = g.area_um2.values[sub]
    deq = np.sqrt(4.0 * area / np.pi)
    x01, x12, fb, fc = zone_boundaries(phase, deq, xc)
    zones = zones_override or c['zones']
    if zones is None:
        zones, src = (x01, x12), 'aus der Mikrostruktur bestimmt'
    else:
        src = 'aus der Markierung übernommen'
    zone = (np.where(xc[None, :] < zones[0], 0,
                     np.where(xc[None, :] < zones[1], 1, 2)).astype(np.int8)
            * np.ones((len(yc), 1), np.int8))

    meta = dict(source_txt='WAAM_N=1_A12D_Uebergangsbereich.txt',
                source_bmp='WAAM_N=1_A12D_Uebergangsbereich.bmp',
                roi_um=[X0, c['y0'], X1, c['y1']], step_um=step,
                um_per_px=S_UM_PER_PX, zones=list(zones), zone_source=src,
                zones_detected=[x01, x12], case_label=c['label'],
                frame='map (x right, y DOWN, z into plane); row 0 = top',
                euler='Bunge deg, TSL map frame', phase='1=fcc 2=bcc',
                zone='0=17-4PH 1=transition 2=316L')
    out = os.path.join(outdir, f"micro_{c['tag']}.npz")
    np.savez_compressed(out, euler_deg=euler, phase=phase, grain_id=gid,
                        zone=zone, x_um=xc, y_um=yc, f_bcc=fb, f_coarse=fc,
                        meta=json.dumps(meta))
    prev = ipf[sub.ravel()].reshape(len(yc), len(xc), 3).astype(np.uint8)
    Image.fromarray(prev).save(os.path.join(outdir, f"micro_{c['tag']}_ipf.png"))
    print(f"  {c['tag']:6s} {len(yc):4d} x {len(xc)} Zellen  "
          f"y = {c['y0']:6.0f}…{c['y1']:6.0f} µm  Zonen {zones[0]:7.1f} / "
          f"{zones[1]:7.1f} µm  ({src})")
    return dict(tag=c['tag'], label=c['label'], y0=c['y0'], y1=c['y1'],
                nx=len(xc), ny=len(yc), zones=list(zones), zone_source=src,
                zones_detected=[x01, x12],
                n_grains=int(len(np.unique(gid))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--txt', required=True)
    ap.add_argument('--bmp', required=True)
    ap.add_argument('--step', type=float, default=S_UM_PER_PX)
    ap.add_argument('--outdir', default=HERE)
    ap.add_argument('--only', default=None, help='nur diesen Fall schreiben')
    ap.add_argument('--rebuild', action='store_true', help='Cache neu berechnen')
    args = ap.parse_args()

    gi, ipf, g, im = fullmap_assignment(args.txt, args.bmp, args.rebuild)
    H_um = im.shape[0] * S_UM_PER_PX
    cs = cases(H_um)
    if args.only:
        cs = [c for c in cs if c['tag'] == args.only]
    print(f'Karte {im.shape[1]}x{im.shape[0]} px = {im.shape[1]*S_UM_PER_PX:.0f} x '
          f'{H_um:.0f} µm; Fälle:')
    summary = [write_case(c, gi, ipf, g, im, args.step, args.outdir) for c in cs]
    with open(os.path.join(args.outdir, 'study_cases.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    print('wrote study_cases.json')


if __name__ == '__main__':
    main()
