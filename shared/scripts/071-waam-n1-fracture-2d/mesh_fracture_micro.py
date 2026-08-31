#!/usr/bin/env python3
"""
Mesh generator for 071: an explicitly modelled EBSD microstructure PATCH
embedded in homogeneous isotropic steel, with a straight crack line and a
refined corridor along it.

Layout (FE frame, y up, all lengths in um; the crack line is y = 0):

    y = +H/2  +---------------------------------------------------+
              |                 embedding (isotropic)             |
    y = +Ly/2 |      +-------------------------------------+      |
              |      |        microstructure patch         |      |
    y = 0   ----------------- crack line (embedded) ----------------
              |      |                                     |      |
    y = -Ly/2 |      +-------------------------------------+      |
              |                                                   |
    y = -H/2  +---------------------------------------------------+
           x = -mx   x = 0                            x = Lx    x = Lx+mx

The patch rectangle is a real geometric entity, so cells never straddle the
patch boundary. Element size: `--e-fine` in the corridor |y| <= `--corridor`
inside the patch, `--e-patch` in the rest of the patch, growing to `--e-far`
in the embedding.

The stiffness itself is NOT stored in the mesh - `run_fracture_simulation.py`
assigns it per cell from the microstructure npz via a centroid lookup
(`materials_fracture_2d.Microstructure.sample`). The cell tags written here
are bookkeeping only:

    0 = embedding (homogeneous isotropic steel)
    1 = microstructure patch

Output: <name>.msh, <name>.xdmf/.h5 (meshio, cell_data "name_to_read").
"""
import argparse
import json
import os

import gmsh
import meshio
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
ap.add_argument('--micro', default=None,
                help='micro_<tag>.npz - patch size is taken from it')
ap.add_argument('--Lx', type=float, default=None, help='patch width [um] (overrides --micro)')
ap.add_argument('--Ly', type=float, default=None, help='patch height [um] (overrides --micro)')
ap.add_argument('--rotated', action='store_true',
                help='the patch will be used rotated by 90 deg -> swap Lx/Ly')
ap.add_argument('--margin-x', type=float, default=None,
                help='embedding margin left/right [um] (absolute; overrides the fraction)')
ap.add_argument('--margin-y', type=float, default=None,
                help='embedding margin top/bottom [um] (absolute; overrides the fraction)')
ap.add_argument('--margin-x-frac', type=float, default=0.5,
                help='embedding margin left/right as a fraction of Lx (default 0.5)')
ap.add_argument('--margin-y-frac', type=float, default=1.0,
                help='embedding margin top/bottom as a fraction of Ly (default 1.0)')
ap.add_argument('--corridor', type=float, default=None,
                help='half-width of the refined crack corridor [um] '
                     '(default 10*epsilon if --epsilon given, else 0.15*Ly)')
ap.add_argument('--epsilon', type=float, default=None,
                help='phase-field length [um]; only used for defaults + report')
ap.add_argument('--e-fine', type=float, default=None,
                help='element size in the corridor [um] (default epsilon/4)')
ap.add_argument('--e-patch', type=float, default=None,
                help='element size in the rest of the patch [um] (default 2.5*e_fine)')
ap.add_argument('--e-far', type=float, default=None,
                help='element size in the embedding [um] (default 15*e_fine)')
ap.add_argument('--name', default='mesh_fracture_micro')
ap.add_argument('--outdir', default=None)
args = ap.parse_args()

outdir = args.outdir or here
os.makedirs(outdir, exist_ok=True)

# ---------------------------------------------------------------- geometry --
Lx, Ly = args.Lx, args.Ly
step = None
if args.micro is not None and (Lx is None or Ly is None):
    p = args.micro if os.path.isabs(args.micro) else os.path.join(here, args.micro)
    d = np.load(p)
    ny, nx = d['phase'].shape
    step = float(json.loads(str(d['meta']))['step_um'])
    Lx, Ly = nx * step, ny * step
    if args.rotated:
        Lx, Ly = Ly, Lx
if Lx is None or Ly is None:
    raise SystemExit('need --micro or both --Lx/--Ly')

mx = args.margin_x if args.margin_x is not None else args.margin_x_frac * Lx
my = args.margin_y if args.margin_y is not None else args.margin_y_frac * Ly

eps = args.epsilon
e_fine = args.e_fine if args.e_fine is not None else (eps / 4.0 if eps else Ly / 200.0)
e_patch = args.e_patch if args.e_patch is not None else 2.5 * e_fine
e_far = args.e_far if args.e_far is not None else 15.0 * e_fine
corridor = args.corridor if args.corridor is not None else (10.0 * eps if eps else 0.15 * Ly)
corridor = min(corridor, 0.49 * Ly)

x0, x1 = -mx, Lx + mx
y0, y1 = -0.5 * Ly - my, 0.5 * Ly + my
px0, px1 = 0.0, Lx
py0, py1 = -0.5 * Ly, 0.5 * Ly

print(f'Patch  {Lx:.1f} x {Ly:.1f} um' + (f' (Zelle {step:.3f} um)' if step else ''))
print(f'Gebiet [{x0:.1f}, {x1:.1f}] x [{y0:.1f}, {y1:.1f}] um')
print(f'Netz   e_fine={e_fine:.3f}  e_patch={e_patch:.3f}  e_far={e_far:.3f}  '
      f'Korridor +-{corridor:.1f} um' + (f'  (epsilon={eps:.3f} -> eps/h={eps/e_fine:.1f})' if eps else ''))

gmsh.initialize()
gmsh.option.setNumber('General.Terminal', 1)
gmsh.model.add('micro_fracture')
occ = gmsh.model.occ

outer = occ.addRectangle(x0, y0, 0.0, x1 - x0, y1 - y0)
patch = occ.addRectangle(px0, py0, 0.0, px1 - px0, py1 - py0)
frag, _ = occ.fragment([(2, outer)], [(2, patch)])
occ.synchronize()

# the crack line: embedded so that mesh facets lie exactly on y = 0
pA = occ.addPoint(x0, 0.0, 0.0)
pB = occ.addPoint(x1, 0.0, 0.0)
crack_line = occ.addLine(pA, pB)
occ.synchronize()
surfaces = [t for t in gmsh.model.getEntities(2)]
# split the crack line at the surface boundaries so it can be embedded
frag2, _ = occ.fragment(surfaces, [(1, crack_line)])
occ.synchronize()

surfaces = [tag for dim, tag in gmsh.model.getEntities(2)]
lines_on_crack = [tag for dim, tag in gmsh.model.getEntities(1)
                  if abs(gmsh.model.occ.getCenterOfMass(1, tag)[1]) < 1e-9
                  and abs(gmsh.model.occ.getCenterOfMass(1, tag)[2]) < 1e-9]

# ------------------------------------------------------------- size fields --
f_corr = gmsh.model.mesh.field.add('Box')
gmsh.model.mesh.field.setNumber(f_corr, 'VIn', e_fine)
gmsh.model.mesh.field.setNumber(f_corr, 'VOut', e_far)
# the corridor starts at the LEFT DOMAIN EDGE so that the initial crack and
# its phase-field profile are resolved just as well as the propagating tip
gmsh.model.mesh.field.setNumber(f_corr, 'XMin', x0)
gmsh.model.mesh.field.setNumber(f_corr, 'XMax', px1 + 0.25 * mx)
gmsh.model.mesh.field.setNumber(f_corr, 'YMin', -corridor)
gmsh.model.mesh.field.setNumber(f_corr, 'YMax', corridor)
gmsh.model.mesh.field.setNumber(f_corr, 'Thickness', 2.0 * corridor)

f_patch = gmsh.model.mesh.field.add('Box')
gmsh.model.mesh.field.setNumber(f_patch, 'VIn', e_patch)
gmsh.model.mesh.field.setNumber(f_patch, 'VOut', e_far)
gmsh.model.mesh.field.setNumber(f_patch, 'XMin', px0)
gmsh.model.mesh.field.setNumber(f_patch, 'XMax', px1)
gmsh.model.mesh.field.setNumber(f_patch, 'YMin', py0)
gmsh.model.mesh.field.setNumber(f_patch, 'YMax', py1)
gmsh.model.mesh.field.setNumber(f_patch, 'Thickness', max(mx, my) * 0.5)

f_min = gmsh.model.mesh.field.add('Min')
gmsh.model.mesh.field.setNumbers(f_min, 'FieldsList', [f_corr, f_patch])
gmsh.model.mesh.field.setAsBackgroundMesh(f_min)
gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)

gmsh.model.addPhysicalGroup(2, surfaces, 1)
gmsh.model.setPhysicalName(2, 1, 'domain')
if lines_on_crack:
    gmsh.model.addPhysicalGroup(1, lines_on_crack, 2)
    gmsh.model.setPhysicalName(1, 2, 'crack_line')

gmsh.model.mesh.generate(2)
msh_path = os.path.join(outdir, args.name + '.msh')
gmsh.write(msh_path)
gmsh.finalize()

# --------------------------------------------------------- meshio -> xdmf ---
mesh = meshio.read(msh_path)
nodes = mesh.points[:, :2]
elems = mesh.get_cells_type('triangle')

referenced = sorted({int(p) for tri in elems for p in tri})
remap = {old: new for new, old in enumerate(referenced)}
nodes = np.asarray([nodes[i] for i in referenced])
elems = np.asarray([[remap[p] for p in tri] for tri in elems], dtype=np.int64)

centroids = nodes[elems].mean(axis=1)
tags = np.zeros(len(elems), dtype=np.int32)
in_patch = ((centroids[:, 0] >= px0) & (centroids[:, 0] <= px1)
            & (centroids[:, 1] >= py0) & (centroids[:, 1] <= py1))
tags[in_patch] = 1

xdmf_path = os.path.join(outdir, args.name + '.xdmf')
meshio.write(xdmf_path,
             meshio.Mesh(points=nodes, cells={'triangle': elems},
                         cell_data={'name_to_read': [tags]}))

meta = dict(Lx_um=Lx, Ly_um=Ly, margin_x_um=mx, margin_y_um=my,
            domain_um=[x0, y0, x1, y1], patch_um=[px0, py0, px1, py1],
            e_fine=e_fine, e_patch=e_patch, e_far=e_far, corridor_um=corridor,
            epsilon_um=eps, micro=args.micro, rotated=bool(args.rotated),
            n_nodes=int(len(nodes)), n_cells=int(len(elems)),
            cell_tag={'0': 'embedding (isotropic)', '1': 'microstructure patch'})
with open(os.path.join(outdir, args.name + '_meta.json'), 'w') as fh:
    json.dump(meta, fh, indent=2)

print(f'{len(nodes)} Knoten, {len(elems)} Dreiecke '
      f'({in_patch.sum()} im Patch, {(~in_patch).sum()} Einbettung)')
print(f'  -> {3 * len(nodes)} DOF im gemischten (u, s)-Raum bei P1')
print(f'geschrieben: {xdmf_path}')
