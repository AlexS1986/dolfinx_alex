# WAAM polycrystal — anisotropic elastic homogenization & directional E-modulus

Elastic FE computations on the Neper-generated WAAM microstructures (316L, 17-4PH),
with a **per-grain anisotropic stiffness**: each grain gets the cubic single-crystal
stiffness of its phase (FCC austenite / BCC martensite), rotated into the sample
frame by the grain's measured Bunge orientation.

Two computations:

1. **`homogenize_rve.py`** — KUBC homogenization of a cube RVE → full effective
   6×6 stiffness `Chom` (per material). Same solver/BC structure as
   `Meshing/pygalmesh/.../009-Binning-Variation-CT-Stiffness/00_template/linearelastic.py`,
   but anisotropic per grain.
2. **`uniaxial_tension.py`** — numerical uniaxial tensile test on a directional
   bar (V/H/45°) → apparent Young's modulus in the loading direction.

Both run in the **dolfinx v0.7.3 container** and reuse the shared `alex` utils.
Single-crystal constants live in `config.json` (GPa, editable).

## Inputs (from the Neper pipeline)

Generated in `Meshing/Neper/data/04_anisotropy_waam/neper_pipeline` (Neper/Gmsh
container):

```bash
# RVE cubes for homogenization (300 grains, rcl 0.5):
MAT=316L   bash 09_homogenization_rve.sh
MAT=17-4PH bash 09_homogenization_rve.sh
#   -> waam_<MAT>_n300.xdmf/.h5  +  grain_ori_<MAT>.txt

# directional bars for the uniaxial test (optional):
MAT=316L   python3 07_tensile_specimens.py
MAT=17-4PH python3 07_tensile_specimens.py
#   -> spec_<MAT>_V/H/45deg.xdmf/.h5  +  grain_ori_<MAT>_<orient>.txt
```

The mesh XDMF carries a single `grain` cell tag (grain id); `grain_ori_*.txt`
maps `grain_id -> phi1 Phi phi2 crystal` (Bunge deg).

## Workflow

**1. Stage inputs (host, both folders visible):**

```bash
python3 prepare_inputs.py --rve 316L 17-4PH --n 300           # RVE cubes
python3 prepare_inputs.py --specimens 316L 17-4PH            # directional bars
# -> copies meshes (.xdmf + .h5) and grain_ori into ./inputs/
```

**2. Edit `config.json`** — set the single-crystal cubic constants C11/C12/C44
per phase (`fcc` = 316L austenite, `bcc` = 17-4PH martensite). Defaults are
literature placeholders.

**3. Run in the dolfinx container:**

```bash
# effective stiffness tensor per material
python3 homogenize_rve.py --mesh inputs/waam_316L_n300.xdmf  --ori inputs/grain_ori_316L.txt  --tag 316L
python3 homogenize_rve.py --mesh inputs/waam_17-4PH_n300.xdmf --ori inputs/grain_ori_17-4PH.txt --tag 17-4PH
#   -> Chom_<tag>.json (6x6 GPa) + directional E + iso-equivalent summary

# directional Young's modulus (per orientation)
python3 uniaxial_tension.py --mesh inputs/spec_316L_V.xdmf     --ori inputs/grain_ori_316L_V.txt     --tag 316L_V
python3 uniaxial_tension.py --mesh inputs/spec_316L_H.xdmf     --ori inputs/grain_ori_316L_H.txt     --tag 316L_H
python3 uniaxial_tension.py --mesh inputs/spec_316L_45deg.xdmf --ori inputs/grain_ori_316L_45deg.txt --tag 316L_45deg
#   -> Emodul_<tag>.json  (E_apparent in the loading direction)
```

With MPI: `mpirun -np <N> python3 homogenize_rve.py ...`.

## Method notes

- **Stiffness assembly (`waam_crystal.py`):** per cell, `C_sample = M(g)·C_xtal·M(g)ᵀ`
  via the 6×6 stress Bond matrix `M` (Voigt order `xx,yy,zz,yz,xz,xy`, engineering
  shear — consistent with `alex.linearelastic`). Phase chosen per grain from the
  `crystal` label.
- **Homogenization (KUBC):** 6 load cases, macro strain `eps_mac` imposed as a
  linear displacement `u = eps_mac·x` on the whole boundary; column `k` of `Chom`
  = volume-averaged stress under unit macro-strain `k`. KUBC gives an **upper
  (stiff) bound**; for tighter bounds use periodic BCs (not implemented here).
- **Uniaxial test:** symmetric BCs (`u_n=0` on the three `min` faces) + applied
  `u_x=delta` on `x_max`, lateral faces free → apparent `E = <σ_xx>/(δ/Lx)`.
  For the V/H/45° bars this yields E parallel / perpendicular / at 45° to the
  build direction.
- **Units:** single-crystal C in GPa → results in GPa; mesh length unit (µm) is
  irrelevant for elastic moduli. Linear elasticity (Newton converges in 1 iter).

## Files

- `waam_crystal.py` — shared: mesh+grain reader, per-cell rotated cubic stiffness, anisotropic stress/energy/averaging
- `homogenize_rve.py` — KUBC homogenization → `Chom_<tag>.json`
- `uniaxial_tension.py` — directional uniaxial test → `Emodul_<tag>.json`
- `config.json` — single-crystal cubic constants per phase (edit me)
- `prepare_inputs.py` — host-side: copy Neper meshes + grain_ori into `inputs/`
