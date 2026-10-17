import sys
sys.path.append("..")

import os
from pathlib import Path
import numpy as np
from mpi4py import MPI
import argparse

import dolfinx as dlfx
from dolfinx import io

import ronny.mesh as mesh
import alex.postprocessing as pp
import alex.os as alexos
import alex.heterogeneous as het
import basix

script_path = os.path.dirname(__file__)

# -----------------------------------------------------------
# ARGUMENT PARSING
# -----------------------------------------------------------

parser = argparse.ArgumentParser(description="Generate domain mesh with voids and inclusions.")
parser.add_argument("--Nholes", type=int, help="Number of holes in x-direction")
parser.add_argument("--dhole", type=float, help="Diameter of circular holes")
parser.add_argument("--wsteg", type=float, help="Width of steg between holes")
parser.add_argument("--NL", type=int, help="Number of elements along domain length")
parser.add_argument("--geometry", type=str, default="porous", choices=["porous", "strip", "solid"],
                    help="porous: spherical pores in every cell (ronny.mesh.MatrixPores3D); strip/solid: the same box without pores, "
                         "meshed with gmsh directly using the same refinement box (solid = the job script also makes the embedding plastic)")
args = parser.parse_args()
geometry = args.geometry

# Set default parameters if CLI args are missing
Nholes = args.Nholes if args.Nholes is not None else 1
dhole = args.dhole if args.dhole is not None else 1.0
wsteg = args.wsteg if args.wsteg is not None else 0.25
nl = args.NL if args.NL is not None else 5

n_void_x = Nholes
n_void_y = 3 #3
n_void_z = 2 #3

# NL 4 and nref 4 works

domain_height_y = 20.0 # 20 usually
n_ref = 20.0
MeshFile = os.path.join(script_path,"domain_mesh.msh")
RecreateMesh = True

# -----------------------------------------------------------
# MPI SETUP
# -----------------------------------------------------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
sys.stdout.flush()

# -----------------------------------------------------------
# DOMAIN DIMENSIONS
# -----------------------------------------------------------

wby =  (domain_height_y - (n_void_y * (dhole + wsteg))) / 2 

wb = (dhole+wsteg)
wbz = wb / 4
#wby = wb
L = n_void_x * (dhole + wsteg) + 2 * wb
H = n_void_y * (dhole + wsteg) + 2 * wby
W = n_void_z * (dhole + wsteg) + 2 * wbz

if rank == 0:
    print("Domain dimensions:", L, H, W)

# -----------------------------------------------------------
# INCLUSIONS
# -----------------------------------------------------------

n_i = 1
inclusions = {
    0: {
        "material": "dummy",
        "shape": "rectangle",
        "center": [0.0, 0.0, 0.0],
        "length": 1.0,
        "stretch_factor": [L - 2 * wb, H - 2 * wby, W - 2 * wbz ],
        "rotation_axis": [0, 0, 1],
        "rotation_angle": 0.0
    }
}

inclusion_marker = [i + 1 for i in range(n_i)]
inclusion_surface_marker = [10 + i + 1 for i in range(n_i)]

# -----------------------------------------------------------
# VOIDS
# -----------------------------------------------------------

n_v = n_void_x * n_void_y * n_void_z
voids = {}

for vx in range(n_void_x):
    for vy in range(n_void_y):
        for vz in range(n_void_z):
            cx = -L / 2 + wb + (vx + 0.5) * (dhole + wsteg)
            cy = -H / 2 + wby + (vy + 0.5) * (dhole + wsteg)
            cz = -W / 2 + wbz+ (vz + 0.5) * (dhole + wsteg)

            voids[(vx, vy, vz)] = {
                "shape": "ellipsoid",
                "center": [cx, cy, cz],
                "length": dhole / 2,
                "stretch_factor": [1, 1, 1],
                "rotation_axis": [0, 0, 1],
                "rotation_angle": 0.0
            }

            if rank == 0:
                print(f"Void {(vx, vy, vz)}: center = {cx, cy, cz}")

# -----------------------------------------------------------
# CREATE MESH
# -----------------------------------------------------------

ModelDim = 3
matrix_marker = 0

def create_solid_mesh(L, H, W, nl, n_ref, inclusions, matrix_marker, inclusion_marker, mesh_file):
    """Box L x H x W without pores: outer matrix (marker 0, the 'effective' region) and the inner block (marker 1, the
    'micro' region of the porous runs), meshed with the same background field as MatrixPores3D.mesh_refinement(n_ref,
    "middle_section"): element size (L/NL)/n_ref for |y| < 1.5 * (pore radius), L/NL elsewhere (HXT, algorithm 10)."""
    import gmsh
    gmsh.initialize()
    gmsh.model.add("solid")
    box = gmsh.model.occ.addBox(-L / 2, -H / 2, -W / 2, L, H, W)
    sx, sy, sz = inclusions[0]["stretch_factor"]
    inner = gmsh.model.occ.addBox(-sx / 2, -sy / 2, -sz / 2, sx, sy, sz)
    gmsh.model.occ.synchronize()
    out, _ = gmsh.model.occ.fragment([(3, box)], [(3, inner)], removeTool=False)
    gmsh.model.occ.synchronize()
    vols = [t for d, t in out if d == 3]
    # the fragment keeps the inner box tag; the remaining volume is the matrix shell
    inner_tags = [t for t in vols if abs(gmsh.model.occ.getMass(3, t) - sx * sy * sz) < 1e-6 * sx * sy * sz]
    outer_tags = [t for t in vols if t not in inner_tags]
    assert len(inner_tags) == 1 and len(outer_tags) >= 1, (vols, inner_tags)
    gmsh.model.addPhysicalGroup(3, outer_tags, matrix_marker)
    gmsh.model.addPhysicalGroup(3, inner_tags, inclusion_marker)
    h_el = L / nl
    r_pore = dhole / 2.0
    f = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(f, "VIn", h_el / n_ref)
    gmsh.model.mesh.field.setNumber(f, "VOut", h_el)
    gmsh.model.mesh.field.setNumber(f, "XMin", -L); gmsh.model.mesh.field.setNumber(f, "XMax", L)
    gmsh.model.mesh.field.setNumber(f, "YMin", -1.5 * r_pore); gmsh.model.mesh.field.setNumber(f, "YMax", 1.5 * r_pore)
    gmsh.model.mesh.field.setNumber(f, "ZMin", -W); gmsh.model.mesh.field.setNumber(f, "ZMax", W)
    gmsh.model.mesh.field.setNumber(f, "Thickness", 2.0 * W)
    gmsh.model.mesh.field.setAsBackgroundMesh(f)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(mesh_file)
    gmsh.finalize()


class _MeshStub:
    def __init__(self, mesh_file):
        self.MeshFilename = mesh_file


if RecreateMesh and geometry != "porous":
    if rank == 0:
        print(f"Creating solid mesh ({geometry}, no pores) with gmsh ...")
        create_solid_mesh(L, H, W, nl, n_ref, inclusions, matrix_marker, inclusion_marker[0], MeshFile)
    domain_mesh = _MeshStub(MeshFile)
elif RecreateMesh and rank == 0:
    print("Creating mesh with MatrixPores3D ...")

    domain_mesh = mesh.MatrixPores3D(
        L=L,
        H=H,
        W=W,
        NL=nl,
        n_i=n_i,
        inclusions=inclusions,
        n_v=n_v,
        voids=voids,
        Hexa=False,
        MeshName="-",
        MeshFilename=MeshFile,
        matrix_marker=matrix_marker,
        inclusion_marker=inclusion_marker,
        inclusion_surface_marker=inclusion_surface_marker,
        Hertzian=False,
        R_ind=0.0,
        n_void_x=n_void_x,
        n_void_y=n_void_y,
        n_void_z=n_void_z
    )

    domain_mesh.create(n_ref=n_ref)
    #domain_mesh.create()

comm.barrier()
MeshFile = comm.bcast(MeshFile, root=0)

# -----------------------------------------------------------
# IMPORT MSH INTO DOLFINX
# -----------------------------------------------------------

print(f"Rank {rank}: reading mesh from {MeshFile}")

domain, cell_markers, facet_markers = io.gmshio.read_from_msh(MeshFile, comm, gdim=ModelDim)

if rank == 0:
    print("Mesh imported into DOLFINx.")

# -----------------------------------------------------------
# WRITE XDMF OUTPUT
# -----------------------------------------------------------

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]

outputfile_xdmf_path = alexos.outputfile_xdmf_full_path(
    script_path, script_name_without_extension
)

if rank == 0:
    print(f"Writing XDMF mesh to: {outputfile_xdmf_path}")

pp.write_meshoutputfile(domain, outputfile_xdmf_path, comm)

marker_outside = 1
marker_inside = 0

cells_inside = cell_markers.find(marker_inside)
cells_outside = cell_markers.find(marker_outside)

S0e = basix.ufl.element("DP", domain.basix_cell(), 0, shape=())
S0 = dlfx.fem.functionspace(domain, S0e)

marker_field = het.set_cell_function_heterogeneous_material(
    domain, marker_inside, marker_outside, cells_inside, cells_outside
)
marker_field.name = "material"

pp.write_field(domain, outputfile_xdmf_path, marker_field, 0.0, comm, S=S0)

with io.XDMFFile(comm, outputfile_xdmf_path, "a") as xdmf:
    xdmf.write_meshtags(cell_markers, domain.geometry)

if rank == 0:
    print("Mesh written to XDMF and ready to use.")

# -----------------------------------------------------------
# SAVE PARAMETERS TO FILE
# -----------------------------------------------------------

parameter_path = os.path.join(script_path, "parameters.txt")
params = {
    "Nholes": Nholes,
    "dhole": dhole,
    "wsteg": wsteg,
    "wb": wb,
    "nl": nl,
    "geometry": geometry
}

if rank == 0:
    pp.append_to_file(parameter_path, params)
    print(f"Parameters saved to {parameter_path}")

