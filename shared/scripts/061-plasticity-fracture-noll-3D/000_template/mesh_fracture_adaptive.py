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
args = parser.parse_args()

# Set default parameters if CLI args are missing
Nholes = args.Nholes if args.Nholes is not None else 6
dhole = args.dhole if args.dhole is not None else 1.0
wsteg = args.wsteg if args.wsteg is not None else 1.0
nl = args.NL if args.NL is not None else 200

n_void_x = Nholes
n_void_y = 3
n_void_z = 3

n_ref = 1.0
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

wby =  (20.0 - (n_void_y * (dhole + wsteg))) / 2 
wb = (dhole+wsteg)
L = n_void_x * (dhole + wsteg) + 2 * wb
H = n_void_y * (dhole + wsteg) + 2 * wby
W = n_void_z * (dhole + wsteg) + 2 * wb

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
        "stretch_factor": [L - 2 * wb, H - 2 * wby, W - 2 * wb],
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
            cz = -W / 2 + wb + (vz + 0.5) * (dhole + wsteg)

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

if RecreateMesh and rank == 0:
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

comm.barrier()

# -----------------------------------------------------------
# IMPORT MSH INTO DOLFINX
# -----------------------------------------------------------

print(f"Rank {rank}: reading mesh from {MeshFile}")

domain, cell_markers, facet_markers = io.gmshio.read_from_msh(
    os.path.join(domain_mesh.MeshFilename), comm, gdim=ModelDim
)

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
    "nl": nl
}

if rank == 0:
    pp.append_to_file(parameter_path, params)
    print(f"Parameters saved to {parameter_path}")
