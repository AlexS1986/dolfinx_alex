import sys
import os
from mpi4py import MPI
import argparse
from pathlib import Path

import dolfinx as dlfx
from dolfinx import io

import ronny.mesh as mesh
import alex.postprocessing as pp
import alex.os as alexos
import alex.heterogeneous as het
import basix

# -----------------------------------------------------------
# ARGUMENT PARSING
# -----------------------------------------------------------
parser = argparse.ArgumentParser(description="Generate a cube mesh with a centered spherical pore.")
parser.add_argument("--dhole", type=float, default=1.0, help="Diameter of the spherical pore")
parser.add_argument("--wsteg", type=float, default=1.0, help="Cube size minus hole (steg)")
parser.add_argument("--NL", type=int, default=5, help="Number of elements along cube edge")
args = parser.parse_args()

dhole = args.dhole
wsteg = args.wsteg
nl = args.NL

# -----------------------------------------------------------
# MPI SETUP
# -----------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
sys.stdout.flush()

# -----------------------------------------------------------
# DOMAIN DIMENSIONS
# -----------------------------------------------------------
cube_size = dhole + wsteg  # Cube side length
L = H = W = cube_size

if rank == 0:
    print("Cube dimensions:", L, H, W)
    print("Pore diameter:", dhole)

# -----------------------------------------------------------
# VOIDS
# -----------------------------------------------------------
voids = {
    (0,0,0): {
        "shape": "ellipsoid",
        "center": [0.0, 0.0, 0.0],
        "length": dhole / 2,
        "stretch_factor": [1, 1, 1],
        "rotation_axis": [0, 0, 1],
        "rotation_angle": 0.0
    }
}
n_v = 1  # only one pore

# -----------------------------------------------------------
# CREATE MESH
# -----------------------------------------------------------
ModelDim = 3
matrix_marker = 0
RecreateMesh = True
MeshFile = "cube_with_pore.msh"

if RecreateMesh and rank == 0:
    print("Creating mesh with MatrixPores3D ...")
    
    # Use the new safe MatrixPores3D class
    domain_mesh = mesh.MatrixPores3D_save(
        L=L,
        H=H,
        W=W,
        NL=nl,
        n_i=0,               # no inclusions
        inclusions={},       # empty dict
        n_v=n_v,
        voids=voids,
        Hexa=False,
        MeshName="-",
        MeshFilename=MeshFile,
        matrix_marker=matrix_marker,
        inclusion_marker=[],
        inclusion_surface_marker=[],
        Hertzian=False,
        R_ind=0.0,
        n_void_x=1,
        n_void_y=1,
        n_void_z=1
    )
    
    domain_mesh.create(n_ref=1.0)

comm.barrier()

# -----------------------------------------------------------
# IMPORT MSH INTO DOLFINX
# -----------------------------------------------------------
mesh_path = os.path.join("/home/", MeshFile)
print(f"Rank {rank}: reading mesh from {mesh_path}")

domain, cell_markers, facet_markers = io.gmshio.read_from_msh(mesh_path, comm, gdim=ModelDim)

if rank == 0:
    print("Mesh imported into DOLFINx.")

# -----------------------------------------------------------
# WRITE XDMF OUTPUT
# -----------------------------------------------------------
script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]

outputfile_xdmf_path = alexos.outputfile_xdmf_full_path(script_path, script_name_without_extension)

if rank == 0:
    print(f"Writing XDMF mesh to: {outputfile_xdmf_path}")

pp.write_meshoutputfile(domain, outputfile_xdmf_path, comm)

# -----------------------------------------------------------
# SAVE PARAMETERS TO FILE
# -----------------------------------------------------------
parameter_path = os.path.join(script_path, "parameters.txt")
params = {
    "dhole": dhole,
    "wsteg": wsteg,
    "NL": nl
}

if rank == 0:
    pp.append_to_file(parameter_path, params)
    print(f"Parameters saved to {parameter_path}")
