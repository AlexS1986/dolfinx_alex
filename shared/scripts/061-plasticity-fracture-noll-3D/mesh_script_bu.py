import sys
sys.path.append("..")

import os
from pathlib import Path
import numpy as np
from mpi4py import MPI

import dolfinx as dlfx
from dolfinx import io

import ronny.mesh as mesh
import alex.postprocessing as pp
import alex.os as alexos

import alex.heterogeneous as het

import basix

# -----------------------------------------------------------
# USER PARAMETERS
# -----------------------------------------------------------

RecreateMesh = True

D = 1.0
W_s = 1.0
W_b = 0.5

n_void_x = 6
n_void_y = 3
n_void_z = 3

NL = 50
n_ref = 1.0

MeshFile = "domain_mesh.msh"


# -----------------------------------------------------------
# MPI
# -----------------------------------------------------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
sys.stdout.flush()


# -----------------------------------------------------------
# DOMAIN DIMENSIONS
# -----------------------------------------------------------

L = n_void_x * (D + W_s) + 2 * W_b
H = n_void_y * (D + W_s) + 2 * W_b
W = n_void_z * (D + W_s) + 2 * W_b

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
        "stretch_factor": [L - 2 * W_b, H - 2 * W_b, W - 2 * W_b],
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
            cx = -L / 2 + W_b + (vx + 0.5) * (D + W_s)
            cy = -H / 2 + W_b + (vy + 0.5) * (D + W_s)
            cz = -W / 2 + W_b + (vz + 0.5) * (D + W_s)

            voids[(vx, vy, vz)] = {
                "shape": "ellipsoid",
                "center": [cx, cy, cz],
                "length": D / 2,
                "stretch_factor": [1, 1, 1],
                "rotation_axis": [0, 0, 1],
                "rotation_angle": 0.0
            }

            if rank == 0:
                print(f"Void {(vx, vy, vz)}: center = {cx, cy, cz}")


# -----------------------------------------------------------
# CREATE GMSH MESH
# -----------------------------------------------------------

ModelDim = 3
matrix_marker = 0

if RecreateMesh:
    if rank == 0:
        print("Creating mesh with MatrixPores3D ...")

        domain_mesh = mesh.MatrixPores3D(
            L=L,
            H=H,
            W=W,
            NL=NL,
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
    os.path.join("/home/",domain_mesh.MeshFilename), comm, gdim=ModelDim
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

marker_field = het.set_cell_function_heterogeneous_material(domain,marker_inside, marker_outside, 
                                                            cells_inside, cells_outside)
marker_field.name = "material"


pp.write_field(domain,outputfile_xdmf_path,marker_field,0.0,comm,S=S0)

with io.XDMFFile(comm, outputfile_xdmf_path, "a") as xdmf:
    xdmf.write_meshtags(cell_markers, domain.geometry)
# -----------------------------------------------------------
# DONE — domain is ready
# -----------------------------------------------------------

if rank == 0:
    print("Mesh written to XDMF and ready to use.")

