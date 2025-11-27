import sys
sys.path.append('..')
import dolfinx as dlfx
import numpy as np
import ufl
import argparse

import basix.ufl

from mpi4py import MPI
from petsc4py import PETSc

#from dolfinx.fem.petsc import NonlinearProblem
#from dolfinx.nls.petsc import NewtonSolver

from dolfinx import io

from pathlib import Path
import ufl.algebra
from ufl.algorithms.compute_form_data \
    import estimate_total_polynomial_degree

import utils.ronny.mesh as mesh
import alex.os as alexos
import alex.postprocessing as pp


import os
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
sys.stdout.flush()

#Argument parser
# parser=argparse.ArgumentParser()
# parser.add_argument("--rec", action="store_true", help="Recreate mesh")
# parser.add_argument("--dimensional", action="store_false", help="Dimensional formulation (True: dimensional, False: non-dimensional)")
# parser.add_argument("--NL", type=int, default=20, help="Number of elements in one direction (number of elements per length unit)")
# parser.add_argument("--n_ref", type=float, default=1.0, help="Mesh refinement factor")
# parser.add_argument("--mat_mod", choices=['lin_el', 'nl', 'l_nl', 'RO'], default='RO', help="constitutive law: lin_el: linear-elastic; RO: Ramberg-Osgood")
# parser.add_argument("--b_RO", type=float, default=0.05, help="slope control parameter b for R-O-law")
# parser.add_argument("--r_RO", type=float, default=10.0, help="curve transition parameter r for R-O-law")
# parser.add_argument("--incremental", action="store_true", help="Use incremental crack driving energy")
# parser.add_argument("--degrad", choices = ['quadratic', 'cubic'], default = 'cubic', help="degree of degradation function")
# parser.add_argument("--split", choices=['spectral', 'vol_dev'], default='vol_dev', help="Spectral split or volumetric-deviatoric split")
# parser.add_argument("--D", type=float, default=1.0, help="diameter of spherical inclusion")
# parser.add_argument("--W_s", type=float, default=1.0, help="web thickness between two voids")
# parser.add_argument("--W_b", type=float, default=0.5, help="boundary wall thickness")
# parser.add_argument("--n_void_x", type=int, default=2, help="number of voids in x-direction")
# parser.add_argument("--n_void_y", type=int, default=1, help="number of voids in y-direction")
# parser.add_argument("--n_void_z", type=int, default=2, help="number of voids in z-direction")
# args=parser.parse_args()

RecreateMesh = True

out_path = os.path.basename(__file__).removesuffix('.py') + '_output'
if rank == 0:
    Path(out_path).mkdir(parents=True, exist_ok=True)
comm.barrier()

ModelDim = 3

NL = 30 #mesh density

D = 1.0  #diameter of voids
W_s = 1.0  #web thickness between two voids
W_b = 0.5  #boundary wall thickness
n_void_x = 3
n_void_y = 3
n_void_z = 3

L = n_void_x*(D + W_s) + 2.0*W_b
H = n_void_y*(D + W_s) + 2.0*W_b
W = n_void_z*(D + W_s) + 2.0*W_b
NL = 30  #number of elements along one edge of the domain
n_ref = 1.0 #mesh refinement factor
print('Domain dimensions: L = ', L, ', H = ', H, ', W = ', W)
print('Inner dimensions: L_eff, H_eff, W_eff = ', L-2*W_b, H-2*W_b, W-2*W_b)

OutFile = str(out_path +'/' + os.path.basename(__file__).removesuffix('.py')) + '_N' + str(NL)
MeshFile = OutFile + '_mesh.msh'
###################################################################################################################################
#Matrix
matrix_marker = 0

#Inclusions
n_i = 1 #5 #15           # number of inclusions
inclusions = {#type: 'inclusion' or 'void', material, shape: 'ellipsoid' or 'rectangle', center: [x,y,z], length (for ellipsoid: radius), stretch_factor: [f_x, f_y, f_z], rotation_axis: [x,y,z], rotation_angle [rad]
    0: {'material': 'TiN_mod', 'shape': 'rectangle', 'center': [0.0, 0.0, 0.0], 'length': 1.0, 'stretch_factor': [L-2*W_b, H-2*W_b, W-2*W_b], 'rotation_axis': [0.0, 0.0, 1.0], 'rotation_angle': 0.0},#-np.pi/4},#np.pi/6}, 'stretch_factor': [L-2*args.W_b, H-2*args.W_b, W-2*args.W_b]
    }

inclusion_marker = []
inclusion_surface_marker = []
for i in range(n_i):
    inclusion_marker.append(i+1)
    inclusion_surface_marker.append(n_i*10+i+1) 

#Voids
n_v = n_void_x*n_void_y*n_void_z #number of voids
voids = {} #
for v_x in range(n_void_x):
    for v_y in range(n_void_y):
        for v_z in range(n_void_z):
            voids[(v_x, v_y, v_z)] = {'shape': 'ellipsoid', 'center': [-L/2+W_b+(v_x+0.5)*(D+W_s), -H/2+W_b+(v_y+0.5)*(D+W_s), -W/2+W_b+(v_z+0.5)*(D+W_s)], 'length': D/2, 'stretch_factor': [1.0, 1.0, 1.0], 'rotation_axis': [0.0, 0.0, 1.0], 'rotation_angle': 0.0}
            print('Void ', (v_x, v_y, v_z), ': center = ', voids[(v_x, v_y, v_z)]['center'])
            print('                  length = ', voids[(v_x, v_y, v_z)]['length'])
 
#Mesh generation
if RecreateMesh:
    if rank == 0:
        domain_mesh = mesh.MatrixPores3D(L=L, H=H, W=W, NL=NL, n_i=n_i, inclusions=inclusions, n_v=n_v, voids=voids, Hexa=False, MeshName='-', MeshFilename = MeshFile, matrix_marker=matrix_marker, inclusion_marker=inclusion_marker, inclusion_surface_marker=inclusion_surface_marker, Hertzian=False, R_ind=0.0, n_void_x=n_void_x, n_void_y=n_void_y, n_void_z=n_void_z)
        domain_mesh.create(n_ref=n_ref) #-> .msh file
    comm.barrier()

#domain, cell_markers, facet_markers, ridge_tags, peak_tags, physical_groups = io.gmshio.read_from_msh(MeshFile, comm, gdim=ModelDim)
domain, cell_markers, facet_markers = io.gmshio.read_from_msh(MeshFile, comm, gdim=ModelDim)


script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]

outputfile_xdmf_path = alexos.outputfile_xdmf_full_path(
    script_path, script_name_without_extension
)

if rank == 0:
    print(f"Writing XDMF mesh to: {outputfile_xdmf_path}")

pp.write_meshoutputfile(domain, outputfile_xdmf_path, comm)

import alex.heterogeneous as het

micro_material_marker = 1
effective_material_marker = 0



micro_material_cells = cell_markers.find(micro_material_marker)
effective_material_cells = cell_markers.find(effective_material_marker)

la_micro = 1
la_effective = 0

la = het.set_cell_function_heterogeneous_material(domain,la_micro, la_effective, micro_material_cells, effective_material_cells)


def write_scalar_fields(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, scalar_fields_as_functions, scalar_field_names, outputfile_xdmf_path: str, t: float):
    Se = basix.ufl.element("DG", domain.basix_cell(), 0, shape=())
    S = dlfx.fem.functionspace(domain, Se)
    # S= dlfx.fem.functionspace(domain, ("DP", 0, ()))
    xdmf_out = dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a')
    for n  in range(0,len(scalar_fields_as_functions)):
            scalar_field_function = scalar_fields_as_functions[n]
            scalar_field_name = scalar_field_names[n]
            scalar_field_expression = dlfx.fem.Expression(scalar_field_function, 
                                                        S.element.interpolation_points())
            out_scalar_field = dlfx.fem.Function(S)
            out_scalar_field.interpolate(scalar_field_expression)
            out_scalar_field.name = scalar_field_name
            
            xdmf_out.write_function(out_scalar_field,t)
    xdmf_out.close()

write_scalar_fields(domain,comm,scalar_field_names=["la"],scalar_fields_as_functions=[la],outputfile_xdmf_path=outputfile_xdmf_path,t=0)

a=1