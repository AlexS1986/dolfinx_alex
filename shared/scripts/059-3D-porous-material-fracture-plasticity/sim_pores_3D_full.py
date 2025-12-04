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

import util.mesh as mesh
import util.phasefield as phasefield
import util.aux as aux
import util.material_models as mm
#import util.geometry as geo
import util.materials as materials

import os
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
sys.stdout.flush()

#Argument parser
parser=argparse.ArgumentParser()
parser.add_argument("--rec", action="store_true", help="Recreate mesh")
parser.add_argument("--dimensional", action="store_false", help="Dimensional formulation (True: dimensional, False: non-dimensional)")
parser.add_argument("--NL", type=int, default=20, help="Number of elements in one direction (number of elements per length unit)")
parser.add_argument("--n_ref", type=float, default=1.0, help="Mesh refinement factor")
parser.add_argument("--mat_mod", choices=['lin_el', 'nl', 'l_nl', 'RO'], default='RO', help="constitutive law: lin_el: linear-elastic; RO: Ramberg-Osgood")
parser.add_argument("--b_RO", type=float, default=0.05, help="slope control parameter b for R-O-law")
parser.add_argument("--r_RO", type=float, default=10.0, help="curve transition parameter r for R-O-law")
parser.add_argument("--incremental", action="store_true", help="Use incremental crack driving energy")
parser.add_argument("--degrad", choices = ['quadratic', 'cubic'], default = 'cubic', help="degree of degradation function")
parser.add_argument("--split", choices=['spectral', 'vol_dev'], default='vol_dev', help="Spectral split or volumetric-deviatoric split")
parser.add_argument("--D", type=float, default=1.0, help="diameter of spherical inclusion")
parser.add_argument("--W_s", type=float, default=1.0, help="web thickness between two inclusions")
parser.add_argument("--W_b", type=float, default=0.5, help="boundary wall thickness")
parser.add_argument("--n_void_x", type=int, default=2, help="number of voids in x-direction")
parser.add_argument("--n_void_y", type=int, default=1, help="number of voids in y-direction")
parser.add_argument("--n_void_z", type=int, default=2, help="number of voids in z-direction")
args=parser.parse_args()

RecreateMesh = args.rec
NonDim = args.dimensional
Incremental = args.incremental
if args.mat_mod == 'RO':
    Incremental = True #enforce incremental formulation for R-O material model
if Incremental:
    scheme = 'incremental'
else:
    scheme = 'regular'

out_path = os.path.basename(__file__).removesuffix('.py') + '_output'
if rank == 0:
    Path(out_path).mkdir(parents=True, exist_ok=True)
comm.barrier()

OutFile = str(out_path +'/' + os.path.basename(__file__).removesuffix('.py')) + '_N' + str(args.NL)
MeshFile = OutFile + '_mesh.msh'
OutFile += '_' + args.mat_mod + '_b' + str(args.b_RO) + '_' + args.split + '_' + scheme
XdmfMeshFile = OutFile + '_mesh.xdmf'
AuxFile = OutFile + '_aux.xdmf'

ModelDim = 3
L = args.n_void_x*(args.D + args.W_s) + 2.0*args.W_b
H = args.n_void_y*(args.D + args.W_s) + 2.0*args.W_b
W = args.n_void_z*(args.D + args.W_s) + 2.0*args.W_b
NL = args.NL  #number of elements along one edge of the domain
n_ref = args.n_ref #mesh refinement factor
print('Domain dimensions: L = ', L, ', H = ', H, ', W = ', W)
print('Inner dimensions: L_eff, H_eff, W_eff = ', L-2*args.W_b, H-2*args.W_b, W-2*args.W_b)

###################################################################################################################################
#Material parameters
##Matrix
matrix_marker = 0

matrix_material = "100Cr6"

Emod_m_base = dlfx.default_scalar_type(materials.materials_dict[matrix_material]["Emod"]) #Young's modulus
nu_m_base = dlfx.default_scalar_type(materials.materials_dict[matrix_material]["nu"]) #Poisson's ratio
Gmod_m_base = mm.mu_from_E_nu(E = Emod_m_base, nu = nu_m_base) #Shear modulus (matrix)
Kmod_m_base = mm.Kmod_from_E_nu(E = Emod_m_base, nu = nu_m_base, dim = ModelDim) #Bulk modulus (matrix)
Gc_m_base = dlfx.default_scalar_type(materials.materials_dict[matrix_material]["GIc"]) #critical energy release rate
eps_y_m_base = dlfx.default_scalar_type(5.e-3) #yield strain [-]
b_RO_m_base = dlfx.default_scalar_type(args.b_RO)
r_RO_m_base = dlfx.default_scalar_type(args.r_RO)

pen_base = dlfx.default_scalar_type(1.0e3*Emod_m_base) #penalty parameter for contact

##Inclusions
n_i = 1 #5 #15           # number of inclusions
inclusions = {#type: 'inclusion' or 'void', material, shape: 'ellipsoid' or 'rectangle', center: [x,y,z], length (for ellipsoid: radius), stretch_factor: [f_x, f_y, f_z], rotation_axis: [x,y,z], rotation_angle [rad]
    0: {'material': 'TiN_mod', 'shape': 'rectangle', 'center': [0.0, 0.0, 0.0], 'length': 1.0, 'stretch_factor': [L-2*args.W_b, H-2*args.W_b, W-2*args.W_b], 'rotation_axis': [0.0, 0.0, 1.0], 'rotation_angle': 0.0},#-np.pi/4},#np.pi/6}, 'stretch_factor': [L-2*args.W_b, H-2*args.W_b, W-2*args.W_b]
    }

inclusion_marker = []
inclusion_surface_marker = []
for i in range(n_i):
    inclusion_marker.append(i+1) #'ellipsoid' 'rectangular' 'cylinder''
    inclusion_surface_marker.append(n_i*10+i+1)

Emod_i_base = []
nu_i_base = []
Gmod_i_base =[]
Kmod_i_base = []
Gc_i_base = []

for i in range(n_i):
    Emod_i_base.append(dlfx.default_scalar_type(materials.materials_dict[inclusions[i]['material']]["Emod"])) #Young's modulus (inclusion)
    nu_i_base.append(dlfx.default_scalar_type(materials.materials_dict[inclusions[i]['material']]["nu"])) #Poisson's ratio (inclusion)
    Gmod_i_base.append(mm.mu_from_E_nu(E = Emod_i_base[i], nu = nu_i_base[i])) #Shear modulus (inclusion)
    Kmod_i_base.append(mm.Kmod_from_E_nu(E = Emod_i_base[i], nu = nu_i_base[i], dim = ModelDim)) #Bulk modulus (inclusion)
    Gc_i_base.append(dlfx.default_scalar_type(materials.materials_dict[inclusions[i]['material']]["GIc"])) #critical energy release rate (inclusion)

#Voids
n_v = args.n_void_x*args.n_void_y*args.n_void_z #8 #number of voids
voids = {} # dict() #type: 'inclusion' or 'void', material, shape: 'ellipsoid' or 'rectangle', center: [x,y,z], length (for ellipsoid: radius), stretch_factor: [f_x, f_y, f_z], rotation_axis: [x,y,z], rotation_angle [rad]
for v_x in range(args.n_void_x):
    for v_y in range(args.n_void_y):
        for v_z in range(args.n_void_z):
            voids[(v_x, v_y, v_z)] = {'shape': 'ellipsoid', 'center': [-L/2+args.W_b+(v_x+0.5)*(args.D+args.W_s), -H/2+args.W_b+(v_y+0.5)*(args.D+args.W_s), -W/2+args.W_b+(v_z+0.5)*(args.D+args.W_s)], 'length': args.D/2, 'stretch_factor': [1.0, 1.0, 1.0], 'rotation_axis': [0.0, 0.0, 1.0], 'rotation_angle': 0.0}
            print('Void ', (v_x, v_y, v_z), ': center = ', voids[(v_x, v_y, v_z)]['center'])
            print('                  length = ', voids[(v_x, v_y, v_z)]['length'])
    
    
    
    # 0: {'shape': 'ellipsoid', 'center': [-0.4, 0.0, 0.0], 'length': 0.03, 'stretch_factor': [1.0, 1.0, 1.0], 'rotation_axis': [0.0, 0.0, 1.0], 'rotation_angle': -np.pi/8},
    # 1: {'shape': 'ellipsoid', 'center': [0.3, -0.2, 0.0], 'length': 0.01, 'stretch_factor': [1.6, 1.0, 1.0], 'rotation_axis': [0.0, 0.0, 1.0], 'rotation_angle': np.pi/8},
    # }
assert len(inclusions) >= n_i, 'Error: length of inclusion array must at least be equal to n_i!'
assert len(voids) >= n_v, 'Error: length of void array must at least be equal to n_v!'

#Mesh generation
if RecreateMesh:
    if rank == 0:
        domain_mesh = mesh.MatrixPores3D(L=L, H=H, W=W, NL=NL, n_i=n_i, inclusions=inclusions, n_v=n_v, voids=voids, Hexa=False, MeshName='-', MeshFilename = MeshFile, matrix_marker=matrix_marker, inclusion_marker=inclusion_marker, inclusion_surface_marker=inclusion_surface_marker, Hertzian=False, R_ind=0.0, n_void_x=args.n_void_x, n_void_y=args.n_void_y, n_void_z=args.n_void_z)
        domain_mesh.create(n_ref=n_ref)
    comm.barrier()

domain, cell_markers, facet_markers = io.gmshio.read_from_msh(MeshFile, comm, gdim=ModelDim)

matrix_cells = cell_markers.find(matrix_marker)
inclusion_cells = []
for i in range(n_i):
    inclusion_cells.append(cell_markers.find(inclusion_marker[i]))

dim = domain.geometry.dim
fdim = dim-1

entities = dlfx.mesh.locate_entities(domain, dim, lambda x: np.full(np.shape(x)[1], True))
hmin = comm.allreduce(np.min(domain.h(domain.topology.dim, entities)), op=MPI.MIN)

# Define boundaries
def right(x):
    return np.isclose(x[0], L/2)

def left(x):
    return np.isclose(x[0], -L/2)

def bottom(x):
    return np.isclose(x[1], -H/2)

def top(x):
    return np.isclose(x[1], H/2)

def front(x):
    return np.isclose(x[2], W/2)

def back(x):
    return np.isclose(x[2], -W/2)

#Time-stepping parameters
Tend = 1.0 #End time
dt_ini = dlfx.default_scalar_type(0.0001) #Initial time step size
dt = dlfx.fem.Constant(domain, dt_ini)
t = dlfx.fem.Constant(domain, -dt_ini) #Initial time
dt_scale_down = 0.5
dt_scale_up = 1.25
max_dt = 0.01
min_iters = 2
max_iters = 8

#Function spaces
elementV = basix.ufl.element('P', domain.basix_cell(), 1, shape=(dim, ))
elementS = basix.ufl.element('P', domain.basix_cell(), 1)
elementT = basix.ufl.element('P', domain.basix_cell(), 0, discontinuous=True, shape=(dim, dim,))
elementV_lin = basix.ufl.element('P', domain.basix_cell(), 1, shape=(dim, ))
elementS_lin = basix.ufl.element('P', domain.basix_cell(), 1)

mel = basix.ufl.mixed_element([elementV, elementS])

M = dlfx.fem.functionspace(domain, mel)
V1, mapu = M.sub(0).collapse()    # sub-function space u (vector)
S1, maps = M.sub(1).collapse()    # sub-function space s (scalar)
T = dlfx.fem.functionspace(domain, elementT)  # Tensor function space
V_lin = dlfx.fem.functionspace(domain, elementV_lin)
S_lin = dlfx.fem.functionspace(domain, elementS_lin)

DG0 = dlfx.fem.functionspace(domain, basix.ufl.element('P', domain.basix_cell(), 0, discontinuous=True))

# # Scalar function space
# elementSC = basix.ufl.element('CG', domain.basix_cell(), 1)
# SC = dlfx.fem.functionspace(domain, elementSC)

# restart fields
# urestart = dlfx.fem.Function(V1)
# trestart = dlfx.fem.Constant(domain, -dt_ini)# -dt_ini) #0.0

# Combined displacement and phase field function
w = dlfx.fem.Function(M)
dw = ufl.TestFunction(M)
w_old = dlfx.fem.Function(M)

#Split functions
u, s = ufl.split(w)
du, ds = ufl.split(dw)
u_old, s_old = ufl.split(w_old)

# restart fields
wrestart = dlfx.fem.Function(M)
s_zerorestart = dlfx.fem.Function(S1)
trestart = dlfx.fem.Constant(domain, -dt_ini)

#Phasefield
epsilon_base = dlfx.default_scalar_type(2*hmin)#(2*1/(n_ref*NL))#(1/300)

mob_base = dlfx.default_scalar_type(10000.0) 

eta = dlfx.fem.Constant(domain, dlfx.default_scalar_type(1e-9)) #Residual stiffness
s0_tol = dlfx.default_scalar_type(5.0e-2) #Crack healing threshold

pf = phasefield.Phasefield(DG0 = DG0, degrad_type=args.degrad, split_type=args.split, incremental=Incremental)

degrad, diff_degrad = pf.degrad_dict[args.degrad]
psi_s = pf.psi_s
crack_driving_energy = pf.crack_driving_energy_dict[args.split][scheme]
crack_driving_energy_old = pf.crack_driving_energy_old

#Distribution of material parameters
if NonDim:
    Emod_m, Emod_i, Gmod_m, Gmod_i, Kmod_m, Kmod_i, Gc_m, Gc_i, eps_y_m, pen = aux.nondim_E_nu(ModelDim=ModelDim, L=L, Emod_m_base=Emod_m_base, Emod_i_base=Emod_i_base, Gmod_m_base=Gmod_m_base, Gmod_i_base=Gmod_i_base, nu_m_base=nu_m_base, nu_i_base=nu_i_base, Gc_m_base=Gc_m_base, Gc_i_base=Gc_i_base, eps_y_m_base=eps_y_m_base, pen_base=pen_base)
    mob = dlfx.fem.Constant(domain, Gc_m_base*1.0/L*mob_base) #non-dimensional mobility constant
    epsilon = dlfx.fem.Constant(domain, dlfx.default_scalar_type(epsilon_base/L)) #width of transition zone for PF (non-dimensional)
    dimensionality_displacement_factor = ufl.sqrt(2*Gmod_m_base/(Gc_m_base*L))
    dimensionality_traction_factor = ufl.sqrt(L/(2*Gmod_m_base*Gc_m_base))
    file = OutFile + '_nondim.xdmf'
else:
    Emod_m = Emod_m_base
    Emod_i = Emod_i_base
    Gmod_m = Gmod_m_base
    Gmod_i = Gmod_i_base
    Kmod_m = Kmod_m_base
    Kmod_i = Kmod_i_base
    Gc_m = Gc_m_base
    Gc_i = Gc_i_base
    eps_y_m = eps_y_m_base
    pen = pen_base
    mob = dlfx.fem.Constant(domain, mob_base) #Mobility constant
    epsilon = dlfx.fem.Constant(domain, dlfx.default_scalar_type(epsilon_base)) #width of transition zone for PF (dimensional)
    dimensionality_displacement_factor = 1.0
    dimensionality_traction_factor = 1.0
    file = OutFile + '.xdmf'

Emod = dlfx.fem.Function(DG0) #Young's modulus
Emod.x.array[matrix_cells] = np.full_like(matrix_cells, Emod_m, dtype=dlfx.default_scalar_type)
for i in range(n_i):
    Emod.x.array[inclusion_cells[i]] = np.full_like(inclusion_cells[i], Emod_i[i], dtype=dlfx.default_scalar_type)
Emod.x.scatter_forward()

nu = dlfx.fem.Function(DG0) #Poisson's ratio
nu.x.array[matrix_cells] = np.full_like(matrix_cells, nu_m_base, dtype=dlfx.default_scalar_type)
for i in range(n_i):
    nu.x.array[inclusion_cells[i]] = np.full_like(inclusion_cells[i], nu_i_base[i], dtype=dlfx.default_scalar_type)
nu.x.scatter_forward()

Gmod = dlfx.fem.Function(DG0) #Shear modulus
Gmod.x.array[matrix_cells] = np.full_like(matrix_cells, Gmod_m, dtype=dlfx.default_scalar_type)
for i in range(n_i):
    Gmod.x.array[inclusion_cells[i]] = np.full_like(inclusion_cells[i], Gmod_i[i], dtype=dlfx.default_scalar_type)
Gmod.x.scatter_forward()

Kmod = dlfx.fem.Function(DG0) #Bulk modulus
Kmod.x.array[matrix_cells] = np.full_like(matrix_cells, Kmod_m, dtype=dlfx.default_scalar_type)
for i in range(n_i):
    Kmod.x.array[inclusion_cells[i]] = np.full_like(inclusion_cells[i], Kmod_i[i], dtype=dlfx.default_scalar_type)
Kmod.x.scatter_forward()

eps_y = dlfx.fem.Function(DG0) #Young's modulus
eps_y.x.array[matrix_cells] = np.full_like(matrix_cells, eps_y_m, dtype=dlfx.default_scalar_type)
for i in range(n_i):
    eps_y.x.array[inclusion_cells[i]] = np.full_like(inclusion_cells[i], 1.0, dtype=dlfx.default_scalar_type)
eps_y.x.scatter_forward()

Gc = dlfx.fem.Function(DG0) #critical energy release rate
Gc.x.array[matrix_cells] = np.full_like(matrix_cells, Gc_m, dtype=dlfx.default_scalar_type)
for i in range(n_i):
    Gc.x.array[inclusion_cells[i]] = np.full_like(inclusion_cells[i], Gc_i[i], dtype=dlfx.default_scalar_type)
Gc.x.scatter_forward()

b_RO = dlfx.fem.Function(DG0) #R-O parameter b
b_RO.x.array[matrix_cells] = np.full_like(matrix_cells, b_RO_m_base, dtype=dlfx.default_scalar_type)
for i in range(n_i):
    b_RO.x.array[inclusion_cells[i]] = np.full_like(inclusion_cells[i], 1.0, dtype=dlfx.default_scalar_type)
b_RO.x.scatter_forward()

r_RO = dlfx.fem.Function(DG0) #R-O parameter r
r_RO.x.array[matrix_cells] = np.full_like(matrix_cells, r_RO_m_base, dtype=dlfx.default_scalar_type)
for i in range(n_i):
    r_RO.x.array[inclusion_cells[i]] = np.full_like(inclusion_cells[i], 1.0, dtype=dlfx.default_scalar_type)
r_RO.x.scatter_forward()

#Stress and strain measures
eps = mm.eps_lin
trace_eps = mm.trace_eps_lin
trace_eps_neg = mm.trace_eps_lin_neg
trace_eps_pos = mm.trace_eps_lin_pos
trace_eps_neg_pp = mm.trace_eps_lin_neg_pp
trace_eps_pos_pp = mm.trace_eps_lin_pos_pp
epsD = mm.epsD_lin

#Matrix
material_model = mm.RambergOsgoodIso(dim = ModelDim, E = Emod, G = Gmod, K = Kmod, nu = nu, b = b_RO, r = r_RO, eps_y = eps_y)
eqeps = material_model.eqeps
sig_D_undegraded = material_model.sig_D
sig_V_undegraded = material_model.sig_V
sig_V_neg = material_model.sig_V_neg
sig_V_pos_undegraded = material_model.sig_V_pos
sig_undegraded = material_model.sig
sig_V_scalar_undegraded = material_model.sig_V_scalar
sig_V_pos_scalar_undegraded = material_model.sig_V_pos_scalar
sig_V_neg_scalar = material_model.sig_V_neg_scalar

def sig_V_pos_degraded(u, s):
    return degrad(s, eta)*sig_V_pos_undegraded(u)

def sig_D_degraded(u, s):
    return degrad(s, eta)*sig_D_undegraded(u)

def sig_V_degraded(u, s):
    return sig_V_neg(u) + degrad(s, eta)*sig_V_pos_undegraded(u)

def sig_V_scalar_degraded(u, s):
    return sig_V_neg_scalar(u) + degrad(s, eta)*sig_V_pos_scalar_undegraded(u)

def sig_degraded(u, s): #Total stress (modified by s)
    return sig_V_neg(u) + degrad(s=s, eta=eta, beta=0.2)*(sig_V_pos_undegraded(u) + sig_D_undegraded(u))

#Energy densities
# def psi_s(s): #Fracture surface energy density
#     return  Gc * (((1 - s) ** 2) / (4 * epsilon) + epsilon * (ufl.dot(ufl.grad(s), ufl.grad(s))))

# crack_driving_energy_old = dlfx.fem.Function(DG0)
# crack_driving_energy_old.x.array[:] = np.zeros_like(crack_driving_energy_old.x.array)
# crack_driving_energy_old.x.scatter_forward()

psi_el_undegraded_old = dlfx.fem.Function(DG0)
psi_el_undegraded_old.x.array[:] = np.zeros_like(psi_el_undegraded_old.x.array)
psi_el_undegraded_old.x.scatter_forward()

psi_el_V_undegraded_old = dlfx.fem.Function(DG0)
psi_el_V_undegraded_old.x.array[:] = np.zeros_like(psi_el_V_undegraded_old.x.array)
psi_el_V_undegraded_old.x.scatter_forward()

psi_el_V_pos_undegraded_old = dlfx.fem.Function(DG0)
psi_el_V_pos_undegraded_old.x.array[:] = np.zeros_like(psi_el_V_pos_undegraded_old.x.array)
psi_el_V_pos_undegraded_old.x.scatter_forward()

psi_el_D_undegraded_old = dlfx.fem.Function(DG0)
psi_el_D_undegraded_old.x.array[:] = np.zeros_like(psi_el_D_undegraded_old.x.array)
psi_el_D_undegraded_old.x.scatter_forward()

psi_el_degraded_old = dlfx.fem.Function(DG0)
psi_el_degraded_old.x.array[:] = np.zeros_like(psi_el_degraded_old.x.array)
psi_el_degraded_old.x.scatter_forward()

if Incremental:
    psi_el_D_undegraded = material_model.psi_el_D_incremental #TODO
    psi_el_V_undegraded = material_model.psi_el_V_incremental
    psi_el_V_pos_undegraded = material_model.psi_el_V_pos_incremental
    psi_el_undegraded = material_model.psi_el_incremental
else:
    psi_el_D_undegraded = material_model.psi_el_D
    psi_el_V_undegraded = material_model.psi_el_V
    psi_el_V_pos_undegraded = material_model.psi_el_V_pos
    psi_el_undegraded = material_model.psi_el

def psi_el_V_pos_degraded(u, u_old, s, eta, sig_V_pos_undegraded):
    return degrad(s, eta)*psi_el_V_pos_undegraded(u, u_old, psi_el_V_pos_undegraded_old, sig_V_pos_undegraded)

def psi_el_degraded(u, u_old, s, eta, sig_D_undegraded, sig_V_pos_undegraded, sig_V_neg):
    return psi_el_degraded_old + degrad(s, eta)*(mm.delta_psi_el_D(u, u_old, sig_D_undegraded) + mm.delta_psi_el_V_pos(u, u_old, sig_V_pos_undegraded)) + mm.delta_psi_el_V_neg(u, u_old, sig_V_neg)

# def crack_driving_energy(u, u_old, crack_driving_energy_old):
#     return crack_driving_energy_old + mm.delta_psi_el_V_pos(u, u_old, sig_V_pos_undegraded) + mm.delta_psi_el_D(u, u_old, sig_D_undegraded)




##BCs
# Identify points for Dirichlet BCs
domain.topology.create_connectivity(0, domain.topology.dim) #0->3 / 2
all_points = dlfx.mesh.locate_entities(domain, 0, lambda x: np.full(np.shape(x)[1], True))
u_dofs_points = dlfx.fem.locate_dofs_topological(M.sub(0), 0, all_points)
s_dofs_points = dlfx.fem.locate_dofs_topological(M.sub(1), 0, all_points)
#print('len(s_dofs_points) = ', len(s_dofs_points))

# Identify facets for Dirichlet BCs
domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim) #dim -> dim
domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim) #2->3  1->2
w_old.x.array[s_dofs_points] = np.ones_like(w_old.x.array[s_dofs_points])
w_old.x.scatter_forward()

all_facets = dlfx.mesh.locate_entities(domain, 2, lambda x: np.full(np.shape(x)[1], True))
s_dofs_facets = dlfx.fem.locate_dofs_topological(M.sub(1), 2, all_facets)
u_dofs_facets = dlfx.fem.locate_dofs_topological(M.sub(0), 2, all_facets)

right_facets = dlfx.mesh.locate_entities_boundary(domain, fdim, right)
left_facets = dlfx.mesh.locate_entities_boundary(domain, fdim, left)
bottom_facets = dlfx.mesh.locate_entities_boundary(domain, fdim, bottom)
top_facets = dlfx.mesh.locate_entities_boundary(domain, fdim, top)
front_facets = dlfx.mesh.locate_entities_boundary(domain, fdim, front)
back_facets = dlfx.mesh.locate_entities_boundary(domain, fdim, back)

marked_facets = np.hstack([left_facets, right_facets, top_facets, bottom_facets, front_facets, back_facets])
marked_values = np.hstack([np.full_like(left_facets, 1), np.full_like(right_facets, 2), np.full_like(top_facets, 3), np.full_like(bottom_facets, 4), np.full_like(front_facets, 5), np.full_like(back_facets, 6)])
sorted_facets = np.argsort(marked_facets)
facet_tags = dlfx.mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

# Displacement Boundary Conditions
tension_x_max_base = dlfx.default_scalar_type(0.005)
tension_y_max_base = dlfx.default_scalar_type(0.005)
tension_z_max_base = dlfx.default_scalar_type(0.005)

compression_x_max_base = dlfx.default_scalar_type(0.1)
compression_y_max_base = dlfx.default_scalar_type(0.1)
compression_z_max_base = dlfx.default_scalar_type(0.1) 

tension_x_max = dimensionality_displacement_factor * tension_x_max_base
tension_y_max = dimensionality_displacement_factor * tension_y_max_base
tension_z_max = dimensionality_displacement_factor * tension_z_max_base

compression_x_max = dimensionality_displacement_factor * compression_x_max_base
compression_y_max = dimensionality_displacement_factor * compression_y_max_base
compression_z_max = dimensionality_displacement_factor * compression_z_max_base

tension_right = dlfx.fem.Constant(domain, tension_x_max)
tension_left = dlfx.fem.Constant(domain, -tension_x_max)
tension_top = dlfx.fem.Constant(domain, tension_y_max)
tension_bottom = dlfx.fem.Constant(domain, -tension_y_max)

# Identify DOFs for Dirichlet BCs
x_right_dofs = dlfx.fem.locate_dofs_topological(M.sub(0).sub(0), fdim, right_facets)
x_left_dofs = dlfx.fem.locate_dofs_topological(M.sub(0).sub(0), fdim, left_facets)
x_bottom_dofs = dlfx.fem.locate_dofs_topological(M.sub(0).sub(0), fdim, bottom_facets)
x_top_dofs = dlfx.fem.locate_dofs_topological(M.sub(0).sub(0), fdim, top_facets)
x_front_dofs = dlfx.fem.locate_dofs_topological(M.sub(0).sub(0), fdim, front_facets)
x_back_dofs = dlfx.fem.locate_dofs_topological(M.sub(0).sub(0), fdim, back_facets)

y_right_dofs = dlfx.fem.locate_dofs_topological(M.sub(0).sub(1), fdim, right_facets)
y_left_dofs = dlfx.fem.locate_dofs_topological(M.sub(0).sub(1), fdim, left_facets)
y_bottom_dofs = dlfx.fem.locate_dofs_topological(M.sub(0).sub(1), fdim, bottom_facets)
y_top_dofs = dlfx.fem.locate_dofs_topological(M.sub(0).sub(1), fdim, top_facets)
y_front_dofs = dlfx.fem.locate_dofs_topological(M.sub(0).sub(1), fdim, front_facets)
y_back_dofs = dlfx.fem.locate_dofs_topological(M.sub(0).sub(1), fdim, back_facets)

z_right_dofs = dlfx.fem.locate_dofs_topological(M.sub(0).sub(2), fdim, right_facets)
z_left_dofs = dlfx.fem.locate_dofs_topological(M.sub(0).sub(2), fdim, left_facets)
z_bottom_dofs = dlfx.fem.locate_dofs_topological(M.sub(0).sub(2), fdim, bottom_facets)
z_top_dofs = dlfx.fem.locate_dofs_topological(M.sub(0).sub(2), fdim, top_facets)
z_front_dofs = dlfx.fem.locate_dofs_topological(M.sub(0).sub(2), fdim, front_facets)
z_back_dofs = dlfx.fem.locate_dofs_topological(M.sub(0).sub(2), fdim, back_facets)

#Dirichlet BCs
s_ini_array = w_old.x.array[s_dofs_points] #initial s array
s_0_ini_indices = np.where(np.isclose(s_ini_array, 0.0)) #identify points with s <= s0_tol

s_0_ini_dofs = s_dofs_points[s_0_ini_indices]
s_0_old_dofs =s_0_ini_dofs #expression used in time loop regardless of initial crack
s_0_ini_bc = dlfx.fem.dirichletbc(dlfx.default_scalar_type(0.0), s_0_ini_dofs, M.sub(1))

fixed_x_right_bc = dlfx.fem.dirichletbc(0.0, x_right_dofs, M.sub(0).sub(0))
fixed_x_left_bc = dlfx.fem.dirichletbc(0.0, x_left_dofs, M.sub(0).sub(0))
fixed_x_front_bc = dlfx.fem.dirichletbc(0.0, x_front_dofs, M.sub(0).sub(0))
fixed_x_back_bc = dlfx.fem.dirichletbc(0.0, x_back_dofs, M.sub(0).sub(0))
fixed_x_bottom_bc = dlfx.fem.dirichletbc(0.0, x_bottom_dofs, M.sub(0).sub(0))
fixed_x_top_bc = dlfx.fem.dirichletbc(0.0, x_top_dofs, M.sub(0).sub(0))

fixed_y_right_bc = dlfx.fem.dirichletbc(0.0, y_right_dofs, M.sub(0).sub(1))
fixed_y_left_bc = dlfx.fem.dirichletbc(0.0, y_left_dofs, M.sub(0).sub(1))
fixed_y_front_bc = dlfx.fem.dirichletbc(0.0, y_front_dofs, M.sub(0).sub(1))
fixed_y_back_bc = dlfx.fem.dirichletbc(0.0, y_back_dofs, M.sub(0).sub(1))
fixed_y_bottom_bc = dlfx.fem.dirichletbc(0.0, y_bottom_dofs, M.sub(0).sub(1))
fixed_y_top_bc = dlfx.fem.dirichletbc(0.0, y_top_dofs, M.sub(0).sub(1))

fixed_z_right_bc = dlfx.fem.dirichletbc(0.0, z_right_dofs, M.sub(0).sub(2))
fixed_z_left_bc = dlfx.fem.dirichletbc(0.0, z_left_dofs, M.sub(0).sub(2))
fixed_z_front_bc = dlfx.fem.dirichletbc(0.0, z_front_dofs, M.sub(0).sub(2))
fixed_z_back_bc = dlfx.fem.dirichletbc(0.0, z_back_dofs, M.sub(0).sub(2))
fixed_z_bottom_bc = dlfx.fem.dirichletbc(0.0, z_bottom_dofs, M.sub(0).sub(2))
fixed_z_top_bc = dlfx.fem.dirichletbc(0.0, z_top_dofs, M.sub(0).sub(2))
   
tension_right_bc = dlfx.fem.dirichletbc(tension_right, x_right_dofs, M.sub(0).sub(0))
tension_left_bc = dlfx.fem.dirichletbc(tension_left, x_left_dofs, M.sub(0).sub(0))
tension_top_bc = dlfx.fem.dirichletbc(tension_top, y_top_dofs, M.sub(0).sub(1))
tension_bottom_bc = dlfx.fem.dirichletbc(tension_bottom, y_bottom_dofs, M.sub(0).sub(1))

uni_tension_x_bcs = [fixed_z_back_bc, fixed_y_bottom_bc, fixed_x_left_bc, tension_right_bc] # simple tension
bi_tension_x_bcs = [fixed_z_back_bc, fixed_y_bottom_bc, tension_left_bc, tension_right_bc] # bilateral tension
bi_tension_y_bcs = [fixed_z_back_bc, fixed_x_left_bc, tension_top_bc, tension_bottom_bc] # bilateral tension
simple_support_bcs = [fixed_x_left_bc, fixed_y_bottom_bc, fixed_z_back_bc]
simple_support_sym_bcs = [fixed_x_left_bc, fixed_x_right_bc, fixed_z_back_bc, fixed_z_front_bc, fixed_y_bottom_bc]
uni_tension_y_bcs = [fixed_z_back_bc, fixed_x_left_bc, fixed_y_bottom_bc, tension_top_bc] # simple tension
uni_tension_y_plane_strain_bcs = [fixed_x_left_bc, fixed_y_bottom_bc, fixed_z_back_bc, fixed_z_front_bc, tension_top_bc] # simple tension plane strain
bi_tension_y_plane_strain_bcs = [fixed_x_left_bc, fixed_z_back_bc, fixed_z_front_bc, tension_top_bc, tension_bottom_bc] # simple tension plane strain

bcs =  bi_tension_y_plane_strain_bcs #simple_support_bcs

#Residuum
dxmod = ufl.Measure("dx", domain=domain, subdomain_data=cell_markers, metadata={"quadrature_degree":2})
res_u = ufl.inner(sig_degraded(u, s), eps(du)) * dxmod
res_s = ((1/mob)*((s - s_old)/dt * ds) + ufl.dot(ufl.grad(ds), 2*Gc*epsilon*ufl.grad(s)) + ds*(
            diff_degrad(s)*crack_driving_energy(u=u, u_old=u_old, Emod=Emod, nu=nu, eps=eps, sig_V_pos_undegraded=sig_V_pos_undegraded, sig_D_undegraded=sig_D_undegraded, sig_undegraded=sig_undegraded, psi_el_V_pos_undegraded=psi_el_V_pos_undegraded, psi_el_D_undegraded=psi_el_D_undegraded, psi_el_V_pos_old=psi_el_V_pos_undegraded_old, psi_el_D_old=psi_el_D_undegraded_old, delta_psi_el_V_pos_undegraded=mm.delta_psi_el_V_pos, delta_psi_el_D_undegraded=mm.delta_psi_el_D, crack_driving_energy_old=crack_driving_energy_old) - (Gc/(2*epsilon))*(1 - s))) * dxmod
res = res_u + res_s

comm.barrier()

# Configure nonlinear solver
nl_problem = dlfx.fem.petsc.NonlinearProblem(res, w, bcs) #, jit_options={"cache_dir": cache_dir, "cffi_extra_compile_args": ["-Ofast"]})
nl_solver = dlfx.nls.petsc.NewtonSolver(comm, nl_problem)

nl_solver.rtol = 1.0e-8#1.0e-8
nl_solver.atol = 5.0e-8#1.0e-8
nl_solver.max_it = max_iters
#nl_solver.convergence_criterion = "incremental"
#nl_solver.damping = 0.005

# Configure linear solver
lin_solver = nl_solver.krylov_solver

opts = PETSc.Options()
option_prefix = lin_solver.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
# opts[
#     {
#         "type": "command",
#         "details": {
#             "key": "workbench.action.viewPetscSolverConfig"
#         }
#     }
# ]
lin_solver.setFromOptions()
# input()
with dlfx.io.XDMFFile(comm, XdmfMeshFile, "w") as xdmfout_mesh:
        xdmfout_mesh.write_mesh(domain)
        cell_markers.name = "Mesh Tags"
        facet_tags.name = "Facet Tags"
        xdmfout_mesh.write_meshtags(cell_markers, domain.geometry, geometry_xpath = '/Xdmf/Domain/Grid/Geometry')
        xdmfout_mesh.write_meshtags(facet_tags, domain.geometry, geometry_xpath = '/Xdmf/Domain/Grid/Geometry')
        xdmfout_mesh.close()


with dlfx.io.XDMFFile(comm, AuxFile, "w") as xdmfout_aux:
        xdmfout_aux.write_mesh(domain)
        Emod.name = "E"
        xdmfout_aux.write_function(Emod)
        Kmod.name = "K"
        xdmfout_aux.write_function(Kmod)
        Gmod.name = "G"
        xdmfout_aux.write_function(Gmod)
        nu.name = "nu"
        xdmfout_aux.write_function(nu)
        Gc.name = "Gc"
        xdmfout_aux.write_function(Gc)
        if args.mat_mod == 'RO':
            b_RO.name = "b_RO"
            xdmfout_aux.write_function(b_RO)
            r_RO.name = "r_RO"
            xdmfout_aux.write_function(r_RO)
        xdmfout_aux.close()

with dlfx.io.XDMFFile(comm, file, "w") as xdmfout:
        xdmfout.write_mesh(domain)
        xdmfout.close()
                  
# Postprocessing fields
u_lin = dlfx.fem.Function(V_lin)
s_lin = dlfx.fem.Function(S_lin)
strain = dlfx.fem.Function(T)
sigma = dlfx.fem.Function(T)

eps_eq = dlfx.fem.Function(DG0)
eps_eq_m = dlfx.fem.Function(DG0)

crack_driving_energy_pp = dlfx.fem.Function(DG0)
psi_el_D_undegraded_pp = dlfx.fem.Function(DG0)
psi_el_V_undegraded_pp = dlfx.fem.Function(DG0)
psi_el_V_pos_undegraded_pp = dlfx.fem.Function(DG0)
psi_el_undegraded_pp = dlfx.fem.Function(DG0)
psi_el_degraded_pp = dlfx.fem.Function(DG0)

sigma_V = dlfx.fem.Function(T)
sigma_V_scalar = dlfx.fem.Function(DG0)
sigma_V_neg = dlfx.fem.Function(T)
sigma_V_pos = dlfx.fem.Function(T)

sigma_D = dlfx.fem.Function(T)

eps_V = dlfx.fem.Function(DG0)
eps_V_pos = dlfx.fem.Function(DG0)
eps_V_neg = dlfx.fem.Function(DG0)

#Start timer
timer = dlfx.common.Timer()
timer.start()

while t.value < Tend+0.5*dt.value:
    if dt.value < 1e-20:
        break
    
    #update time step
    t.value += dt.value
    if t.value < 0.0:
        if rank == 0:
            print("t < 0.0!")
            sys.stdout.flush()
        break

    # reset solver status
    restart_solution = False
    iters = int()
    converged = False
    if rank==0:
        print('')
        print('t = ', t.value)
        print('')
        print('dt=', dt.value)
        sys.stdout.flush()
        
    tension_right.value = t.value*tension_x_max
    tension_left.value = -t.value*tension_x_max
    tension_top.value = t.value*tension_y_max
    tension_bottom.value = -t.value*tension_y_max
    
    try:    
        iters, conv = nl_solver.solve(w)
    except RuntimeError:
        dt.value = dt_scale_down * dt.value
        restart_solution = True
        if rank == 0:
            print('Decreasing dt to {0:.4e}'.format(dt.value))
            sys.stdout.flush()
            #nl_solver.damping = 0.5
    w.x.scatter_forward()
    comm.barrier()

    if not restart_solution:
        if rank==0:
            print('')
            print('t = ', t.value)
            print('')
            print('iters, conv:', iters, conv)
            print('')
            sys.stdout.flush()
                
        u = w.sub(0).collapse()
        s = w.sub(1).collapse()
        u_old = w_old.sub(0).collapse()

        wrestart.x.array[:] = w.x.array[:]
        wrestart.x.scatter_forward()
        trestart.value = t.value

        #update s_0 bcs
        s_new_array = w.x.array[s_dofs_points] #s array of current time step
        s_0_current_indices = np.where(np.less_equal(s_new_array, s0_tol)) #identify points with s <= s0_tol in current time step
        s_0_current_dofs = s_dofs_points[s_0_current_indices] #array of points with s <= s0_tol in current time steps, includes points already considered in previous time step
        s_0_new_dofs = np.setdiff1d(s_0_current_dofs, s_0_old_dofs) #array of points with s <= s0_tol in current time steps, excludes points already considered in previous time step
        
        #print('current crack dofs on rank ', rank, ' at time ', t.value, ' = ', s_0_current_dofs)
        print('---')
        print('number of current crack dofs on rank ', rank, ' at time ', t.value, ' = ', len(s_0_current_dofs))
        sys.stdout.flush()

        if len(s_0_new_dofs) == 0:
            #output_step = args.output_step
            print('---')
            print('no new crack dofs on rank ', rank, ' at time ', t.value)
            print('---')
            sys.stdout.flush()
        else:
            #output_step = output_step_smaller
            #s_zero_bc = dlfx.fem.dirichletbc(dlfx.fem.Constant(domain, 0.0), s_zero_dofs, M.sub(1))
            s_0_new_bc = dlfx.fem.dirichletbc(dlfx.default_scalar_type(0.0), s_0_new_dofs, M.sub(1))
            bcs.append(s_0_new_bc)
            print('---')
            #print('-> new crack dofs on rank ', rank, ' at time ', t.value, ' = ', s_0_new_dofs)
            print('number of new crack dofs on rank ', rank, ' at time ', t.value, ' = ', len(s_0_new_dofs))
            print('---')
            sys.stdout.flush()
                
        s_0_old_dofs = s_0_current_dofs #update s_0_old_dofs for next time step

        #set up postprocessing fields
        u_lin_expr = dlfx.fem.Expression(u, V_lin.element.interpolation_points())
        u_lin.interpolate(u_lin_expr)
        u_lin.x.scatter_forward()

        s_lin_expr = dlfx.fem.Expression(s, S_lin.element.interpolation_points())
        s_lin.interpolate(s_lin_expr)
        s_lin.x.scatter_forward()
                      
        strain_expr = dlfx.fem.Expression(eps(u), T.element.interpolation_points())
        strain.interpolate(strain_expr)
        strain.x.scatter_forward()

        eps_V_expr = dlfx.fem.Expression(trace_eps(u), DG0.element.interpolation_points())
        eps_V.interpolate(eps_V_expr)
        eps_V.x.scatter_forward()

        eps_V_pos_expr = dlfx.fem.Expression(trace_eps_pos(u), DG0.element.interpolation_points())
        eps_V_pos.interpolate(eps_V_pos_expr)
        eps_V_pos.x.scatter_forward()

        eps_V_neg_expr = dlfx.fem.Expression(trace_eps_neg(u), DG0.element.interpolation_points())
        eps_V_neg.interpolate(eps_V_neg_expr)
        eps_V_neg.x.scatter_forward()

        eps_eq_expr = dlfx.fem.Expression(eqeps(u), DG0.element.interpolation_points())
        eps_eq.interpolate(eps_eq_expr)
        eps_eq.x.scatter_forward()

        sigma_expr = dlfx.fem.Expression(sig_degraded(u, s), T.element.interpolation_points())
        sigma.interpolate(sigma_expr)
        sigma.x.scatter_forward()

        sigma_V_scalar_expr = dlfx.fem.Expression(sig_V_scalar_degraded(u, s), DG0.element.interpolation_points())
        sigma_V_scalar.interpolate(sigma_V_scalar_expr)
        sigma_V_scalar.x.scatter_forward()

        sigma_D_expr = dlfx.fem.Expression(sig_D_degraded(u, s), T.element.interpolation_points())
        sigma_D.interpolate(sigma_D_expr)
        sigma_D.x.scatter_forward()
        
        crack_driving_energy_pp_expr = dlfx.fem.Expression(crack_driving_energy(u=u, u_old=u_old, Emod=Emod, nu=nu, eps=eps, sig_V_pos_undegraded=sig_V_pos_undegraded, sig_D_undegraded=sig_D_undegraded, sig_undegraded=sig_undegraded, psi_el_V_pos_undegraded=psi_el_V_pos_undegraded, psi_el_D_undegraded=psi_el_D_undegraded, psi_el_V_pos_old=psi_el_V_pos_undegraded_old, psi_el_D_old=psi_el_D_undegraded_old, delta_psi_el_V_pos_undegraded=mm.delta_psi_el_V_pos, delta_psi_el_D_undegraded=mm.delta_psi_el_D, crack_driving_energy_old=crack_driving_energy_old), DG0.element.interpolation_points())
        crack_driving_energy_pp.interpolate(crack_driving_energy_pp_expr)
        crack_driving_energy_pp.x.scatter_forward()

        # psi_el_V_undegraded_pp_expr = dlfx.fem.Expression(psi_el_V_undegraded(u, u_old, psi_el_V_undegraded_old, sig_V_pos_undegraded, sig_V_neg), DG0.element.interpolation_points())
        # psi_el_V_undegraded_pp.interpolate(psi_el_V_undegraded_pp_expr)
        # psi_el_V_undegraded_pp.x.scatter_forward()

        psi_el_V_pos_undegraded_pp_expr = dlfx.fem.Expression(psi_el_V_pos_undegraded(u, u_old, psi_el_V_pos_undegraded_old, sig_V_pos_undegraded), DG0.element.interpolation_points())
        psi_el_V_pos_undegraded_pp.interpolate(psi_el_V_pos_undegraded_pp_expr) 
        psi_el_V_pos_undegraded_pp.x.scatter_forward()

        psi_el_D_undegraded_pp_expr = dlfx.fem.Expression(psi_el_D_undegraded(u, u_old, psi_el_D_undegraded_old, sig_D_undegraded), DG0.element.interpolation_points())
        psi_el_D_undegraded_pp.interpolate(psi_el_D_undegraded_pp_expr)
        psi_el_D_undegraded_pp.x.scatter_forward()

        # psi_el_undegraded_pp_expr = dlfx.fem.Expression(psi_el_undegraded(u, u_old, psi_el_D_undegraded_old, sig_D_undegraded, sig_V_pos_undegraded, sig_V_neg), DG0.element.interpolation_points())
        # psi_el_undegraded_pp.interpolate(psi_el_undegraded_pp_expr)
        # psi_el_undegraded_pp.x.scatter_forward()

        psi_el_degraded_pp_expr = dlfx.fem.Expression(psi_el_degraded(u, u_old, s, eta, sig_D_undegraded, sig_V_pos_undegraded, sig_V_neg), DG0.element.interpolation_points())
        psi_el_degraded_pp.interpolate(psi_el_degraded_pp_expr)
        psi_el_degraded_pp.x.scatter_forward()


        with dlfx.io.XDMFFile(comm, file, "a") as xdmfout:
            u_lin.name = "displacement u"
            xdmfout.write_function(u_lin, t.value)

            s_lin.name = "s"
            xdmfout.write_function(s_lin, t.value)
            
            # strain.name = "strain epsilon"
            # xdmfout.write_function(strain, t.value)

            # eps_V.name = "volumetric strain"
            # xdmfout.write_function(eps_V, t.value)

            # eps_eq.name = "equivalent strain"
            # xdmfout.write_function(eps_eq, t.value)
            
            sigma.name = "sigma"
            xdmfout.write_function(sigma, t.value)
            
            # sigma_V_scalar.name = "sig_V sc."
            # xdmfout.write_function(sigma_V_scalar, t.value)

            # sigma_D.name = "sig_D"
            # xdmfout.write_function(sigma_D, t.value)

            # crack_driving_energy_pp.name = "crack driving energy density"
            # xdmfout.write_function(crack_driving_energy_pp, t.value)
        
        
        # psi_el_undegraded_old.x.array[:] = psi_el_undegraded_pp.x.array[:]
        # psi_el_undegraded_old.x.scatter_forward()

        # psi_el_undegraded_old.x.array[:] = psi_el_undegraded_pp.x.array[:]
        # psi_el_undegraded_old.x.scatter_forward()
        
        crack_driving_energy_old.x.array[:] = crack_driving_energy_pp.x.array[:]
        crack_driving_energy_old.x.scatter_forward()

        psi_el_V_pos_undegraded_old.x.array[:] = psi_el_V_pos_undegraded_pp.x.array[:]
        psi_el_V_pos_undegraded_old.x.scatter_forward()

        psi_el_D_undegraded_old.x.array[:] = psi_el_D_undegraded_pp.x.array[:]
        psi_el_D_undegraded_old.x.scatter_forward()

        # psi_el_V_undegraded_old.x.array[:] = psi_el_V_undegraded_pp.x.array[:]
        # psi_el_V_undegraded_old.x.scatter_forward()
            
        w_old.x.array[:] = w.x.array[:]
        w_old.x.scatter_forward()    
    
        # modify dt if number of iterations is out of specified range
        max_iters_glob = comm.allreduce(np.max(iters), op=MPI.MAX)
        if max_iters_glob < min_iters:
            dt.value = min(dt_scale_up * dt.value, max_dt)
            if rank == 0:
                print('Increasing dt to {0:.4e}'.format(dt.value))
                sys.stdout.flush()

        if max_iters_glob > max_iters:
            dt.value = dt_scale_down * dt.value
            if rank == 0:
                print('Decreasing dt to {0:.4e}'.format(dt.value))
                sys.stdout.flush()

        comm.barrier()

    else:
        
        t.value = trestart.value
        w.x.array[:] = wrestart.x.array[:]
        w.x.scatter_forward()
        
    comm.Barrier()

    # report runtime to screen
if rank == 0:
    print('')
    print('-----------------------------')
    print('elapsed time:  ', timer.elapsed())
    print('')
    sys.stdout.flush()