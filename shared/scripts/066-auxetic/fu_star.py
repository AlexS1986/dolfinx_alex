
#flat tip with corner star in the middle 

import os
import dolfinx as dlfx
import gmsh
import numpy as np
import ufl
import basix.ufl
from dolfinx import io
from mpi4py import MPI
from petsc4py import PETSc
comm = MPI.COMM_WORLD 
from dolfinx.fem import petsc
from dolfinx.nls.petsc import NewtonSolver

# Set output directory to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
msh_file = os.path.join(script_dir, "fu_star.msh")
xdmf_file = os.path.join(script_dir, "fu_star.xdmf")


#material properties - isotropic linear elastic material 
E = 10.0 #Young's modulus  ---- or shear modulus G
nu = 0.3 #Poisson's ratio
G = E / (2*(1 + nu)) #shear modulus
K = E / (2*(1 - nu)) #bulk modulus


gmsh.initialize()
gmsh.model.add("fu_star")

#small star
hole_scale = 0.45
y_inner = gmsh.model.occ.addPoint(0, 0.2 * hole_scale, 0) 
y_outer = gmsh.model.occ.addPoint(0.5 * hole_scale, 0.5 * hole_scale, 0)
x_inner = gmsh.model.occ.addPoint(0.2 * hole_scale, 0, 0)
origin = gmsh.model.occ.addPoint(0, 0, 0)


sl1 = gmsh.model.occ.addLine(origin, y_inner)
sl2 = gmsh.model.occ.addLine(y_inner, y_outer)
sl3 = gmsh.model.occ.addLine(y_outer, x_inner)
sl4 = gmsh.model.occ.addLine(x_inner, origin)

loop = gmsh.model.occ.addCurveLoop([sl1, sl2, sl3, sl4])
quarter = gmsh.model.occ.addPlaneSurface([loop])

# Mirror to create full small star
left = gmsh.model.occ.copy([(2, quarter)])
gmsh.model.occ.mirror(left, 1, 0, 0, 0)
half = gmsh.model.occ.copy([(2, quarter), left[0]])
gmsh.model.occ.mirror(half, 0, 1, 0, 0)

small_star_list = [(2, quarter), left[0], half[0], half[1]]

small_star_fused, _ = gmsh.model.occ.fuse([small_star_list[0]], small_star_list[1:], removeObject=True, removeTool=True)

#large star
t = 0.0625 #beam thickness

#points/parameters
hx1, vy2 = -0.5, 0.5 
hx2, vy1 = -0.2, 0.2
s1_x, s1_y = -0.275, 0.225 
s2_x, s2_y = -0.225, 0.275 

#top left
h1 = gmsh.model.occ.addPoint(hx1, t, 0)
h2 = gmsh.model.occ.addPoint(hx2, t, 0)
v1 = gmsh.model.occ.addPoint(-t, vy1, 0)
v2 = gmsh.model.occ.addPoint(-t, vy2, 0)

slant1 = gmsh.model.occ.addPoint(s1_x, s1_y, 0)
slant2 = gmsh.model.occ.addPoint(s2_x, s2_y, 0)

L1 = gmsh.model.occ.addLine(h1, h2)
L2 = gmsh.model.occ.addLine(h2, slant1)
L3 = gmsh.model.occ.addLine(slant1, slant2)
L4 = gmsh.model.occ.addLine(slant2, v1)
L5 = gmsh.model.occ.addLine(v1, v2)

#top right
h1_m = gmsh.model.occ.addPoint(-hx1, t, 0)
h2_m = gmsh.model.occ.addPoint(-hx2, t, 0)
v1_m = gmsh.model.occ.addPoint(t, vy1, 0)
v2_m = gmsh.model.occ.addPoint(t, vy2, 0)

slant1_m = gmsh.model.occ.addPoint(-s1_x, s1_y, 0)
slant2_m = gmsh.model.occ.addPoint(-s2_x, s2_y, 0)

L1_m = gmsh.model.occ.addLine(h1_m, h2_m)
L2_m = gmsh.model.occ.addLine(h2_m, slant1_m)
L3_m = gmsh.model.occ.addLine(slant1_m, slant2_m)
L4_m = gmsh.model.occ.addLine(slant2_m, v1_m)
L5_m = gmsh.model.occ.addLine(v1_m, v2_m)

L6 = gmsh.model.occ.addLine(v2, v2_m) #top flat

#bottom right
h1_mr = gmsh.model.occ.addPoint(-hx1, -t, 0)
h2_mr = gmsh.model.occ.addPoint(-hx2, -t, 0)
v1_mr = gmsh.model.occ.addPoint(t, -vy1, 0)
v2_mr = gmsh.model.occ.addPoint(t, -vy2, 0)

slant1_mr = gmsh.model.occ.addPoint(-s1_x, -s1_y, 0)
slant2_mr = gmsh.model.occ.addPoint(-s2_x, -s2_y, 0)

L1_mr = gmsh.model.occ.addLine(h1_mr, h2_mr)
L2_mr = gmsh.model.occ.addLine(h2_mr, slant1_mr)
L3_mr = gmsh.model.occ.addLine(slant1_mr, slant2_mr)
L4_mr = gmsh.model.occ.addLine(slant2_mr, v1_mr)
L5_mr = gmsh.model.occ.addLine(v1_mr, v2_mr)

L9 = gmsh.model.occ.addLine(h1_m, h1_mr) #right flat

#bottom left
h1_ml = gmsh.model.occ.addPoint(hx1, -t, 0)
h2_ml = gmsh.model.occ.addPoint(hx2, -t, 0)
v1_ml = gmsh.model.occ.addPoint(-t, -vy1, 0)
v2_ml = gmsh.model.occ.addPoint(-t, -vy2, 0)

slant1_ml = gmsh.model.occ.addPoint(s1_x, -s1_y, 0)
slant2_ml = gmsh.model.occ.addPoint(s2_x, -s2_y, 0)

L1_ml = gmsh.model.occ.addLine(h1_ml, h2_ml)
L2_ml = gmsh.model.occ.addLine(h2_ml, slant1_ml)
L3_ml = gmsh.model.occ.addLine(slant1_ml, slant2_ml)
L4_ml = gmsh.model.occ.addLine(slant2_ml, v1_ml)
L5_ml = gmsh.model.occ.addLine(v1_ml, v2_ml)

L7 = gmsh.model.occ.addLine(v2_mr, v2_ml) #bottom flat
L8 = gmsh.model.occ.addLine(h1_ml, h1)    #left flat


curve = gmsh.model.occ.addCurveLoop([L1, L2, L3, L4, L5, L6, 
                                     L5_m, L4_m, L3_m, L2_m, L1_m, L9, 
                                     L1_mr, L2_mr, L3_mr, L4_mr, L5_mr, L7, 
                                     L5_ml, L4_ml, L3_ml, L2_ml, L1_ml, L8])

surface = gmsh.model.occ.addPlaneSurface([curve])


final_shape, _ = gmsh.model.occ.cut([(2, surface)], small_star_fused, removeObject=True, removeTool=True)

gmsh.model.occ.synchronize()

final_surface_tag = final_shape[0][1]
gmsh.model.addPhysicalGroup(2, [final_surface_tag], 1)
gmsh.model.mesh.generate(2)
gmsh.write(msh_file)

gmsh.finalize()

domain, cell_markers, facet_markers = io.gmshio.read_from_msh(msh_file, comm, gdim=2)

#function space
#vector elements
el = basix.ufl.element("P", domain.basix_cell(), 1, shape=(2,))
V = dlfx.fem.functionspace(domain, el)

#TENSOR valued elements
el2 = basix.ufl.element("P", domain.basix_cell(), 0, discontinuous = True, shape=(2,2))
T = dlfx.fem.functionspace(domain, el2) 

#now define actual functions 
u = dlfx.fem.Function(V)
du = ufl.TestFunction(V) #test function of displacement

sigma = dlfx.fem.Function(T) #stress function


#boundary conditions - left & right defined in case of errors for checking

def bottom(x):
    return np.isclose(x[1], -0.5) 
def top(x):
    return np.isclose(x[1], 0.5)
def left(x):
    return np.isclose(x[0], -0.5) 
def right(x):   
    return np.isclose(x[0], 0.5) 


#to see tension and compression
top_facets = dlfx.mesh.locate_entities_boundary(domain, 1, top)
bottom_facets = dlfx.mesh.locate_entities_boundary(domain, 1, bottom)
left_facets = dlfx.mesh.locate_entities_boundary(domain, 1, left)
right_facets = dlfx.mesh.locate_entities_boundary(domain, 1, right)

y_top_dofs = dlfx.fem.locate_dofs_topological(V.sub(1), 1, top_facets)
y_bottom_dofs = dlfx.fem.locate_dofs_topological(V.sub(1), 1, bottom_facets)

x_top_dofs = dlfx.fem.locate_dofs_topological(V.sub(0), 1, top_facets)
x_bottom_dofs = dlfx.fem.locate_dofs_topological(V.sub(0), 1, bottom_facets)
x_left_dofs = dlfx.fem.locate_dofs_topological(V.sub(0), 1, left_facets)
x_right_dofs = dlfx.fem.locate_dofs_topological(V.sub(0), 1, right_facets)

tension_top_bc = dlfx.fem.dirichletbc(0.1, y_top_dofs, V.sub(1))
tension_bottom_bc = dlfx.fem.dirichletbc(-0.1, y_bottom_dofs, V.sub(1))

#fixed
fixed_top = dlfx.fem.dirichletbc(0.0, x_top_dofs, V.sub(1))
fixed_bottom = dlfx.fem.dirichletbc(0.0, x_bottom_dofs, V.sub(1))
fixed_left = dlfx.fem.dirichletbc(0.0, x_left_dofs, V.sub(0))
fixed_right = dlfx.fem.dirichletbc(0.0, x_right_dofs, V.sub(0))

bcs =  [tension_top_bc, tension_bottom_bc, fixed_top, fixed_bottom] #what I am solving for

#weak form

def eps(u): # Linearized strain tensor
	return 0.5 * (ufl.grad(u) + ufl.grad(u).T)

def sig_D(u): #Deviatoric stress
       return 2 * G * ufl.dev(eps(u))
    
def sig_V(u): #Volumetric stress
       return K * ufl.tr(eps(u)) * ufl.Identity(2)

def sig(u): #Total stress
        return sig_V(u) + sig_D(u)

res = ufl.inner(sig(u), eps(du))* ufl.dx #over whole domain 

# Configure nonlinear solver
petsc_options = {
    "snes_type": "newtonls",
    "snes_atol": 1e-12,
    "snes_rtol": 5e-12,
    "snes_stol": 0.0,
    "snes_monitor": None,
    "snes_error_if_not_converged": True,
    "ksp_error_if_not_converged": True,
    "ksp_type": "preonly",#"gmres",
    "ksp_rtol": 1e-10,
    "pc_type": "lu",#"hypre",
    "pc_factor_mat_solver_type": "mumps"
}

opts = PETSc.Options("thesis")
for key, value in petsc_options.items():
    opts[key] = value

problem = petsc.NonlinearProblem(res, u, bcs=bcs)
solver = NewtonSolver(comm, problem)
solver.solve(u) #solves for displacement field internally

#post-processing
sigma_expr = dlfx.fem.Expression(sig(u), T.element.interpolation_points())
sigma.interpolate(sigma_expr)
sigma.x.scatter_forward()

with dlfx.io.XDMFFile(comm, xdmf_file, "w") as xdmfout:
        xdmfout.write_mesh(domain)
        xdmfout.close()

with dlfx.io.XDMFFile(comm, xdmf_file, "a") as xdmfout:
        u.name = "displacement"
        sigma.name = "stress"
        xdmfout.write_function(u)
        xdmfout.write_function(sigma)
        xdmfout.close()
