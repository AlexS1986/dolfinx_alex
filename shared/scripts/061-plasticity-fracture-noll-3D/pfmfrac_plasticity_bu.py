import alex.linearelastic
import alex.phasefield
import dolfinx as dlfx
from mpi4py import MPI
from petsc4py import PETSc as petsc


import ufl 
import numpy as np
import os 
import sys
import math

import alex.os
import alex.boundaryconditions as bc
import alex.postprocessing as pp
import alex.solution as sol
import alex.linearelastic as le
import alex.phasefield as pf
import basix
import alex.plasticity

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
logfile_path = alex.os.logfile_full_path(script_path,script_name_without_extension)
outputfile_graph_path = alex.os.outputfile_graph_full_path(script_path,script_name_without_extension)
outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path,script_name_without_extension)
outputfile_vtk_path = alex.os.outputfile_vtk_full_path(script_path,script_name_without_extension)

# set FEniCSX log level
# dlfx.log.set_log_level(log.LogLevel.INFO)
# dlfx.log.set_output_file('xxx.log')

# set and start stopwatch
timer = dlfx.common.Timer()
timer.start()

# set MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
sys.stdout.flush()


# generate domain
#domain = dlfx.mesh.create_unit_square(comm, N, N, cell_type=dlfx.mesh.CellType.quadrilateral)

#N=16
#domain = dlfx.mesh.create_unit_cube(comm,N,N,N,cell_type=dlfx.mesh.CellType.hexahedron)
#domain = dlfx.mesh.create_unit_cube(comm,N,N,N,cell_type=dlfx.mesh.CellType.tetrahedron)


with dlfx.io.XDMFFile(comm, os.path.join(script_path,"mesh_script.xdmf"), 'r') as mesh_inp: 
    domain = mesh_inp.read_mesh(name="mesh")
    mesh_tags = mesh_inp.read_meshtags(domain,name="Cell tags")



dt = 0.0001

t_global = dlfx.fem.Constant(domain,0.0)
dt_global = dlfx.fem.Constant(domain, dt)

deg_quad = 1
# function space using mesh and degree
Ve = basix.ufl.element("P", domain.basix_cell(), deg_quad, shape=(domain.geometry.dim,)) #displacements
Te = basix.ufl.element("P", domain.basix_cell(), deg_quad, shape=())
W = dlfx.fem.FunctionSpace(domain, ufl.MixedElement([Ve, Te]))

dim = domain.topology.dim
alex.os.mpi_print('spatial dimensions: '+str(dim), rank)

x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = pp.compute_bounding_box(comm, domain)
pp.print_bounding_box(rank, x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all)

v_crack = 1.0 # const for all simulations
Tend = (x_max_all-0.0) * 2.0 / v_crack

# elastic constants
lam = dlfx.fem.Constant(domain, 1.0)
mu = dlfx.fem.Constant(domain, 1.0)

# residual stiffness
eta = dlfx.fem.Constant(domain, 0.001)
Gc = dlfx.fem.Constant(domain, 1.0)
epsilon = dlfx.fem.Constant(domain, 0.1)

Mob = dlfx.fem.Constant(domain, 1000.0)
iMob = dlfx.fem.Constant(domain, 1.0/Mob.value)

E_mod = alex.linearelastic.get_emod(lam.value, mu.value)
K1 = dlfx.fem.Constant(domain, 1.5 * math.sqrt(Gc.value*E_mod))

sig_y = dlfx.fem.Constant(domain, 1.0)
hard = dlfx.fem.Constant(domain, 0.2222222)


(
    alpha_n, alpha_tmp,

    e_p_11_n, e_p_22_n, e_p_33_n,
    e_p_12_n, e_p_13_n, e_p_23_n,

    e_p_11_n_tmp, e_p_22_n_tmp, e_p_33_n_tmp,
    e_p_12_n_tmp, e_p_13_n_tmp, e_p_23_n_tmp
) = alex.plasticity.define_internal_state_variables_basix_3D(domain, deg_quad, quad_scheme="default"
)

dx_integration_plasticity = alex.plasticity.define_custom_integration_measure_that_matches_quadrature_degree_and_scheme(domain, deg_quad, "default")
quadrature_points, cells = alex.plasticity.get_quadraturepoints_and_cells_for_inter_polation_at_gauss_points(domain, deg_quad)



# define crack by boundary
crack_tip_start_location_x = 0.2*(x_max_all-x_min_all) + x_min_all
crack_tip_start_location_y = (y_max_all + y_min_all) / 2.0
def crack(x):
    x_log = x[0] < (crack_tip_start_location_x)
    y_log = np.isclose(x[1],crack_tip_start_location_y,atol=(0.02*((y_max_all-y_min_all))))
    # x_log = x[0]< 50
    # y_log = np.isclose(x[1],200,atol=(0.02*(200)))
    return np.logical_and(y_log,x_log)


# define boundary condition on top and bottom
fdim = domain.topology.dim -1
domain.topology.create_connectivity(fdim, domain.topology.dim)

crackfacets = dlfx.mesh.locate_entities(domain, fdim, crack)
crackdofs = dlfx.fem.locate_dofs_topological(W.sub(1), fdim, crackfacets)
bccrack = dlfx.fem.dirichletbc(0.0, crackdofs, W.sub(1))


e_p_n_3D = ufl.as_tensor([[e_p_11_n, e_p_12_n, e_p_13_n], 
                          [e_p_12_n, e_p_22_n, e_p_23_n],
                          [e_p_13_n, e_p_23_n, e_p_33_n]])


phaseFieldProblem = pf.StaticPhaseFieldProblem_plasticity_noll(degradationFunction=pf.degrad_cubic,
                                                   psisurf=pf.psisurf_from_function,dx=dx_integration_plasticity, sig_y=sig_y.value, hard=hard.value,alpha_n=alpha_n,e_p_n=e_p_n_3D)


# phaseFieldProblem = pf.StaticPhaseFieldProblem3D(degradationFunction=pf.degrad_quadratic,
#                                                    psisurf=pf.psisurf)

             
# define solution, restart, trial and test space
w =  dlfx.fem.Function(W)
u,s = w.split()
wrestart =  dlfx.fem.Function(W)
wm1 =  dlfx.fem.Function(W) # trial space
dw = ufl.TestFunction(W)
ddw = ufl.TrialFunction(W)

def before_first_time_step():
    # initialize s=1 
    wm1.sub(1).x.array[:] = np.ones_like(wm1.sub(1).x.array[:])
    wrestart.x.array[:] = wm1.x.array[:]
    # prepare newton-log-file
    if rank == 0:
        sol.prepare_newton_logfile(logfile_path)
        pp.prepare_graphs_output_file(outputfile_graph_path)
    # prepare xdmf output 
    pp.write_meshoutputfile(domain, outputfile_xdmf_path, comm)
    # pp.write_meshoutputfile(domain, outputfile_vtk_path, comm)

def before_each_time_step(t,dt):
    # report solution status
    if rank == 0:
        sol.print_time_and_dt(t,dt)
      


def get_residuum_and_gateaux(delta_t: dlfx.fem.Constant):
    [Res, dResdw] = phaseFieldProblem.prep_newton(
        w=w,wm1=wm1,dw=dw,ddw=ddw,lam=lam, mu = mu,
        Gc=Gc,epsilon=epsilon, eta=eta,
        iMob=iMob, delta_t=delta_t)
    return [Res, dResdw]


# setup tracking
Se = ufl.FiniteElement("Lagrange", domain.ufl_cell(),1) 
S = dlfx.fem.FunctionSpace(domain,Se)
s_zero_for_tracking_at_nodes = dlfx.fem.Function(S)
c = dlfx.fem.Constant(domain, petsc.ScalarType(1))
sub_expr = dlfx.fem.Expression(c,S.element.interpolation_points())
s_zero_for_tracking_at_nodes.interpolate(sub_expr)


# surfing BCs
vcrack_const = dlfx.fem.Constant(domain, np.array([v_crack,0.0,0.0],dtype=dlfx.default_scalar_type))
crack_start = dlfx.fem.Constant(domain, np.array([0.0,crack_tip_start_location_y,0.0],dtype=dlfx.default_scalar_type))
w_D = dlfx.fem.Function(W) # for dirichlet BCs
xxK1 = dlfx.fem.Constant(domain, np.array([0.0,0.0,0.0],dtype=dlfx.default_scalar_type))


def compute_surf_displacement():
    x = ufl.SpatialCoordinate(domain)
    


    #xxK1 = crack_start + vcrack_const * t_global 
    dx = x[0] - xxK1[0]
    dy = x[1] - xxK1[1]
    
    nu = alex.linearelastic.get_nu(lam=lam, mu=mu) # should be effective values?
    r = ufl.sqrt(ufl.inner(dx,dx) + ufl.inner(dy,dy))
    theta = ufl.atan2(dy, dx)
    
    u_x = K1 / (2.0 * mu * math.sqrt(2.0 * math.pi))  * ufl.sqrt(r) * (3.0 - 4.0 * nu - ufl.cos(theta)) * ufl.cos(0.5 * theta)
    u_y = K1 / (2.0 * mu * math.sqrt(2.0 * math.pi))  * ufl.sqrt(r) * (3.0 - 4.0 * nu - ufl.cos(theta)) * ufl.sin(0.5 * theta)
    return ufl.as_vector([u_x, u_y, 0.0]) 

atol=(x_max_all-x_min_all)*0.02 # for selection of boundary

boundary_surfing_bc = bc.get_boundary_of_box_as_function(domain, comm,atol=atol,epsilon=0.0)
bc_expression = dlfx.fem.Expression(compute_surf_displacement(),W.sub(0).element.interpolation_points())
facets_at_boundary = dlfx.mesh.locate_entities_boundary(domain, fdim, boundary_surfing_bc)
dofs_at_boundary = dlfx.fem.locate_dofs_topological(W.sub(0), fdim, facets_at_boundary) 


def get_bcs(t):
    xxK1.value = np.array([crack_start.value[0] + vcrack_const.value[0] * t_global.value,
                           crack_start.value[1] + vcrack_const.value[1] * t_global.value,
                           0.0],dtype=dlfx.default_scalar_type)
    
    
    bcs = []
    w_D.sub(0).interpolate(bc_expression)
    bc_surf : dlfx.fem.DirichletBC = dlfx.fem.dirichletbc(w_D,dofs_at_boundary)

    
    # irreversibility
    if(abs(t)> sys.float_info.epsilon*5): # dont do before first time step
        bcs.append(pf.irreversibility_bc(domain,W,wm1))
    bcs.append(bccrack)
    bcs.append(bc_surf)
    
    return bcs

n = ufl.FacetNormal(domain)
external_surface_tag = 5
#external_surface_tags = pp.tag_part_of_boundary(domain,bc.get_boundary_of_box_as_function(domain, comm,atol=atol*0.0),external_surface_tag)
external_surface_tags = pp.tag_part_of_boundary(domain,boundary_surfing_bc,external_surface_tag)
ds = ufl.Measure('ds', domain=domain, subdomain_data=external_surface_tags,metadata={"quadrature_degree": deg_quad, "quadrature_scheme": "default"})

top_surface_tag = 9
top_surface_tags = pp.tag_part_of_boundary(domain,bc.get_top_boundary_of_box_as_function(domain, comm,atol=atol*0.0),top_surface_tag)
ds_top_tagged = ufl.Measure('ds', domain=domain, subdomain_data=top_surface_tags,metadata={"quadrature_degree": deg_quad, "quadrature_scheme": "default"})


success_timestep_counter = dlfx.fem.Constant(domain,0.0)
postprocessing_interval = dlfx.fem.Constant(domain,1.0)

Work = dlfx.fem.Constant(domain,0.0)

TEN = dlfx.fem.functionspace(domain, ("DP", deg_quad-1, (dim, dim)))
sigma_interpolated = dlfx.fem.Function(TEN) 
eshelby_interpolated = dlfx.fem.Function(TEN) 

S0e = basix.ufl.element("DP", domain.basix_cell(), 0, shape=())
S0 = dlfx.fem.functionspace(domain, S0e)


def after_timestep_success(t,dt,iters):
    
    um1, _ = ufl.split(wm1)
    
    alex.plasticity.update_e_p_n_and_alpha_arrays_3D(
        u,
        e_p_11_n_tmp, e_p_22_n_tmp, e_p_33_n_tmp,
        e_p_12_n_tmp, e_p_13_n_tmp, e_p_23_n_tmp,
        e_p_11_n,     e_p_22_n,     e_p_33_n,
        e_p_12_n,     e_p_13_n,     e_p_23_n,
        alpha_tmp, alpha_n,
        domain, cells, quadrature_points,
        sig_y, hard, mu
    )
    
    sigma = phaseFieldProblem.sigma_degraded(u,s,lam,mu,eta)
    tensor_field_expression = dlfx.fem.Expression(sigma, 
                                                         TEN.element.interpolation_points())
    tensor_field_name = "sigma"

    sigma_interpolated.interpolate(tensor_field_expression)
    sigma_interpolated.name = tensor_field_name
    Rx_top, Ry_top, Rz_top = pp.reaction_force(sigma_interpolated,n=n,ds=ds_top_tagged(top_surface_tag),comm=comm)
    
    dW = pp.work_increment_external_forces(sigma_interpolated,u,um1,n,ds,comm=comm)
    Work.value = Work.value + dW
    
    A = pf.get_surf_area(s,epsilon=epsilon,dx=ufl.dx, comm=comm)
    
    E_el = phaseFieldProblem.get_E_el_global(s,eta,u,lam,mu,dx=ufl.dx,comm=comm)
    
    if rank == 0:
        sol.write_to_newton_logfile(logfile_path,t,dt,iters)
        
    eshelby = phaseFieldProblem.getEshelby(w,eta,lam,mu)
    tensor_field_expression = dlfx.fem.Expression(eshelby, 
                                                         TEN.element.interpolation_points())
    tensor_field_name = "eshelby"
    eshelby_interpolated.interpolate(tensor_field_expression)
    eshelby_interpolated.name = tensor_field_name
        
    
    J3D_glob_x, J3D_glob_y, J3D_glob_z = alex.linearelastic.get_J_3D(eshelby_interpolated,n,ds=ds(external_surface_tag),comm=comm)

    
    if rank == 0:
        print(pp.getJString(J3D_glob_x, J3D_glob_y, J3D_glob_z))
        

    
    # s_aux = dlfx.fem.Function(S)
    # s_aux.interpolate(s)
    
    # s_zero_for_tracking.x.array[:] = s.collapse().x.array[:]
    s_zero_for_tracking_at_nodes.interpolate(s)
    x_tip, max_y, max_z, min_x, min_y, min_z = pp.crack_bounding_box_3D(domain, pf.get_dynamic_crack_locator_function(wm1,s_zero_for_tracking_at_nodes),comm)
    if rank == 0:
        print("Crack tip position x: " + str(x_tip))
        pp.write_to_graphs_output_file(outputfile_graph_path,t, J3D_glob_x, J3D_glob_y, J3D_glob_z,x_tip, xxK1.value[0], Rx_top, Ry_top, Rz_top, dW, Work.value, A)
    
    
    # update
    wm1.x.array[:] = w.x.array[:]
    wrestart.x.array[:] = w.x.array[:]
    
    pp.write_tensor_fields(domain,comm,[sigma_interpolated],["sigma"],outputfile_xdmf_path=outputfile_xdmf_path,t=t)
    pp.write_field(domain,outputfile_xdmf_path,alpha_n,t,comm,S=S0)
    pp.write_phasefield_mixed_solution(domain,outputfile_xdmf_path, w, t, comm)

    
    
def after_timestep_restart(t,dt,iters):
    w.x.array[:] = wrestart.x.array[:]
    
    
def after_last_timestep():
    # stopwatch stop
    timer.stop()

    # report runtime to screen
    if rank == 0:
        runtime = timer.elapsed()
        sol.print_runtime(runtime)
        sol.write_runtime_to_newton_logfile(logfile_path,runtime)
        pp.print_graphs_plot(outputfile_graph_path,script_path,legend_labels=["Jx", "Jy", "Jz", "x_tip", "xtip", "Rx", "Ry", "Rz", "dW", "W", "A"])

        # cleanup only necessary on cluster
        results_folder_path = alex.os.create_results_folder(script_path)
        alex.os.copy_contents_to_results_folder(script_path,results_folder_path)

sol.solve_with_newton_adaptive_time_stepping(
    domain=domain,
    w=w,
    Tend=Tend,
    before_first_timestep_hook=before_first_time_step,
    after_last_timestep_hook=after_last_timestep,
    before_each_timestep_hook=before_each_time_step,
    get_residuum_and_gateaux=get_residuum_and_gateaux,
    get_bcs=get_bcs,
    after_timestep_restart_hook=after_timestep_restart,
    after_timestep_success_hook=after_timestep_success,
    comm=comm,
    print_bool=True,
    t=t_global,
    dt=dt_global,
)

