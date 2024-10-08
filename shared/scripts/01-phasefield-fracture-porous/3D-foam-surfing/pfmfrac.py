import dolfinx as dlfx
from mpi4py import MPI
from petsc4py import PETSc as petsc

import ufl 
import numpy as np
import os 
import sys
import math

import alex.os
import alex.phasefield as pf
import alex.boundaryconditions as bc
import alex.postprocessing as pp
import alex.solution as sol
import alex.linearelastic

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
logfile_path = alex.os.logfile_full_path(script_path,script_name_without_extension)
outputfile_graph_path = alex.os.outputfile_graph_full_path(script_path,script_name_without_extension)
outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path,script_name_without_extension)


# set and start stopwatch
timer = dlfx.common.Timer()
timer.start()

# set MPI environment
comm, rank, size = alex.os.set_mpi()
alex.os.print_mpi_status(rank, size)

with dlfx.io.XDMFFile(comm, os.path.join(alex.os.resources_directory,'foam_mesh.xdmf'), 'r') as mesh_inp: 
    domain = mesh_inp.read_mesh()

Tend = 5.0
dt = 0.0001

# function space using mesh and degree
Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1) # displacements
Te = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1) # fracture fields
W = dlfx.fem.FunctionSpace(domain, ufl.MixedElement([Ve, Te]))


dim = domain.topology.dim
alex.os.mpi_print('spatial dimensions: '+str(dim), rank)

x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = pp.compute_bounding_box(comm, domain)
pp.print_bounding_box(rank, x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all)

# elastic constants
lam = dlfx.fem.Constant(domain, 10.0)
mu = dlfx.fem.Constant(domain, 10.0)

# residual stiffness
eta = dlfx.fem.Constant(domain, 0.01)

# phase field parameters
Gc = dlfx.fem.Constant(domain, 1.0)
# epsilon = dlfx.fem.Constant(domain, 0.3*(x_max_all - x_min_all))
epsilon = dlfx.fem.Constant(domain, 100.0)
Mob = dlfx.fem.Constant(domain, 1000.0)
iMob = dlfx.fem.Constant(domain, 1.0/Mob.value)

# define crack by boundary
crack_tip_start_location_x = 0.05*(x_max_all-x_min_all) + x_min_all
crack_tip_start_location_y = (y_max_all / 2.0)
def crack(x):
    # x_log = x[0]< (crack_tip_start_location_x)
    # y_log = np.isclose(x[1],crack_tip_start_location_y,atol=(0.02*((y_max_all-y_min_all))))
    x_log = x[0]< 50
    y_log = np.isclose(x[1],200,atol=(0.02*(200)))
    return np.logical_and(y_log,x_log)

fdim = domain.topology.dim -1
crackfacets = dlfx.mesh.locate_entities(domain, fdim, crack)
crackdofs = dlfx.fem.locate_dofs_topological(W.sub(1), fdim, crackfacets)
bccrack = dlfx.fem.dirichletbc(0.0, crackdofs, W.sub(1))


E_mod = alex.linearelastic.get_emod(lam.value, mu.value)
K1 = dlfx.fem.Constant(domain, 0.001 * math.sqrt(Gc.value*E_mod))
xtip = np.array([0.0, 200],dtype=dlfx.default_scalar_type)
xK1 = dlfx.fem.Constant(domain, xtip)
bcs = bc.get_total_surfing_boundary_condition_at_box(domain=domain,comm=comm,functionSpace=W,subspace_idx=0,K1=K1,xK1=xK1,lam=lam,mu=mu,epsilon=0.0* epsilon.value) # fix boundary everywhere? -> epsilon = 0.0
bcs.append(bccrack)


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


def before_each_time_step(t,dt):
    # report solution status
    if rank == 0:
        sol.print_time_and_dt(t,dt)
        

phaseFieldProblem = pf.StaticPhaseFieldProblem3D(degradationFunction=pf.degrad_quadratic,
                                                   psisurf=pf.psisurf)


def get_residuum_and_gateaux(delta_t: dlfx.fem.Constant):
    [Res, dResdw] = phaseFieldProblem.prep_newton(
        w=w,wm1=wm1,dw=dw,ddw=ddw,lam=lam, mu = mu,
        Gc=Gc,epsilon=epsilon, eta=eta,
        iMob=iMob, delta_t=delta_t)
    return [Res, dResdw]

Se = ufl.FiniteElement("Lagrange", domain.ufl_cell(),1) 
S = dlfx.fem.FunctionSpace(domain,Se)
s_zero_for_tracking_at_nodes = dlfx.fem.Function(S)
c = dlfx.fem.Constant(domain, petsc.ScalarType(1))
sub_expr = dlfx.fem.Expression(c,S.element.interpolation_points())
s_zero_for_tracking_at_nodes.interpolate(sub_expr)

[Res, dResdw] = get_residuum_and_gateaux(delta_t=dt)

w_D = dlfx.fem.Function(W) # for dirichlet BCs
xtip = np.array([50 , 200],dtype=dlfx.default_scalar_type)
xK1 = dlfx.fem.Constant(domain, xtip)
bcs = bc.get_total_surfing_boundary_condition_at_box(domain=domain,comm=comm,
                                                     functionSpace=W,subspace_idx=0,
                                                     K1=K1,xK1=xK1,lam=lam,mu=mu,
                                                     epsilon=0.0*epsilon.value,w_D=w_D)

solver = sol.get_solver(w,comm,8,Res,dResdw=dResdw,bcs=bcs)

def get_bcs(t):
    v_crack = 700/Tend
    # xtip = np.array([crack_tip_start_location_x + v_crack * t, crack_tip_start_location_y])
    xtip = np.array([50 + v_crack * t, 200],dtype=dlfx.default_scalar_type)
    xK1 = dlfx.fem.Constant(domain, xtip)

    # Only update the displacement field w_D
    bc.surfing_boundary_conditions(w_D,K1,xK1,lam,mu,subspace_index=0) 
    
    # bcs = bc.get_total_surfing_boundary_condition_at_box(domain=domain,comm=comm,functionSpace=W,subspace_idx=0,K1=K1,xK1=xK1,lam=lam,mu=mu,epsilon=0.0*epsilon.value) # fix boundary everywhere? -> epsilon = 0.0
    # bcs = bc.get_total_linear_displacement_boundary_condition_at_box(domain, comm, W,eps_mac,0)
    
    # irreversibility
    if(abs(t)> sys.float_info.epsilon*5): # dont do before first time step
        bcs.append(pf.irreversibility_bc(domain,W,wm1))
    
    
    bcs.append(bccrack)
    return bcs

n = ufl.FacetNormal(domain)
external_surface_tags = pp.tag_part_of_boundary(domain,bc.get_boundary_of_box_as_function(domain, comm),5)
ds = ufl.Measure('ds', domain=domain, subdomain_data=external_surface_tags)

def after_timestep_success(t,dt,iters):
    pp.write_phasefield_mixed_solution(domain,outputfile_xdmf_path, w, t, comm)

    # write to newton-log-file
    if rank == 0:
        sol.write_to_newton_logfile(logfile_path,t,dt,iters)
        
    # compute J-Integral
    eshelby = phaseFieldProblem.getEshelby(w,eta,lam,mu)
    divEshelby = ufl.div(eshelby)
    pp.write_vector_fields(domain=domain,comm=comm,vector_fields_as_functions=[divEshelby],
                            vector_field_names=["Ge"], 
                            outputfile_xdmf_path=outputfile_xdmf_path,t=t)
    
    J3D_glob_x, J3D_glob_y, J3D_glob_z = alex.linearelastic.get_J_3D(eshelby, ds=ds(5), n=n,comm=comm)

    
    if rank == 0:
        print(pp.getJString(J3D_glob_x, J3D_glob_y, J3D_glob_z))
        

    # update
    wm1.x.array[:] = w.x.array[:]
    wrestart.x.array[:] = w.x.array[:]
    
    # s_aux = dlfx.fem.Function(S)
    # s_aux.interpolate(s)
    
    # s_zero_for_tracking.x.array[:] = s.collapse().x.array[:]
    s_zero_for_tracking_at_nodes.interpolate(s)
    x_tip, max_y, max_z, min_x, min_y, min_z = pp.crack_bounding_box_3D(domain, pf.get_dynamic_crack_locator_function(wm1,s_zero_for_tracking_at_nodes),comm)
    if rank == 0:
        print("Crack tip position x: " + str(x_tip))
        pp.write_to_graphs_output_file(outputfile_graph_path,t, J3D_glob_x, J3D_glob_y, J3D_glob_z,x_tip)
    
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
        pp.print_graphs_plot(outputfile_graph_path,script_path,legend_labels=["x_tip"])
    

sol.solve_with_newton_adaptive_time_stepping(
    domain,
    w,
    Tend,
    dt,
    before_first_timestep_hook=before_first_time_step,
    after_last_timestep_hook=after_last_timestep,
    before_each_timestep_hook=before_each_time_step,
    get_residuum_and_gateaux=get_residuum_and_gateaux,
    get_bcs=get_bcs,
    after_timestep_restart_hook=after_timestep_restart,
    after_timestep_success_hook=after_timestep_success,
    comm=comm,
    print_bool=True,
    solver=solver
)

