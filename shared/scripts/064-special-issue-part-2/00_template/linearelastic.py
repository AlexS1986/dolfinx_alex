import alex.homogenization
import alex.linearelastic
import alex.phasefield
import alex.util
import dolfinx as dlfx
from mpi4py import MPI


import ufl 
import numpy as np
import os 
import sys

import alex.os
import alex.boundaryconditions as bc
import alex.postprocessing as pp
import alex.solution as sol
import alex.linearelastic as le

import json

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
logfile_path = alex.os.logfile_full_path(script_path,script_name_without_extension)
outputfile_graph_path = alex.os.outputfile_graph_full_path(script_path,script_name_without_extension)
outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path,script_name_without_extension)

# set and start stopwatch
timer = dlfx.common.Timer()
timer.start()

# set MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
sys.stdout.flush()

N = 16 

#     # generate domain
#     #domain = dlfx.mesh.create_unit_square(comm, N, N, cell_type=dlfx.mesh.CellType.quadrilateral)
domain = dlfx.mesh.create_unit_cube(comm,N,N,N,cell_type=dlfx.mesh.CellType.hexahedron)

# with dlfx.io.XDMFFile(comm, os.path.join(script_path, 'dlfx_mesh.xdmf'), 'r') as mesh_inp:
#    domain = mesh_inp.read_mesh()


dt = dlfx.fem.Constant(domain,1.0)
t = dlfx.fem.Constant(domain,0.00)
column = dlfx.fem.Constant(domain,0.0)
Tend = 6.0 * dt.value

# mu_val = alex.linearelastic.get_mu(1000,0.35)
# la_val = alex.linearelastic.get_lambda(1000,0.35)

# elastic constants
# lam = dlfx.fem.Constant(domain, la_val)
# mu = dlfx.fem.Constant(domain, mu_val)
lam = dlfx.fem.Constant(domain, 51100.0)
mu = dlfx.fem.Constant(domain, 26300.0)

E_mod = alex.linearelastic.get_emod(lam.value, mu.value)

# function space using mesh and degree
Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1) # displacements
V = dlfx.fem.FunctionSpace(domain, Ve)
Se = ufl.FiniteElement("Lagrange", domain.ufl_cell(),1) 
S = dlfx.fem.FunctionSpace(domain,Se)

# define boundary condition on top and bottom
fdim = domain.topology.dim -1

bcs = []
             
# define solution, restart, trial and test space
u =  dlfx.fem.Function(V)
urestart =  dlfx.fem.Function(V)
du = ufl.TestFunction(V)
ddu = ufl.TrialFunction(V)

def before_first_time_step():
    urestart.x.array[:] = np.ones_like(urestart.x.array[:])
    
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
      
linearElasticProblem = alex.linearelastic.StaticLinearElasticProblem()

def get_residuum_and_gateaux(delta_t: dlfx.fem.Constant):
    [Res, dResdw] = linearElasticProblem.prep_newton(u,du,ddu,lam,mu)
    return [Res, dResdw]




x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = bc.get_dimensions(domain,comm)

atol=(x_max_all-x_min_all)*0.05 # for selection of boundary


u_D = dlfx.fem.Function(V)

x = ufl.SpatialCoordinate(domain)

alpha=1.0
beta=0.05

def compute_mixed_bc_u(eps_rate,t,direction=0):
    u = eps_rate*x[direction]*t
    return ufl.as_ufl(u)
     


# boundary = bc.get_boundary_of_box_as_function(domain,comm,atol=atol)
# facets_at_boundary = dlfx.mesh.locate_entities_boundary(domain, fdim, boundary)
# dofs_at_boundary = dlfx.fem.locate_dofs_topological(V, fdim, facets_at_boundary) 

left_right_boundary = bc.get_leftright_boundary_of_box_as_function(domain,atol=atol,comm=comm)
facets_at_left_right_boundary = dlfx.mesh.locate_entities_boundary(domain, fdim, left_right_boundary)
dofs_at_left_right_boundary = dlfx.fem.locate_dofs_topological(V.sub(0), fdim, facets_at_left_right_boundary) 
#bc_expression_left_right = dlfx.fem.Expression(compute_mixed_bc_u(eps_rate=(alpha+beta),direction=0,t=t),V.sub(0).element.interpolation_points())


top_bottom_boundary = bc.get_top_boundary_of_box_as_function(domain,atol=atol,comm=comm)
facets_at_top_bottom_boundary = dlfx.mesh.locate_entities_boundary(domain, fdim, top_bottom_boundary)
dofs_at_top_bottom_boundary = dlfx.fem.locate_dofs_topological(V.sub(1), fdim, facets_at_top_bottom_boundary) 
#bc_expression_top_bottom = dlfx.fem.Expression(compute_mixed_bc_u(eps_rate=(-alpha+beta),direction=1,t=t),V.sub(1).element.interpolation_points())


front_back_boundary = bc.get_frontback_boundary_of_box_as_function(domain,atol=atol,comm=comm)
facets_at_front_back_boundary = dlfx.mesh.locate_entities_boundary(domain, fdim, front_back_boundary)
dofs_at_front_back_boundary = dlfx.fem.locate_dofs_topological(V.sub(2), fdim, facets_at_front_back_boundary) 
#bc_expression_front_back = dlfx.fem.Expression(compute_mixed_bc_u(eps_rate=beta,direction=2,t=t),V.sub(2).element.interpolation_points())




eps_mac = dlfx.fem.Constant(domain, np.array([[(alpha+beta)*t.value, 0.0, 0.0],
                     [0.0, (-alpha+beta)*t.value, 0.0],
                     [0.0, beta*t.value, 0.0]]))


def get_bcs(t):
    # Eps_Voigt = np.zeros((6,))
    # Eps_Voigt[int(t)] = 1.0
    
    # eps_mac.value = np.array([[Eps_Voigt[0], Eps_Voigt[5]/2.0, Eps_Voigt[4]/2.0],
    #                                           [Eps_Voigt[5]/2.0, Eps_Voigt[1], Eps_Voigt[3]/2.0],
    #                                           [Eps_Voigt[4]/2.0, Eps_Voigt[3]/2.0, Eps_Voigt[2]]])
        
    # if (t>5):
    #      eps_mac.value = np.array([[0.0, 0.0, 0.0],
    #                  [0.0, 0.0, 0.0],
    #                  [0.0, 0.0, 0.0]])
    
    # comm.barrier()
    
    def compute_linear_displacement():
        x = ufl.SpatialCoordinate(domain)
        
        u_x = eps_mac.value[0,0]*x[0] + eps_mac.value[0,1]*x[1] + eps_mac.value[0,2]*x[2]
        u_y = eps_mac.value[1,0]*x[0] + eps_mac.value[1,1]*x[1] + eps_mac.value[1,2]*x[2]
        u_z = eps_mac.value[2,0]*x[0] + eps_mac.value[2,1]*x[1] + eps_mac.value[2,2]*x[2]
        #u_linear_displacement = ufl.inner(eps_mac,x)
        return ufl.as_vector([u_x, u_y, u_z])
    
    

     
   
    bc_expression = dlfx.fem.Expression(compute_linear_displacement(),V.element.interpolation_points())
    u_D.interpolate(bc_expression)
    
    # u_D.sub(0).interpolate(bc_expression_left_right)
    # u_D.sub(1).interpolate(bc_expression_top_bottom )
    # u_D.sub(2).interpolate(bc_expression_front_back)
    
    bc_left_right = dlfx.fem.dirichletbc(u_D,dofs_at_left_right_boundary,V)
    bc_top_bottom = dlfx.fem.dirichletbc(u_D,dofs_at_top_bottom_boundary,V)
    bc_front_back = dlfx.fem.dirichletbc(u_D,dofs_at_front_back_boundary,V)
    
    
    # u_D.interpolate(bc_expression)
    
    # bc_linear_displacement = dlfx.fem.dirichletbc(u_D,dofs_at_boundary)
    
    ux_right = 1.0
    ux_left = 0.0
    
    bc_right = bc.define_dirichlet_bc_from_value(domain,ux_right,0,bc.get_right_boundary_of_box_as_function(domain,comm,atol),V,0)
    bc_left = bc.define_dirichlet_bc_from_value(domain,ux_left,0,bc.get_left_boundary_of_box_as_function(domain,comm,atol),V,0)
    
    
    bcs = [bc_left, bc_right]
    
    #bcs = [bc_left_right, bc_top_bottom, bc_front_back]
    return bcs

#n = ufl.FacetNormal(domain)
simulation_result = np.array([0.0])

Chom = np.zeros((6, 6))



## create integration measure for homogenization
tag_value_hom_cells = 1
marker1 = bc.dont_get_boundary_of_box_as_function(domain,comm,atol=atol)
marked_cells = dlfx.mesh.locate_entities(domain, dim=3, marker=marker1)
marked_values = np.full(len(marked_cells), tag_value_hom_cells, dtype=np.int32)
cell_tags = dlfx.mesh.meshtags(domain, 3, marked_cells, marked_values)

submesh, _ , _, _ = dlfx.mesh.create_submesh(domain, 3, cell_tags.indices[cell_tags.values == tag_value_hom_cells])
coords = submesh.geometry.x

dx_hom_cells = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
x_min_hom_all, x_max_hom_all, y_min_hom_all, y_max_hom_all, z_min_hom_all, z_max_hom_all = bc.get_tagged_subdomain_bounds(domain, cell_tags, tag_value_hom_cells, comm)

# x_min_hom = np.min(coords[:,0]) 
# x_max_hom = np.max(coords[:,0])   
# y_min_hom = np.min(coords[:,1]) 
# y_max_hom = np.max(coords[:,1])   
# z_min_hom = np.min(coords[:,2]) 
# z_max_hom = np.max(coords[:,2])

# comm.Barrier()
# x_min_hom_all = comm.allreduce(x_min_hom, op=MPI.MIN)
# x_max_hom_all = comm.allreduce(x_max_hom, op=MPI.MAX)
# y_min_hom_all = comm.allreduce(y_min_hom, op=MPI.MIN)
# y_max_hom_all = comm.allreduce(y_max_hom, op=MPI.MAX)
# z_min_hom_all = comm.allreduce(z_min_hom, op=MPI.MIN)
# z_max_hom_all = comm.allreduce(z_max_hom, op=MPI.MAX)
# comm.Barrier()

vol = (x_max_hom_all-x_min_hom_all) * (y_max_hom_all - y_min_hom_all) * (z_max_hom_all - z_min_hom_all)
vol_material = alex.homogenization.get_filled_vol(dx_hom_cells(tag_value_hom_cells),comm)
vol_overall = (x_max_all-x_min_all) * (y_max_all - y_min_all) * (z_max_all - z_min_all)

if rank == 0:
    print("=== Homogenization Box Boundaries ===")
    print(f"x: [{x_min_hom_all}, {x_max_hom_all}]")
    print(f"y: [{y_min_hom_all}, {y_max_hom_all}]")
    print(f"z: [{z_min_hom_all}, {z_max_hom_all}]")
    print("Volume of Cuboid for Homogenization: " + str(vol))
    print("Volume of Real Material in Homogenization Box: " + str(vol_material))

    print("\n=== Total Domain Boundaries ===")
    print(f"x: [{x_min_all}, {x_max_all}]")
    print(f"y: [{y_min_all}, {y_max_all}]")
    print(f"z: [{z_min_all}, {z_max_all}]")
    print("Volume of Cuboid Total: " + str(vol_overall))

    


def after_timestep_success(t,dt,iters):
    u.name = "u"
    pp.write_vector_field(domain,outputfile_xdmf_path,u,t,comm)
    sigma_for_unit_strain = alex.homogenization.compute_averaged_sigma(u,lam,mu, vol,comm=comm,dx=dx_hom_cells(tag_value_hom_cells))
    
    eps = alex.linearelastic.eps_as_tensor(u)
    pp.compute_and_write_tensor_eigenvalue(domain,eps,"epsilon_main",t,outputfile_xdmf_path,comm)
    
    # write to newton-log-file
    comm.barrier()
    if rank == 0:
        if column.value < 6:
            Chom[int(column.value)] = sigma_for_unit_strain
        else:
            t = 2.0*Tend # exit
            return
        #print(column.value)
        column.value = column.value + 1
        sol.write_to_newton_logfile(logfile_path,t,dt,iters)
        
    urestart.x.array[:] = u.x.array[:] 
               
def after_timestep_restart(t,dt,iters):
    #raise RuntimeError("Linear computation - NO RESTART NECESSARY")
    u.x.array[:] = urestart.x.array[:]
     
def after_last_timestep():
    # stopwatch stop
    timer.stop()

    if rank == 0:
        print(np.array_str(Chom, precision=2))
        print(alex.homogenization.print_results(Chom))

        # Save Chom to JSON
        chom_path = os.path.join(script_path, "Chom.json")
        with open(chom_path, "w") as f:
            json.dump(Chom.tolist(), f, indent=4)
        print(f"Saved Chom matrix to: {chom_path}")

        runtime = timer.elapsed()
        sol.print_runtime(runtime)
        sol.write_runtime_to_newton_logfile(logfile_path, runtime)

sol.solve_with_newton_adaptive_time_stepping(
    domain,
    u,
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
    t=t,
    dt_never_scale_up=True
)
