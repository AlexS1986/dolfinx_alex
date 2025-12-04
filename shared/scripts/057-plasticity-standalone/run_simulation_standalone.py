import dolfinx as dlfx
import os
import numpy as np
import ufl
import basix
from functools import reduce
from typing import Callable


import alex.os
import alex.boundaryconditions as bc
import alex.postprocessing as pp
import alex.solution as sol


import shutil
from datetime import datetime
import alex.plasticity

import os
import sys
import shutil
from datetime import datetime
from mpi4py import MPI
import glob

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

import matplotlib.pyplot as plt


def logfile_full_path(script_path: str, script_name_without_extension: str) -> str:
    return os.path.join(script_path, script_name_without_extension + "_log.txt")

def outputfile_graph_full_path(script_path: str, script_name_without_extension: str) -> str:
    return os.path.join(script_path, script_name_without_extension + "_graphs.txt")

def outputfile_xdmf_full_path(script_path: str, script_name_without_extension: str) -> str:
    return os.path.join(script_path, script_name_without_extension + ".xdmf")

def print_mpi_status(rank, size):
    print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
    sys.stdout.flush()

def set_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm,rank,size

def mpi_print(output, rank=0):
    if rank == 0:
        print(output)
        sys.stdout.flush
    return

def define_custom_integration_measure_that_matches_quadrature_degree_and_scheme(domain, deg_quad, quad_scheme):
    dx = ufl.Measure(
    "dx",
    domain=domain,
    metadata={"quadrature_degree": deg_quad, "quadrature_scheme": quad_scheme},
    )
    
    return dx

def get_quadraturepoints_and_cells_for_inter_polation_at_gauss_points(domain, deg_quad):
    basix_celltype = getattr(basix.CellType, domain.topology.cell_types[0].name) # 7.3
    #basix_celltype = getattr(basix.CellType, domain.topology.cell_type.name) # 8.0
    quadrature_points, weights = basix.make_quadrature(basix_celltype, deg_quad,rule=basix.quadrature.string_to_type("default"))

    map_c = domain.topology.index_map(domain.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)
    return quadrature_points,cells

def f_tr_func(u,e_p_n,alpha_n,sig_y,hard,mu):
        eps_np1_3D_plane_strain = assemble_3D_representation_of_plane_strain_eps(u)
        
        
        #e_np1 = ufl.dev(ufl.sym(ufl.grad(u)))
        e_np1 = ufl.dev(eps_np1_3D_plane_strain)
        s_tr = 2.0*mu*(e_np1-e_p_n)
        norm_s_tr = ufl.sqrt(ufl.inner(s_tr,s_tr))
        f_tr = norm_s_tr -np.sqrt(2.0/3.0) * (sig_y+hard*alpha_n)
        return f_tr

def assemble_3D_representation_of_plane_strain_eps(u):
    eps_np1_2D = ufl.sym(ufl.grad(u))
    eps_np1_3D_plane_strain = ufl.as_tensor([[eps_np1_2D[0,0], eps_np1_2D[0,1], 0.0],
                                                [ eps_np1_2D[1,0], eps_np1_2D[1,1], 0.0],
                                                [ 0.0,             0.0,             0.0]])
                                            
    return eps_np1_3D_plane_strain

def get_K(lam: float, mu: float):
    return  lam + 2.0 / 3.0 * mu

def sig_plasticity(u,e_p_n,alpha_n,sig_y,hard,lam,mu):  
    eps_np1 = assemble_3D_representation_of_plane_strain_eps(u)
    e_np1 = ufl.dev(eps_np1)
        
    s_tr = 2.0*mu*(e_np1-e_p_n)
        
    norm_s_tr_val = ufl.sqrt(ufl.inner(s_tr,s_tr))
    norm_s_tr = ufl.conditional(ufl.lt(norm_s_tr_val, 1000.0*np.finfo(np.float64).resolution), 1000.0*np.finfo(np.float64).resolution, norm_s_tr_val)
    
    #norm_s_tr = ufl.sqrt(ufl.inner(s_tr,s_tr))
    
    f_tr = f_tr_func(u,e_p_n,alpha_n,sig_y,hard,mu)
    dgamma = f_tr / (2.0*(mu+hard/3))
    
    N_np1 = s_tr / norm_s_tr
    s_np1 = ufl.conditional(ufl.le(f_tr,0.0),s_tr,s_tr - 2.0*mu*dgamma*N_np1)
    K = get_K(lam=lam,mu=mu)
    sig_3D = K * ufl.tr(eps_np1)*ufl.Identity(3) + s_np1
    
    sig_2D = ufl.as_tensor([[sig_3D[0,0], sig_3D[0,1]],
                            [sig_3D[1,0], sig_3D[1,1]]])
    
    return sig_2D

def prepare_newton_logfile(logfile_path: str):
    for file in glob.glob(logfile_path):
        os.remove(logfile_path)
    logfile = open(logfile_path, 'w')  
    logfile.write('# time, dt, no. iterations (for convergence) \n')
    logfile.close()
    return True

class Plasticity_incremental_2D:
    # Constructor method
    def __init__(self, 
                       sig_y: any,
                       hard: any,
                       alpha_n: any,
                       e_p_n: any,
                       H: any,
                       dx: any = ufl.dx,
                 ):


        # Set all parameters here! Material etc
        self.dx = dx
        self.sig_y = sig_y
        self.hard = hard
        self.e_p_n = e_p_n
        self.alpha_n = alpha_n
        self.H = H
        
        
    def prep_newton(self, u: any, um1: any, du: ufl.TestFunction, ddu: ufl.TrialFunction, lam: dlfx.fem.Function, mu: dlfx.fem.Function ):
        def residuum(u: any, du: any,  um1:any):
            
            delta_u = u - um1
            
            equi =  (ufl.inner(self.sigma(u,lam,mu),  0.5*(ufl.grad(du) + ufl.grad(du).T)))*self.dx # ufl.derivative(pot, u, du)
            H_np1 = self.update_H(u,delta_u=delta_u,lam=lam,mu=mu)
            
            Res = equi
            return [ Res, None]        
        return residuum(u,du,um1)
    
    def sigma(self, u,lam,mu):
        return  sig_plasticity(u,e_p_n=self.e_p_n,alpha_n=self.alpha_n,sig_y=self.sig_y,hard=self.hard,lam=lam,mu=mu)
        # return 1.0 * le.sigma_as_tensor3D(u=u,lam=lam,mu=mu)
    
    def eps(self,u):
        return ufl.sym(ufl.grad(u)) #0.5*(ufl.grad(u) + ufl.grad(u).T)
    
    def deveps(self,u):
        return ufl.dev(self.eps(u))
    
    def eqeps(self,u):
        return ufl.sqrt(2.0/3.0 * ufl.inner(self.eps(u),self.eps(u))) 
    
    def update_H(self, u, delta_u,lam,mu):
        u_n = u-delta_u
        delta_eps = 0.5*(ufl.grad(delta_u) + ufl.grad(delta_u).T)
        W_np1 = ufl.inner(self.sigma(u=u,lam=lam,mu=mu), delta_eps )
        W_n = ufl.inner(self.sigma(u=u_n,lam=lam,mu=mu), delta_eps )
        H_np1 = ( self.H + 0.5 * (W_n+W_np1))
        return H_np1
    
    def psiel(self,u,lam,mu):
        return  self.H
    
    def get_E_el_global(self,u,lam,mu, dx: ufl.Measure, comm: MPI.Intercomm) -> float:
        Pi = dlfx.fem.assemble_scalar(dlfx.fem.form(self.psiel(u,lam,mu) * dx))
        return comm.allreduce(Pi,MPI.SUM)
    
def interpolate_quadrature(domain, cells, quadrature_points, ufl_expr):
    expr_expr = dlfx.fem.Expression(ufl_expr, quadrature_points)
    expr_eval = expr_expr.eval(domain, cells)
    return expr_eval.flatten()[:]
    # function.x.array[:] = expr_eval.flatten()[:]
    
def prepare_newton_logfile(logfile_path: str):
    for file in glob.glob(logfile_path):
        os.remove(logfile_path)
    logfile = open(logfile_path, 'w')  
    logfile.write('# time, dt, no. iterations (for convergence) \n')
    logfile.close()
    return True

def prepare_graphs_output_file(output_file_path: str):
    for file in glob.glob(output_file_path):
        os.remove(output_file_path)
    logfile = open(output_file_path, 'w')  
    logfile.write('# This is a general outputfile for displaying scalar quantities vs time, first column is time, further columns are data \n')
    logfile.close()
    return True

def write_meshoutputfile(domain: dlfx.mesh.Mesh,
                                       outputfile_path: str,
                                       comm: MPI.Intercomm,
                                       meshtags: any = None):
    
    if outputfile_path.endswith(".xdmf"):
        with dlfx.io.XDMFFile(comm, outputfile_path, 'w') as xdmfout:
            xdmfout.write_mesh(domain)
            if( not meshtags is None):
                xdmfout.write_meshtags(meshtags, domain.geometry)
            xdmfout.close()
        # xdmfout = dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a')
    # xdmfout.write_function()
        # xdmfout.write_function(field, t)
            # xdmfout.write_function(field_interp, t)
    elif outputfile_path.endswith(".vtk"):
        with dlfx.io.VTKFile(comm, outputfile_path, 'w') as vtkout:
            vtkout.write_mesh(domain)
            # vtkout.write_function(field_interp,t)
    # xdmfout = dlfx.io.XDMFFile(comm, outputfile_path, 'w')
    else:
        return False
    
    return True

def print_time_and_dt(t: float, dt: float):
    print(' ')
    print('==================================================')
    print('Computing solution at time = {0:.4e}'.format(t))
    print('==================================================')
    print('Current time step dt = {0:.4e}'.format(dt))
    print('==================================================')
    print(' ')
    sys.stdout.flush()
    return True

def close_func(x,value,atol):
        if atol:
            return np.isclose(x,value,atol=atol)
        else:
            return np.isclose(x,value)
        
def get_dimensions(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm):
        x_min = np.min(domain.geometry.x[:,0]) 
        x_max = np.max(domain.geometry.x[:,0])   
        y_min = np.min(domain.geometry.x[:,1]) 
        y_max = np.max(domain.geometry.x[:,1])   
        z_min = np.min(domain.geometry.x[:,2]) 
        z_max = np.max(domain.geometry.x[:,2])

        # find global min/max over all mpi processes
        comm.Barrier()
        x_min_all = comm.allreduce(x_min, op=MPI.MIN)
        x_max_all = comm.allreduce(x_max, op=MPI.MAX)
        y_min_all = comm.allreduce(y_min, op=MPI.MIN)
        y_max_all = comm.allreduce(y_max, op=MPI.MAX)
        z_min_all = comm.allreduce(z_min, op=MPI.MIN)
        z_max_all = comm.allreduce(z_max, op=MPI.MAX)
        comm.Barrier()
        return x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all

def get_top_boundary_of_box_as_function(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, atol: float=None):
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = get_dimensions(domain, comm)
    def boundary(x):
        ymax = close_func(x[1],y_max_all,atol=atol)
        boundaries = [ymax]
        return reduce(np.logical_or, boundaries)
    return boundary

def get_bottom_boundary_of_box_as_function(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, atol: float=None):
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = get_dimensions(domain, comm)
    def boundary(x):
        ymin = close_func(x[1],y_min_all,atol=atol)
        boundaries = [ymin]
        return reduce(np.logical_or, boundaries)
    return boundary

def define_dirichlet_bc_from_value(domain: dlfx.mesh.Mesh,
                                                         desired_value_at_boundary: float,
                                                         coordinate_idx,
                                                         where_function,
                                                         functionSpace: dlfx.fem.FunctionSpace,
                                                         subspace_idx: int) -> dlfx.fem.DirichletBC:
    fdim = domain.topology.dim-1
    facets_at_boundary = dlfx.mesh.locate_entities_boundary(domain, fdim, where_function)
    if subspace_idx < 0: # not a phase field mixed function space
        space = functionSpace.sub(coordinate_idx)
    else:
        space = functionSpace.sub(subspace_idx).sub(coordinate_idx)
    dofs_at_boundary = dlfx.fem.locate_dofs_topological(space, fdim, facets_at_boundary)
    bc = dlfx.fem.dirichletbc(desired_value_at_boundary,dofs_at_boundary,space)
    return bc

def tag_part_of_boundary(domain: dlfx.mesh.Mesh, where, tag: int) -> any:
    '''
        https://jsdokken.com/dolfinx-tutorial/chapter2/hyperelasticity.html
        assigns tags to part of the boundary, which can then be used for surface integral
        
        returns the facet_tags
    '''
    fdim = domain.topology.dim - 1
    facets = dlfx.mesh.locate_entities_boundary(domain, fdim, where)
    marked_facets = np.hstack([facets])
    marked_values = np.hstack([np.full_like(facets,tag)])
    sorted_facets = np.argsort(marked_facets)
    facet_tags = dlfx.mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])
    
    
    return facet_tags

def get_boundary_of_box_as_function(
    domain: dlfx.mesh.Mesh,
    comm: MPI.Intracomm,
    atol = None,
    atol_x = None,
    atol_y = None,
    atol_z = None
):
    
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = get_dimensions(domain, comm)

    def boundary(x):
        ax = atol_x if atol_x is not None else atol
        ay = atol_y if atol_y is not None else atol
        az = atol_z if atol_z is not None else atol

        xmin = close_func(x[0], x_min_all, atol=ax)
        xmax = close_func(x[0], x_max_all, atol=ax)
        ymin = close_func(x[1], y_min_all, atol=ay)
        ymax = close_func(x[1], y_max_all, atol=ay)
        
        if domain.geometry.dim == 3:
            zmin = close_func(x[2], z_min_all, atol=az)
            zmax = close_func(x[2], z_max_all, atol=az)
            boundaries = [xmin, xmax, ymin, ymax, zmin, zmax]
        else:
            boundaries = [xmin, xmax, ymin, ymax]

        return reduce(np.logical_or, boundaries)

    return boundary

def update_e_p(u,e_p_n,alpha_n,sig_y,hard,mu):
    e_np1 = ufl.dev(assemble_3D_representation_of_plane_strain_eps(u))
    s_tr = 2.0*mu*(e_np1-e_p_n)
        
    norm_s_tr = ufl.sqrt(ufl.inner(s_tr,s_tr))
        
    f_tr = f_tr_func(u,e_p_n,alpha_n,sig_y,hard,mu)
    dgamma = f_tr / (2.0*(mu+hard/3))
    N_np1 = s_tr / norm_s_tr
    eps_p_np1 = ufl.conditional(ufl.le(f_tr,0.0),e_p_n,e_p_n+dgamma*N_np1)
    return eps_p_np1

def update_alpha(u,e_p_n,alpha_n,sig_y,hard,mu):
    f_tr = f_tr_func(u,e_p_n,alpha_n,sig_y,hard,mu)
    dgamma = f_tr / (2.0*(mu+hard/3))
    alpha_np1 = ufl.conditional(ufl.le(f_tr,0.0),alpha_n,alpha_n+np.sqrt(2/3)*dgamma)
    return alpha_np1

def update_e_p_n_and_alpha_arrays(u,e_p_11_n_tmp,e_p_22_n_tmp,e_p_12_n_tmp,e_p_33_n_tmp,
                           e_p_11_n,e_p_22_n,e_p_12_n,e_p_33_n,
                           alpha_tmp,alpha_n,domain,cells,quadrature_points,sig_y,hard,mu):
    e_p_11_n_tmp.x.array[:] = e_p_11_n.x.array[:]
    e_p_22_n_tmp.x.array[:] = e_p_22_n.x.array[:]
    e_p_12_n_tmp.x.array[:] = e_p_12_n.x.array[:]
    e_p_33_n_tmp.x.array[:] = e_p_33_n.x.array[:]
    e_p_n_tmp = ufl.as_tensor([[e_p_11_n_tmp, e_p_12_n_tmp, 0.0], 
                               [e_p_12_n_tmp, e_p_22_n_tmp, 0.0],
                               [0.0         ,          0.0, e_p_33_n_tmp]])
    
    alpha_tmp.x.array[:] = alpha_n.x.array[:]
    alpha_expr = update_alpha(u,e_p_n=e_p_n_tmp,alpha_n=alpha_n,sig_y=sig_y.value,hard=hard.value,mu=mu)
    alpha_n.x.array[:] = interpolate_quadrature(domain, cells, quadrature_points,alpha_expr)
    
    
    
    e_p_np1_expr = update_e_p(u,e_p_n=e_p_n_tmp,alpha_n=alpha_tmp,sig_y=sig_y.value,hard=hard.value,mu=mu)
    
    e_p_11_expr = e_p_np1_expr[0,0]
    e_p_11_n.x.array[:] = interpolate_quadrature(domain, cells, quadrature_points,e_p_11_expr)
    
    e_p_22_expr = e_p_np1_expr[1,1]
    e_p_22_n.x.array[:] = interpolate_quadrature(domain, cells, quadrature_points,e_p_22_expr)
    
    e_p_12_expr = e_p_np1_expr[0,1]
    e_p_12_n.x.array[:] = interpolate_quadrature(domain, cells, quadrature_points,e_p_12_expr)

    e_p_33_expr = e_p_np1_expr[2,2]
    e_p_33_n.x.array[:] = interpolate_quadrature(domain, cells, quadrature_points,e_p_33_expr)

def reaction_force(sigma_func, n: ufl.FacetNormal, ds: ufl.Measure, comm: MPI.Intercomm,):
    if n.ufl_shape[0] == 3:
        Rx = dlfx.fem.assemble_scalar(dlfx.fem.form(ufl.dot(sigma_func,n)[0] * ds))
        Ry = dlfx.fem.assemble_scalar(dlfx.fem.form(ufl.dot(sigma_func,n)[1] * ds))
        Rz = dlfx.fem.assemble_scalar(dlfx.fem.form(ufl.dot(sigma_func,n)[2] * ds))
        return [comm.allreduce(Rx,MPI.SUM), comm.allreduce(Ry,MPI.SUM), comm.allreduce(Rz,MPI.SUM)]
    elif n.ufl_shape[0] == 2:
        Rx = dlfx.fem.assemble_scalar(dlfx.fem.form(ufl.dot(sigma_func,n)[0] * ds))
        Ry = dlfx.fem.assemble_scalar(dlfx.fem.form(ufl.dot(sigma_func,n)[1] * ds))
        return [comm.allreduce(Rx,MPI.SUM), comm.allreduce(Ry,MPI.SUM)]
    else:
        raise NotImplementedError(f"dim {sigma_func.function_space.mesh.geometry.dim} not implemented")
    
def work_increment_external_forces(sigma_func, u: dlfx.fem.Function, um1: dlfx.fem.Function, n: ufl.FacetNormal, ds: ufl.Measure, comm: MPI.Intercomm,):
    du = u-um1
    t = ufl.dot(sigma_func,n)
    dW = dlfx.fem.assemble_scalar(dlfx.fem.form(ufl.inner(t,du)*ds))
    return comm.allreduce(dW,MPI.SUM)

def write_to_newton_logfile(logfile_path: str, t: float, dt: float, iters: int):
    logfile = open(logfile_path, 'a')
    logfile.write(str(t)+'  '+str(dt)+'  '+str(iters)+'\n')
    logfile.close()
    return True

def write_to_graphs_output_file(output_file_path: str, *args):
    logfile = open(output_file_path, 'a')
    formatted_data = ' '.join(['{:.4e}' for _ in range(len(args))])
    logfile.write(formatted_data.format(*args) + '\n')
    logfile.close()
    return True

def write_vector_fields(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, vector_fields_as_functions, vector_field_names, outputfile_xdmf_path: str, t: float):
    Ve = basix.ufl.element("P", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
    V = dlfx.fem.functionspace(domain, Ve)
    xdmf_out = dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a')
    for n  in range(0,len(vector_fields_as_functions)):
            vector_field_function = vector_fields_as_functions[n]
            vector_field_name = vector_field_names[n]
            vector_field_expression = dlfx.fem.Expression(vector_field_function, 
                                                        V.element.interpolation_points())
            out_vector_field = dlfx.fem.Function(V)
            out_vector_field.interpolate(vector_field_expression)
            out_vector_field.name = vector_field_name
            
            xdmf_out.write_function(out_vector_field,t)
    xdmf_out.close()
    
def write_tensor_fields(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, tensor_fields_as_functions, tensor_field_names, outputfile_xdmf_path: str, t: float):
    dim = domain.topology.dim
    TEN = dlfx.fem.functionspace(domain, ("DP", 0, (dim, dim)))
    with dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a') as xdmf_out:
        for n  in range(0,len(tensor_fields_as_functions)):
            tensor_field_function = tensor_fields_as_functions[n]
            #tensor_field_function.function_space
            tensor_field_name = tensor_field_names[n]
            tensor_field_expression = dlfx.fem.Expression(tensor_field_function, 
                                                         TEN.element.interpolation_points())
            out_tensor_field = dlfx.fem.Function(TEN) 
            out_tensor_field.interpolate(tensor_field_expression)
            out_tensor_field.name = tensor_field_name
            
            xdmf_out.write_function(out_tensor_field,t)
            
def print_runtime(runtime: float):
    print('') 
    print('-----------------------------')
    print('elapsed time:', runtime)
    print('-----------------------------')
    print('') 
    sys.stdout.flush()
    return True

def write_runtime_to_newton_logfile(logfile_path: str, runtime: float):
    logfile = open(logfile_path, 'a')
    logfile.write('# \n')
    logfile.write('# elapsed time:  '+str(runtime)+'\n')
    logfile.write('# \n')
    logfile.close()
    return True

def print_graphs_plot(output_file_path, print_path, legend_labels=None, default_label="Column", filename=None):
    def read_from_graphs_output_file(output_file_path):
        with open(output_file_path, 'r') as file:
            data = [line.strip().split() for line in file.readlines() if not line.startswith('#')]
        return data
    
    data = read_from_graphs_output_file(output_file_path)
    
    t_values = []
    column_values = [[] for _ in range(len(data[0]) - 1)]  # Initialize lists for column values
    
    for line in data:
        t_values.append(float(line[0]))
        for i in range(1, len(line)):  # Start from index 1 to skip time column
            column_values[i - 1].append(float(line[i]))

    if legend_labels is None:
        legend_labels = [default_label + str(i + 1) for i in range(len(column_values))]
    elif len(legend_labels) < len(column_values):
        legend_labels += [default_label + str(i + 1) for i in range(len(legend_labels), len(column_values))]

    for i, values in enumerate(column_values):
        plt.plot(t_values, values, label=legend_labels[i])

    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title('Columns vs Time')
    plt.legend()

    # Use provided filename or default to 'graphs.png'
    save_filename = filename if filename else 'graphs.png'
    plt.savefig(f"{print_path}/{save_filename}")
    plt.close()
    
def print_timestep_overview(iters: int, converged: bool, restart_solution: bool):
    print('-----------------------------')
    print(' No. of iterations: ', iters)
    print(' Converged:         ', converged)
    print(' Restarting:        ', restart_solution)
    print('-----------------------------')
    sys.stdout.flush()
    return True
    
def print_total_dofs(w, comm, rank):
    num_dofs = np.shape(w.x.array[:])[0]
    comm.Barrier()
    num_dofs_all = comm.allreduce(num_dofs, op=MPI.SUM)
    comm.Barrier()
    if rank == 0:
        print('solving fem problem with', num_dofs_all,'dofs ...')
        sys.stdout.flush()
        
def get_solver(w, comm, max_iters, Res, dResdw, bcs):
    if dResdw is not None:
        problem = NonlinearProblem(Res, w, bcs, dResdw)
    else:
        #problem = NonlinearProblem(Res, w, bcs)
        problem = NonlinearProblem(Res, w, bcs)
        
    solver = NewtonSolver(comm, problem)
    solver.report = True
    solver.max_it = max_iters

    # ksp = solver.krylov_solver
    # opts = PETSc.Options()
    # option_prefix = ksp.getOptionsPrefix()
    # opts[f"{option_prefix}ksp_type"] = "fcg"
    # # opts[f"{option_prefix}pc_type"] = "lu"
    # opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    # ksp.setFromOptions()
    
    if comm.Get_rank() == 0:
        ksp = solver.krylov_solver
        print("Default KSP Type:", ksp.getType())
        print("Default PC Type:", ksp.getPC().getType())
    
    return solver, problem

def default_hook_t(t):
    return
def default_hook_tdt(t,dt):
    return
def default_hook():
    return
def default_hook_dt(dt):
    return
def default_hook_all(t,dt,iters):
    return

def print_no_convergence(dt: float):
    print('-----------------------------')
    print('!!! NO CONVERGENCE => dt: ', dt)
    print('-----------------------------')
    sys.stdout.flush()
    return True

def print_increasing_dt(dt: float):
    print('-----------------------------')
    print('!!! Increasing dt to: ', dt)
    print('-----------------------------')
    sys.stdout.flush()
    return True        

def solve_with_newton_adaptive_time_stepping(domain: dlfx.mesh.Mesh,
                                             w: dlfx.fem.Function, 
                                             Tend: float,
                                             dt: dlfx.fem.Constant,
                                             before_first_timestep_hook: Callable = default_hook,
                                             after_last_timestep_hook: Callable = default_hook,
                                             before_each_timestep_hook: Callable = default_hook_tdt, 
                                             get_residuum_and_gateaux: Callable = default_hook_dt,
                                             get_bcs: Callable = default_hook_t,
                                             after_timestep_success_hook: Callable = default_hook_all,
                                             after_timestep_restart_hook: Callable = default_hook_all,
                                             comm: MPI.Intercomm = MPI.COMM_WORLD,
                                             print_bool = False,
                                             solver : NewtonSolver = None,
                                             t: dlfx.fem.Constant = None,
                                             dt_max: dlfx.fem.Constant = None,
                                             dt_never_scale_up: bool = False,
                                             trestart: dlfx.fem.Constant = None,
                                             max_iters = 8,
                                             min_iters = 4, 
                                             arc_length=False,
                                             λ_arc_length=None,
                                             arc_length_ds=0.01):
    rank = comm.Get_rank()
    
    if print_bool:
        print_total_dofs(w, comm, rank)
    
    # time stepping
    # max_iters = 8
    # min_iters = 4
    dt_scale_down = 0.5
    dt_scale_up = 2.0
    
    if not dt_max is None:
        if dt.value >= dt_max.value:
            dt.value = dt_max.value
    
    # t = 0
    if t is None:
        t = dlfx.fem.Constant(domain, 0.0)
    if trestart is None:
        trestart = dlfx.fem.Constant(domain, 0.0)
    # delta_t = dlfx.fem.Constant(domain, dt)
    # dtt = dt.value 

    before_first_timestep_hook()

    # nn = 0
    choose_default_solver_each_time_step = False
    if solver is None:
            choose_default_solver_each_time_step = True
    
    if arc_length:
        raise NotImplementedError("Not implemented")
    # gc.set_debug(gc.DEBUG_LEAK)
    while t.value < Tend:
        # if comm.Get_rank() == 0:
        #     print(f"time: {t.value} Tend: {Tend}")
        # if choose_default_solver_each_time_step and solver is not None:
        #     comm.Barrier()
        #     del solver
        #     gc.collect()
        #     comm.Barrier()
        # dt.value = dtt

        before_each_timestep_hook(t.value,dt.value)
            
        [Res, dResdw] = get_residuum_and_gateaux(dt)
        
        bcs = get_bcs(t.value)
        
        if choose_default_solver_each_time_step:
            if comm.Get_rank() == 0:
                print(f"NO SOLVER PROVIDED. Default solver created each time step")
            solver, problem = get_solver(w, comm, max_iters, Res, dResdw, bcs)
            
        
        # control adaptive time adjustment
        # restart_solution = False
        converged = False
        iters = 0 # iters always needs to be defined
        try:
            if arc_length:  
                raise NotImplementedError("Not implemented")
            else:
                (iters, converged) = solver.solve(w)
        except RuntimeError as e:
            if comm.Get_rank() == 0:
                print(e)
            if not arc_length:
                dt.value = dt_scale_down*dt.value
                # restart_solution = True
                if rank == 0 and print_bool:
                    print_no_convergence(dt.value)
                
        
        if converged:
            after_timestep_success_hook(t.value,dt.value,iters)
        
        if not arc_length:
            if converged and iters < min_iters and t.value > np.finfo(float).eps and iters > 0:
                
                if not dt_never_scale_up:
                    if dt_max is None:
                        dt.value = dt_scale_up*dt.value
                        if rank == 0 and print_bool:
                            print_increasing_dt(dt.value)
                    else:
                        if not (dt_scale_up*dt.value > dt_max.value): 
                            dt.value = dt_scale_up*dt.value
                            if rank == 0 and print_bool:
                                print_increasing_dt(dt.value)
                        else:
                            dt.value = dt_max.value
                    
        restart_solution = False
        if arc_length:
            if converged:
                # do nothing arc length handles time update?
                a=1
            else:
                trestart.value = t.value
        else:
            if converged:
                #after_timestep_success_hook(t.value,dt.value,iters)
                trestart.value = t.value
                t.value = t.value+dt.value
            else:
                restart_solution = True
                after_timestep_restart_hook(t.value,dt.value,iters)
                t.value = trestart.value+dt.value
        
        if rank == 0 and print_bool:    
            print_timestep_overview(iters,converged,restart_solution) 
    after_last_timestep_hook() 
    
    
### START MAIN SCRIPT    

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
logfile_path = logfile_full_path(script_path,script_name_without_extension)
outputfile_graph_path = outputfile_graph_full_path(script_path,script_name_without_extension)
outputfile_xdmf_path = outputfile_xdmf_full_path(script_path,script_name_without_extension)
parameter_path = os.path.join(script_path,"parameters.txt")

# set MPI environment
comm, rank, size = set_mpi()
print_mpi_status(rank, size)



N=50
domain = dlfx.mesh.create_unit_square(comm, N, N, cell_type=dlfx.mesh.CellType.quadrilateral)
    
dim = domain.topology.dim
mpi_print('spatial dimensions: '+str(dim), rank)
    


# Material definition ##################################################
micro_material_marker = 1
effective_material_marker = 0


# Simulation parameters ####
dt_start = 0.01
dt_max_in_critical_area = dt_start
dt_global = dlfx.fem.Constant(domain, dt_start)
t_global = dlfx.fem.Constant(domain,0.0)
trestart_global = dlfx.fem.Constant(domain,0.0)
Tend = 3.0
dt_global.value = dt_max_in_critical_area
dt_max = dlfx.fem.Constant(domain,dt_max_in_critical_area)



la = dlfx.fem.Constant(domain, 1.0)
mu = dlfx.fem.Constant(domain, 1.0)

sig_y = dlfx.fem.Constant(domain, 1.0)
hard = dlfx.fem.Constant(domain, 0.6)

# Function space and FE functions ########################################################
Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1) # displacements
V = dlfx.fem.FunctionSpace(domain, Ve)


# define solution, restart, trial and test space
u =  dlfx.fem.Function(V)
urestart =  dlfx.fem.Function(V)
um1 =  dlfx.fem.Function(V) # trial space
um1.x.array[:] = np.zeros_like(um1.x.array[:])
du = ufl.TestFunction(V)
ddu = ufl.TrialFunction(V)

deg_quad = 1  # quadrature degree for internal state variable representation
gdim = 2
H,alpha_n,alpha_tmp, e_p_11_n, e_p_22_n, e_p_12_n, e_p_33_n, e_p_11_n_tmp, e_p_22_n_tmp, e_p_12_n_tmp, e_p_33_n_tmp = alex.plasticity.define_internal_state_variables_basix_2D(gdim, domain, deg_quad,quad_scheme="default")
dx = define_custom_integration_measure_that_matches_quadrature_degree_and_scheme(domain, deg_quad, "default")
quadrature_points, cells = get_quadraturepoints_and_cells_for_inter_polation_at_gauss_points(domain, deg_quad)
H.x.array[:] = np.zeros_like(H.x.array[:])
alpha_n.x.array[:] = np.zeros_like(alpha_n.x.array[:])
alpha_tmp.x.array[:] = np.zeros_like(alpha_tmp.x.array[:])
e_p_11_n.x.array[:] = np.zeros_like(e_p_11_n.x.array[:])
e_p_22_n.x.array[:] = np.zeros_like(e_p_22_n.x.array[:])
e_p_12_n.x.array[:] = np.zeros_like(e_p_12_n.x.array[:])
e_p_33_n.x.array[:] = np.zeros_like(e_p_33_n.x.array[:])
e_p_11_n_tmp.x.array[:] = np.zeros_like(e_p_11_n_tmp.x.array[:])
e_p_22_n_tmp.x.array[:] = np.zeros_like(e_p_22_n_tmp.x.array[:])
e_p_12_n_tmp.x.array[:] = np.zeros_like(e_p_12_n_tmp.x.array[:])
e_p_33_n_tmp.x.array[:] = np.zeros_like(e_p_33_n_tmp.x.array[:])




## define boundary conditions crack
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)


e_p_n_3D = ufl.as_tensor([[e_p_11_n, e_p_12_n, 0.0], 
                          [e_p_12_n, e_p_22_n, 0.0],
                          [0.0, 0.0, e_p_33_n]])

plasticityProblem = Plasticity_incremental_2D(sig_y=sig_y.value, hard=hard.value,alpha_n=alpha_n,e_p_n=e_p_n_3D,H=H)

# pf.StaticPhaseFieldProblem2D_incremental_plasticity(degradationFunction=pf.degrad_cubic,
#                                                    psisurf=pf.psisurf_from_function,dx=dx, sig_y=sig_y.value, hard=hard.value,alpha_n=alpha_n,e_p_n=e_p_n_3D,H=H)
timer = dlfx.common.Timer()
def before_first_time_step():
    timer.start()
    urestart.x.array[:] = um1.x.array[:]
    # prepare newton-log-file
    if rank == 0:
        prepare_newton_logfile(logfile_path)
        prepare_graphs_output_file(outputfile_graph_path)
    # prepare xdmf output 
    write_meshoutputfile(domain, outputfile_xdmf_path, comm)

def before_each_time_step(t,dt):
    # report solution status
    if rank == 0:
        print_time_and_dt(t,dt)



def get_residuum_and_gateaux(delta_t: dlfx.fem.Constant):
    [Res, dResdw] = plasticityProblem.prep_newton(u=u,um1=um1,du=du,ddu=ddu,lam=la, mu=mu) 
    return [Res, dResdw]



atol=0.0 # for selection of boundary


def all(x):
        return np.full_like(x[0],True)
    


u_D = dlfx.fem.Function(V) # for dirichlet BCs
def top_displacement():    
    u_y = ufl.conditional(ufl.le(t_global,ufl.as_ufl(1.0)),t_global,ufl.as_ufl(1.0-(t_global-1.0)))
    u_x = ufl.as_ufl(0.0)
    return ufl.as_vector([u_x, u_y]) # only 2 components in 2D

bc_top_expression = dlfx.fem.Expression(top_displacement(),V.element.interpolation_points())

boundary_top_bc = get_top_boundary_of_box_as_function(domain,comm,atol=atol*0.0)
facets_at_boundary = dlfx.mesh.locate_entities_boundary(domain, fdim, boundary_top_bc)
dofs_at_boundary = dlfx.fem.locate_dofs_topological(V, fdim, facets_at_boundary) 

def get_bcs(t):
    
    u_D.interpolate(bc_top_expression)
    bc_top : dlfx.fem.DirichletBC = dlfx.fem.dirichletbc(u_D,dofs_at_boundary)
     
    bc_bottom_y = define_dirichlet_bc_from_value(domain,0.0,1,get_bottom_boundary_of_box_as_function(domain,comm,atol=atol),V,-1)
    bc_bottom_x = define_dirichlet_bc_from_value(domain,0.0,0,get_bottom_boundary_of_box_as_function(domain,comm,atol=atol),V,-1)

    bcs = [bc_top,bc_bottom_y,bc_bottom_x]
    return bcs


n = ufl.FacetNormal(domain)
external_surface_tag = 5
external_surface_tags = tag_part_of_boundary(domain,get_boundary_of_box_as_function(domain, comm,atol=atol*0.0),external_surface_tag)
ds = ufl.Measure('ds', domain=domain, subdomain_data=external_surface_tags,metadata={"quadrature_degree": deg_quad, "quadrature_scheme": "default"})

top_surface_tag = 9
top_surface_tags = tag_part_of_boundary(domain,get_top_boundary_of_box_as_function(domain, comm,atol=atol*0.0),top_surface_tag)
ds_top_tagged = ufl.Measure('ds', domain=domain, subdomain_data=top_surface_tags,metadata={"quadrature_degree": deg_quad, "quadrature_scheme": "default"})

Work = dlfx.fem.Constant(domain,0.0)

success_timestep_counter = dlfx.fem.Constant(domain,0.0)
postprocessing_interval = dlfx.fem.Constant(domain,20.0)
TEN = dlfx.fem.functionspace(domain, ("DP", 0, (dim, dim)))
def after_timestep_success(t,dt,iters):
    
    delta_u = u - um1  
    H_expr = plasticityProblem.update_H(u,delta_u=delta_u,lam=la,mu=mu)
    H.x.array[:] = interpolate_quadrature(domain, cells, quadrature_points,H_expr)
    
    
    update_e_p_n_and_alpha_arrays(u,e_p_11_n_tmp,e_p_22_n_tmp,e_p_12_n_tmp,e_p_33_n_tmp,
                           e_p_11_n,e_p_22_n,e_p_12_n,e_p_33_n,
                           alpha_tmp,alpha_n,domain,cells,quadrature_points,sig_y,hard,mu)
    
    
    # update u from Δu
    
    sigma = plasticityProblem.sigma(u,la,mu)
    tensor_field_expression = dlfx.fem.Expression(sigma, 
                                                         TEN.element.interpolation_points())
    tensor_field_name = "sigma"
    sigma_interpolated = dlfx.fem.Function(TEN) 
    sigma_interpolated.interpolate(tensor_field_expression)
    sigma_interpolated.name = tensor_field_name
    
    #pp.write_tensor_fields(domain,comm,[sigma],["sigma"],outputfile_xdmf_path,t)
    Rx_top, Ry_top = reaction_force(sigma_interpolated,n=n,ds=ds_top_tagged(top_surface_tag),comm=comm)
    

    dW = work_increment_external_forces(sigma_interpolated,u,um1,n,ds,comm=comm)
    Work.value = Work.value + dW
    
    
    E_el = plasticityProblem.get_E_el_global(u,la,mu,dx=ufl.dx,comm=comm)
    
    # write to newton-log-file
    if rank == 0:
        write_to_newton_logfile(logfile_path,t,dt,iters)
    
    
    if rank == 0:
        if (t>1):
            u_y = 1.0-(t-1.0)
        else:
            u_y = t
        write_to_graphs_output_file(outputfile_graph_path,t,  Ry_top,u_y)


    # update
    um1.x.array[:] = u.x.array[:]
    urestart.x.array[:] = u.x.array[:]
    # break out of loop if no postprocessing required
    success_timestep_counter.value = success_timestep_counter.value + 1.0
    # break out of loop if no postprocessing required
    if not int(success_timestep_counter.value) % int(postprocessing_interval.value) == 0: 
        return 
    
    write_vector_fields(domain,comm,[u],["u"],outputfile_xdmf_path,t)
    write_tensor_fields(domain,comm,[sigma_interpolated],["sigma"],outputfile_xdmf_path,t)

def after_timestep_restart(t,dt,iters):
    u.x.array[:] = urestart.x.array[:]

def after_last_timestep():
    # stopwatch stop
    timer.stop()

    # report runtime to screen
    if rank == 0:
        runtime = timer.elapsed()
        print_runtime(runtime)
        write_runtime_to_newton_logfile(logfile_path,runtime)
        print_graphs_plot(outputfile_graph_path,script_path,legend_labels=[ "R_y", "u_y"])
        

sol.solve_with_newton_adaptive_time_stepping(
    domain,
    u,
    Tend,
    dt_global,
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
    dt_max=dt_max,
    trestart=trestart_global,
    #max_iters=20
)




# copy relevant files

# Step 1: Create a unique timestamped directory
def create_timestamped_directory(base_dir="."):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory_name = os.path.join(base_dir, f"simulation_{timestamp}")
    os.makedirs(directory_name, exist_ok=True)
    return directory_name

# Step 2: Copy files to the timestamped directory
def copy_files_to_directory(files, target_directory):
    for file in files:
        if os.path.exists(file):
            shutil.copy(file, target_directory)
        else:
            print(f"Warning: File '{file}' does not exist and will not be copied.")

