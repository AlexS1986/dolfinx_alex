#linear displacements
import dolfinx as dlfx
from typing import Callable, Optional
import numpy as np
from mpi4py import MPI
from functools import reduce
import math
from alex.linearelastic import get_nu
import alex.util as ut

def linear_displacements_mixed(mixedFunctionSpace: dlfx.fem.FunctionSpace, 
                               eps_mac: dlfx.fem.Constant,
                               subspace_idx:int = 0):
    w_D = dlfx.fem.Function(mixedFunctionSpace)
    dim = ut.get_dimension_of_function(w_D.sub(subspace_idx))
    for k in range(0, dim):
        w_D.sub(subspace_idx).sub(k).interpolate(lambda x: eps_mac.value[k, 0]*x[0] + eps_mac.value[k, 1]*x[1] + eps_mac.value[k, 2]*x[2] )
        w_D.x.scatter_forward()
    
    # def u_x(x):
    #     return eps_mac.value[0, 0]*x[0] + eps_mac.value[0, 1]*x[1] + eps_mac.value[0, 2]*x[2]
    # w_D.sub(0).sub(0).interpolate(u_x)
    # w_D.sub(subspace_idx).sub(0).interpolate(lambda x: eps_mac.value[0, 0]*x[0] + eps_mac.value[0, 1]*x[1] + eps_mac.value[0, 2]*x[2] )
    # w_D.sub(subspace_idx).x.scatter_forward()
    # w_D.sub(subspace_idx).sub(1).interpolate(lambda x: eps_mac.value[1, 0]*x[0] + eps_mac.value[1, 1]*x[1] + eps_mac.value[1, 2]*x[2] )
    # w_D.sub(subspace_idx).x.scatter_forward()
    # w_D.sub(subspace_idx).sub(2).interpolate(lambda x: eps_mac.value[2, 0]*x[0] + eps_mac.value[2, 1]*x[1] + eps_mac.value[2, 2]*x[2] )
    # w_D.sub(subspace_idx).x.scatter_forward()
    return w_D

def linear_displacements(V: dlfx.fem.FunctionSpace, 
                               eps_mac: dlfx.fem.Constant):
    u_D = dlfx.fem.Function(V)
    dim = ut.get_dimension_of_function(u_D)
    if dim == 3:
        for k in range(0, dim):
            u_D.sub(k).interpolate(lambda x: eps_mac.value[k, 0]*x[0] + eps_mac.value[k, 1]*x[1] + eps_mac.value[k, 2]*x[2] )
            u_D.x.scatter_forward()
    elif dim == 2:
         for k in range(0, dim):
            u_D.sub(k).interpolate(lambda x: eps_mac.value[k, 0]*x[0] + eps_mac.value[k, 1]*x[1] )
            u_D.x.scatter_forward()
        
        
    # def u_x(x):
    #     return eps_mac.value[0, 0]*x[0] + eps_mac.value[0, 1]*x[1] + eps_mac.value[0, 2]*x[2]
    # w_D.sub(0).sub(0).interpolate(u_x)
    # u_D.sub(subspace_idx).sub(0).interpolate(lambda x: eps_mac.value[0, 0]*x[0] + eps_mac.value[0, 1]*x[1] + eps_mac.value[0, 2]*x[2] )
    # u_D.sub(subspace_idx).x.scatter_forward()
    # u_D.sub(subspace_idx).sub(1).interpolate(lambda x: eps_mac.value[1, 0]*x[0] + eps_mac.value[1, 1]*x[1] + eps_mac.value[1, 2]*x[2] )
    # u_D.sub(subspace_idx).x.scatter_forward()
    # u_D.sub(subspace_idx).sub(2).interpolate(lambda x: eps_mac.value[2, 0]*x[0] + eps_mac.value[2, 1]*x[1] + eps_mac.value[2, 2]*x[2] )
    # u_D.sub(subspace_idx).x.scatter_forward()
    return u_D

def define_dirichlet_bc_from_interpolated_function(domain: dlfx.mesh.Mesh,
                                                         desired_value_at_boundary_function: dlfx.fem.Function,
                                                         where_function: Callable,
                                                         functionSpace: dlfx.fem.FunctionSpace,
                                                         subspace_idx: int) -> dlfx.fem.DirichletBC:
    fdim = domain.topology.dim-1
    facets_at_boundary = dlfx.mesh.locate_entities_boundary(domain, fdim, where_function)
    if subspace_idx < 0:
        dofs_at_boundary = dlfx.fem.locate_dofs_topological(functionSpace, fdim, facets_at_boundary)
    else:
        dofs_at_boundary = dlfx.fem.locate_dofs_topological(functionSpace.sub(subspace_idx), fdim, facets_at_boundary)
    bc : dlfx.fem.DirichletBC = dlfx.fem.dirichletbc(desired_value_at_boundary_function,dofs_at_boundary)
    return bc

def define_dirichlet_bc_from_value(domain: dlfx.mesh.Mesh,
                                                         desired_value_at_boundary: float,
                                                         coordinate_idx,
                                                         where_function: Callable,
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
    
    

def get_tagged_subdomain_bounds(domain, cell_tags, tag_value, comm: MPI.Intracomm):
    """
    Create a submesh for a tagged subdomain and compute its global bounding box.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        The parent mesh.
    cell_tags : dolfinx.mesh.MeshTags
        MeshTags object tagging the subdomain.
    tag_value : int
        The tag value identifying the subdomain.
    comm : MPI.Intracomm
        The MPI communicator.

    Returns
    -------
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all : float
        Global min/max coordinates of the subdomain across all ranks.
    """
    # Extract tagged cell indices
    tagged_cells = cell_tags.indices[cell_tags.values == tag_value]

    # Create submesh
    submesh, *_ = dlfx.mesh.create_submesh(domain, 3, tagged_cells)

    # Get submesh coordinates
    coords = submesh.geometry.x

    # Handle case where this rank has no submesh cells
    if coords.size == 0:
        x_min = np.inf
        x_max = -np.inf
        y_min = np.inf
        y_max = -np.inf
        z_min = np.inf
        z_max = -np.inf
    else:
        x_min = np.min(coords[:, 0])
        x_max = np.max(coords[:, 0])
        y_min = np.min(coords[:, 1])
        y_max = np.max(coords[:, 1])
        z_min = np.min(coords[:, 2])
        z_max = np.max(coords[:, 2])

    # Allreduce to get global min/max across all ranks
    x_min_all = comm.allreduce(x_min, op=MPI.MIN)
    x_max_all = comm.allreduce(x_max, op=MPI.MAX)
    y_min_all = comm.allreduce(y_min, op=MPI.MIN)
    y_max_all = comm.allreduce(y_max, op=MPI.MAX)
    z_min_all = comm.allreduce(z_min, op=MPI.MIN)
    z_max_all = comm.allreduce(z_max, op=MPI.MAX)

    return x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all
    
def print_dimensions(x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all, comm: MPI.Intercomm):
    if comm.rank == 0:
        print('x_min, x_max: '+str(x_min_all)+', '+str(x_max_all))
        print('y_min, y_max: '+str(y_min_all)+', '+str(y_max_all))
        print('z_min, z_max: '+str(z_min_all)+', '+str(z_max_all))  

def get_boundary_of_box_as_function(
    domain: dlfx.mesh.Mesh,
    comm: MPI.Intracomm,
    atol: Optional[float] = None,
    atol_x: Optional[float] = None,
    atol_y: Optional[float] = None,
    atol_z: Optional[float] = None
) -> Callable:
    
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


def dont_get_boundary_of_box_as_function(
    domain: dlfx.mesh.Mesh,
    comm: MPI.Intercomm,
    atol: Optional[float] = None,
    atol_x: Optional[float] = None,
    atol_y: Optional[float] = None,
    atol_z: Optional[float] = None
) -> Callable:
    
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = get_dimensions(domain, comm)

    def boundary(x):
        ax = atol_x if atol_x is not None else atol
        ay = atol_y if atol_y is not None else atol
        az = atol_z if atol_z is not None else atol

        lower_x = x[0] > x_min_all + (ax if ax is not None else 0.0)
        upper_x = x[0] < x_max_all - (ax if ax is not None else 0.0)
        lower_y = x[1] > y_min_all + (ay if ay is not None else 0.0)
        upper_y = x[1] < y_max_all - (ay if ay is not None else 0.0)

        boundaries = [lower_x, upper_x, lower_y, upper_y]

        if domain.geometry.dim == 3:
            lower_z = x[2] > z_min_all + (az if az is not None else 0.0)
            upper_z = x[2] < z_max_all - (az if az is not None else 0.0)
            boundaries.extend([lower_z, upper_z])

        return reduce(np.logical_and, boundaries)

    return boundary




def get_frontback_boundary_of_box_as_function(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, atol: float=None) -> Callable:
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = get_dimensions(domain, comm)
    def boundary(x):
        # Only 3D
        zmin = close_func(x[2],z_min_all,atol=atol)
        zmax = close_func(x[2],z_max_all,atol=atol)
        boundaries = [zmin, zmax]
        return reduce(np.logical_or, boundaries)
    return boundary

def get_front_boundary_of_box_as_function(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, atol: float=None) -> Callable:
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = get_dimensions(domain, comm)
    def boundary(x):
        zmax = close_func(x[2],z_max_all,atol=atol)
        boundaries = [zmax]
        return reduce(np.logical_or, boundaries)
    return boundary

def get_back_boundary_of_box_as_function(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, atol: float=None) -> Callable:
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = get_dimensions(domain, comm)
    def boundary(x):
        zmin = close_func(x[2],z_min_all,atol=atol)
        boundaries = [zmin]
        return reduce(np.logical_or, boundaries)
    return boundary

def get_top_boundary_of_box_as_function(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, atol: float=None) -> Callable:
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = get_dimensions(domain, comm)
    def boundary(x):
        ymax = close_func(x[1],y_max_all,atol=atol)
        boundaries = [ymax]
        return reduce(np.logical_or, boundaries)
    return boundary

def get_left_boundary_of_box_as_function(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, atol: float=None) -> Callable:
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = get_dimensions(domain, comm)
    def boundary(x):
        xmin = close_func(x[0],x_min_all,atol=atol)
        boundaries = [xmin]
        return reduce(np.logical_or, boundaries)
    return boundary

def get_right_boundary_of_box_as_function(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, atol: float=None) -> Callable:
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = get_dimensions(domain, comm)
    def boundary(x):
        xmax = close_func(x[0],x_max_all,atol=atol)
        boundaries = [xmax]
        return reduce(np.logical_or, boundaries)
    return boundary

def get_bottom_boundary_of_box_as_function(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, atol: float=None) -> Callable:
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = get_dimensions(domain, comm)
    def boundary(x):
        ymin = close_func(x[1],y_min_all,atol=atol)
        boundaries = [ymin]
        return reduce(np.logical_or, boundaries)
    return boundary

def get_topbottom_boundary_of_box_as_function(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, atol: float=None) -> Callable:
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = get_dimensions(domain, comm)
    def boundary(x):
        ymin = close_func(x[1],y_min_all,atol=atol)
        ymax = close_func(x[1],y_max_all,atol=atol)
        boundaries = [ymin, ymax]
        return reduce(np.logical_or, boundaries)
    return boundary

def get_leftright_boundary_of_box_as_function(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, atol: float=None) -> Callable:
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = get_dimensions(domain, comm)
    def boundary(x):
        xmin = close_func(x[0],x_min_all,atol=atol)
        xmax = close_func(x[0],x_max_all,atol=atol)
        boundaries = [xmin, xmax]
        return reduce(np.logical_or, boundaries)
    return boundary

def get_leftrighttop_boundary_of_box_as_function(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, atol: float=None, epsilon: float = 0.0) -> Callable:
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = get_dimensions(domain, comm)
    def boundary(x):
        xmin = close_func(x[0],x_min_all,atol=atol)
        xmax = close_func(x[0],x_max_all,atol=atol)
        ymax = close_func(x[1],y_max_all,atol=atol)
        boundaries = [xmin, xmax, ymax]
        
        applied_at_height = np.greater_equal(x[1],(y_min_all + 4.0 * epsilon))
        return np.logical_and(reduce(np.logical_or, boundaries),applied_at_height)
    return boundary


def get_2D_boundary_of_box_as_function(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, atol: float=None, epsilon: float = 0.0) -> Callable:
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = get_dimensions(domain, comm)
    def boundary(x):
        xmin = close_func(x[0],x_min_all,atol=atol)
        xmax = close_func(x[0],x_max_all,atol=atol)
        ymax = close_func(x[1],y_max_all,atol=atol)
        ymin = close_func(x[1],y_min_all,atol=atol)
        boundaries = [xmin, xmax, ymax,ymin]
        
        y_middle = (y_max_all+y_min_all) / 2.0
        rangey = 4.0 * epsilon
        applied_at_height = np.logical_or(np.greater_equal(x[1],(y_middle + rangey)), np.less_equal(x[1],(y_middle - rangey)) )
        return np.logical_and(reduce(np.logical_or, boundaries),applied_at_height)
    return boundary

def get_corner_of_box_as_function(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm) -> Callable:
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = get_dimensions(domain, comm)
    def boundary(x):
        xmin = np.isclose(x[0],x_min_all)
        xmax = np.isclose(x[0],x_max_all)
        ymin = np.isclose(x[1],y_min_all)
        ymax = np.isclose(x[1],y_max_all)
        if domain.geometry.dim == 3:
            zmin = np.isclose(x[2],z_min_all)
            zmax = np.isclose(x[2],z_max_all)
            boundaries = [xmin, ymin, zmin]
        elif domain.geometry.dim == 2: #2D
            boundaries = [xmin, ymin]
        else:
            raise NotImplementedError()
        return reduce(np.logical_and, boundaries)
    return boundary


def get_boundary_for_surfing_boundary_condition_at_box_as_function(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, excluded_where_function: Callable, atol: float) -> Callable:
    
    # TODO total boundary or only apply at top and bottom?
    # total_boundary = get_boundary_of_box_as_function(domain, comm,atol=atol)
    total_boundary = get_topbottom_boundary_of_box_as_function(domain, comm,atol=atol)
    
  
    
    def boundary(x):
        return np.logical_and(total_boundary(x), np.logical_not(excluded_where_function(x)))
    return boundary

# def get_boundary_for_surfing_boundary_condition_2D(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, atol: float, epsilon: float) -> Callable:
    
#     # TODO total boundary or only apply at top and bottom?
#     # total_boundary = get_boundary_of_box_as_function(domain, comm,atol=atol)
#     total_boundary =   get_boundary_of_box_as_function(domain,comm,atol=atol)
#     # get_topbottom_boundary_of_box_as_function(domain, comm,atol=atol)
#     x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = get_dimensions(domain, comm)
#     def crack_boundary_where(x):
#         # x_range = x[0] < xtip + 3.0 * epsilon TODO is this necessary
#         x_range = np.isclose(x[0],x_min_all,atol=0.0001*epsilon)
#         y_range = np.isclose(x[1],(y_max_all - y_min_all)/2.0,atol=2.0*epsilon)
#         excluded = np.logical_and(x_range, y_range)
#         return excluded
    
#     def boundary(x):
#         return np.logical_and(total_boundary(x), np.logical_not(crack_boundary_where(x)))
#     return boundary
    

def surfing_boundary_conditions(w_D: dlfx.fem.Function, K1: dlfx.fem.Constant, xK1: dlfx.fem.Constant, lam: dlfx.fem.Constant, mu: dlfx.fem.Constant, subspace_index: int =0) -> dlfx.fem.Function:
    def get_polar_coordinates(x):
        delta_x = x[0] - xK1.value[0]
        delta_y = x[1] - xK1.value[1]
        r = np.hypot(delta_x, delta_y)
        theta = np.arctan2(delta_y, delta_x)
        return r, theta
    
    nu = get_nu(lam=lam.value, mu=mu.value)  
    def u_x(x):
        r, theta = get_polar_coordinates(x)
        u_x = K1.value / (2.0 * mu.value * math.sqrt(2.0 * math.pi)) * np.sqrt(r) * (3.0 - 4.0 * nu  -np.cos(theta)) * np.cos(0.5*theta)
        return u_x
        
    def u_y(x):
        r, theta = get_polar_coordinates(x)
        
        u_y = K1.value / (2.0 * mu.value * math.sqrt(2.0 * math.pi)) * np.sqrt(r) * (3.0 - 4.0 * nu  -np.cos(theta)) * np.sin(0.5*theta)
        return u_y
        
    def u_z(x):
        return 0.0 * x[2]
    
    if subspace_index >= 0:      
        # w_D = dlfx.fem.Function(functionSpace) # TODO do not create a fem.Function in every time step
        w_D.sub(subspace_index).sub(0).interpolate(u_x)
        w_D.sub(subspace_index).x.scatter_forward()
        w_D.sub(subspace_index).sub(1).interpolate(u_y)
        w_D.sub(subspace_index).x.scatter_forward()
        if w_D.function_space.mesh.geometry.dim == 3: # only in 3D
            w_D.sub(subspace_index).sub(2).interpolate(u_z)
            w_D.sub(subspace_index).x.scatter_forward()
        return w_D 
    else:
        w_D.sub(0).interpolate(u_x)
        w_D.x.scatter_forward()
        w_D.sub(1).interpolate(u_y)
        w_D.x.scatter_forward()
        if w_D.function_space.mesh.geometry.dim == 3: # only in 3D
            w_D.sub(2).interpolate(u_z)
            w_D.x.scatter_forward()
        return w_D 
          

def get_total_surfing_boundary_condition_at_box(domain: dlfx.mesh.Mesh, 
                                                               comm: MPI.Intercomm,
                                                               functionSpace: dlfx.fem.FunctionSpace,
                                                               subspace_idx: int,
                                                               K1: dlfx.fem.Constant,
                                                               xK1: dlfx.fem.Constant,
                                                               lam: dlfx.fem.Constant,
                                                               mu: dlfx.fem.Constant,
                                                               epsilon: float,
                                                               atol = None,
                                                               w_D: dlfx.fem.Function = None):
    
    if w_D is None:
        w_D = dlfx.fem.Function(functionSpace)
    w_D = surfing_boundary_conditions(w_D,K1,xK1,lam,mu,subspace_index=subspace_idx)
    
    '''
        only if crack extends in x direction and starts at xmin at y = (y_max-y_min)/2
    '''
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = get_dimensions(domain, comm)
    def crack_boundary_where(x):
        xtip = xK1.value[0] 
        # x_range = x[0] < xtip + 3.0 * epsilon TODO is this necessary
        x_range = np.isclose(x[0],x_min_all,atol=3.0*epsilon)
        y_range = np.isclose(x[1],(y_max_all - y_min_all)/2.0,atol=3.0*epsilon)
        excluded = np.logical_and(x_range, y_range)
        return excluded
    
    def nothing_excluded(x):
        np.full_like(x[0],False)
    
    
    where = get_boundary_for_surfing_boundary_condition_at_box_as_function(domain,comm,excluded_where_function=nothing_excluded, atol=atol)
    bcs = []
    
    
    bcs.append(define_dirichlet_bc_from_interpolated_function(domain, w_D, where, functionSpace,subspace_idx))
    
    # set displacement perpendicular to front and back faces to zero ~plane strain, only 3D
    if functionSpace.mesh.geometry.dim == 3:
        bcs.append(define_dirichlet_bc_from_value(domain=domain,
                                             desired_value_at_boundary=0.0,
                                             coordinate_idx=2,
                                             where_function=get_frontback_boundary_of_box_as_function(domain,comm,atol=atol),
                                             functionSpace=functionSpace,
                                             subspace_idx=subspace_idx))
    return bcs
    
def close_func(x,value,atol):
        if atol:
            return np.isclose(x,value,atol=atol)
        else:
            return np.isclose(x,value)
        
def away_func(x,value,atol):
        if atol:
            reduce(np.logical_not, [np.isclose(x,value,atol=atol)])
        else:
            return reduce(np.logical_not,[np.isclose(x,value)])
    

def get_total_linear_displacement_boundary_condition_at_box(domain: dlfx.mesh.Mesh, 
                                                               comm: MPI.Intercomm,
                                                               functionSpace: dlfx.fem.FunctionSpace,
                                                               eps_mac: dlfx.fem.Constant,
                                                               subspace_idx: int = -1,
                                                               atol : float = None 
                                                               ):
    
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = get_dimensions(domain, comm)
    

    
    # define top boundary
    def top(x):
        return close_func(x[1],y_max_all,atol)
        # return np.isclose(x[1], y_max_all)

    # define bottom boundary
    def bottom(x):
        return close_func(x[1],y_min_all,atol)
        # return np.isclose(x[1], y_min_all)

    def left(x):
        return close_func(x[0],x_min_all,atol)
        # return np.isclose(x[0], x_min_all)

    def right(x):
        return close_func(x[0],x_max_all,atol)
        # return np.isclose(x[0], x_max_all)

    def front(x):
        return close_func(x[2], z_max_all,atol)

    def back(x):
        return close_func(x[2], z_min_all,atol)
    
    bcs = []
    if subspace_idx < 0:
        w_D = linear_displacements(V=functionSpace,eps_mac=eps_mac)
    else:
        w_D = linear_displacements_mixed(functionSpace, subspace_idx=subspace_idx, eps_mac=eps_mac)
        
    for where_function in [top, bottom,left, right, front, back]:
        bcs.append(define_dirichlet_bc_from_interpolated_function(domain,w_D,where_function,functionSpace,subspace_idx))
    
    return bcs

def get_total_linear_displacement_boundary_condition_at_box_for_incremental_formulation(domain: dlfx.mesh.Mesh, 
                                                               w_n: dlfx.fem.Function,
                                                               comm: MPI.Intercomm,
                                                               functionSpace: dlfx.fem.FunctionSpace,
                                                               eps_mac: dlfx.fem.Constant,
                                                               subspace_idx: int = -1,
                                                               atol : float = None 
                                                               ):
    
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = get_dimensions(domain, comm)
    

    
    # define top boundary
    def top(x):
        return close_func(x[1],y_max_all,atol)
        # return np.isclose(x[1], y_max_all)

    # define bottom boundary
    def bottom(x):
        return close_func(x[1],y_min_all,atol)
        # return np.isclose(x[1], y_min_all)

    def left(x):
        return close_func(x[0],x_min_all,atol)
        # return np.isclose(x[0], x_min_all)

    def right(x):
        return close_func(x[0],x_max_all,atol)
        # return np.isclose(x[0], x_max_all)

    def front(x):
        return close_func(x[2], z_max_all,atol)

    def back(x):
        return close_func(x[2], z_min_all,atol)
    
    bcs = []
    if subspace_idx < 0:
        dw_D = linear_displacements(V=functionSpace,eps_mac=eps_mac)
        dw_D.x.array[:] = dw_D.x.array[:] - w_n.x.array[:]
    else:
        dw_D = linear_displacements_mixed(functionSpace, subspace_idx=subspace_idx, eps_mac=eps_mac)
        dw_D.x.array[:] = dw_D.x.array[:] - w_n.x.array[:]
        
    dw_D.x.scatter_forward()
    for where_function in [top, bottom,left, right, front, back]:
        bcs.append(define_dirichlet_bc_from_interpolated_function(domain,dw_D,where_function,functionSpace,subspace_idx))
    
    return bcs




        
    
    
    
    
    