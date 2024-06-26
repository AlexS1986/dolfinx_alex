import alex.linearelastic
import alex.phasefield
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

def run_simulation(sig_mac_param, comm: MPI.Intercomm):
    
    script_path = os.path.dirname(__file__)
    script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
    logfile_path = alex.os.logfile_full_path(script_path,script_name_without_extension)
    outputfile_graph_path = alex.os.outputfile_graph_full_path(script_path,script_name_without_extension)
    outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path,script_name_without_extension)

    # set FEniCSX log level
    # dlfx.log.set_log_level(log.LogLevel.INFO)
    # dlfx.log.set_output_file('xxx.log')

    # set and start stopwatch
    timer = dlfx.common.Timer()
    timer.start()

    # set MPI environment
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    # print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
    # sys.stdout.flush()

    # mesh 
    N = 16 

    # generate domain
    #domain = dlfx.mesh.create_unit_square(comm, N, N, cell_type=dlfx.mesh.CellType.quadrilateral)
    domain = dlfx.mesh.create_unit_cube(comm,N,N,N,cell_type=dlfx.mesh.CellType.hexahedron)

    # with dlfx.io.XDMFFile(comm, os.path.join(alex.os.resources_directory,'polycrystal_cube.xdmf'), 'r') as mesh_inp: 
    #     domain = mesh_inp.read_mesh()


    dt = 0.05
    Tend = 2.0 * dt

    # elastic constants
    lam = dlfx.fem.Constant(domain, 10.0)
    mu = dlfx.fem.Constant(domain, 10.0)
    E_mod = alex.linearelastic.get_emod(lam.value, mu.value)

    # function space using mesh and degree
    Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1) # displacements
    V = dlfx.fem.FunctionSpace(domain, Ve)

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
        # if rank == 0:
            # sol.prepare_newton_logfile(logfile_path)
            # pp.prepare_graphs_output_file(outputfile_graph_path)
        # prepare xdmf output 
        # pp.write_mesh_and_get_outputfile_xdmf(domain, outputfile_xdmf_path, comm)

    def before_each_time_step(t,dt):
        return
        # report solution status
        # if rank == 0:
        #     sol.print_time_and_dt(t,dt)
        
    linearElasticProblem = alex.linearelastic.StaticLinearElasticProblem()

    def get_residuum_and_gateaux(delta_t: dlfx.fem.Constant):
        [Res, dResdw] = linearElasticProblem.prep_newton(u,du,ddu,lam,mu)
        return [Res, dResdw]


    # eps_mac = dlfx.fem.Constant(domain, np.array([[0.0, 0.0, 0.0],
    #                     [0.0, 0.6, 0.0],
    #                     [0.0, 0.0, 0.0]]))


    def get_bcs(t):
        
        vertices_at_corner = dlfx.mesh.locate_entities(domain,fdim-1,bc.get_corner_of_box_as_function(domain,comm))
        dofs_at_corner_x = dlfx.fem.locate_dofs_topological(V.sub(0),fdim-1,vertices_at_corner)
        bc_corner_x = dlfx.fem.dirichletbc(0.0,dofs_at_corner_x,V.sub(0))
        dofs_at_corner_y = dlfx.fem.locate_dofs_topological(V.sub(1),fdim-1,vertices_at_corner)
        bc_corner_y = dlfx.fem.dirichletbc(0.0,dofs_at_corner_y,V.sub(1))
        dofs_at_corner_z = dlfx.fem.locate_dofs_topological(V.sub(2),fdim-1,vertices_at_corner)
        bc_corner_z = dlfx.fem.dirichletbc(0.0,dofs_at_corner_z,V.sub(2))
        bcs = [bc_corner_x,  bc_corner_y, bc_corner_z]
        
        sig_mac = dlfx.fem.Constant(domain, sig_mac_param)
        
        linearElasticProblem.set_traction_bc(sigma=sig_mac,u=u,n=n,ds=ufl.ds)
        
        # bcs = bc.get_total_linear_displacement_boundary_condition_at_box(domain, comm, V,eps_mac=eps_mac)
        return bcs

    n = ufl.FacetNormal(domain)
    simulation_result = np.array([0.0])
    def after_timestep_success(t,dt,iters):
        
        u.name = "u"
        # pp.write_vector_field(domain,outputfile_xdmf_path,u,t,comm)
        
        sig_vm = le.sigvM(le.sigma_as_tensor(u,lam,mu))
        # sig_vm.name = "sigvm"
        
        # pp.write_scalar_fields(domain,comm,[sig_vm],["sigvm"],outputfile_xdmf_path,t)
        
        sig_vm_max, sig_vm_min = pp.get_extreme_values_of_scalar_field(domain,sig_vm,comm)
        simulation_result[0] = sig_vm_max
        
        # write to newton-log-file
        # if rank == 0:
            
        #     sol.write_to_newton_logfile(logfile_path,t,dt,iters)
            
        urestart.x.array[:] = u.x.array[:] 
                
        
        
    def after_timestep_restart(t,dt,iters):
        u.x.array[:] = urestart.x.array[:]
        
        
    def after_last_timestep():
        # stopwatch stop
        timer.stop()

        # report runtime to screen
        if rank == 0:
            runtime = timer.elapsed()
            # sol.print_runtime(runtime)
            # sol.write_runtime_to_newton_logfile(logfile_path,runtime)
            # pp.print_graphs_plot(outputfile_graph_path,script_path,legend_labels=["Jx_surf", "Ge_x_div", "Jx_nodal_forces", "Gad_x", "Gdis_x"])
            
            # cleanup only necessary on cluster
            # results_folder_path = alex.os.create_results_folder(script_path)
            # alex.os.copy_contents_to_results_folder(script_path,results_folder_path)

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
        comm=comm
    )
    return comm.allreduce(simulation_result,MPI.MAX)

# result = run_simulation()



