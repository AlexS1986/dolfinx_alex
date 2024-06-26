import cube_function
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]

# if comm.Get_rank() == 0:
#     # Remove the file if it exists
#     file_path = os.path.join(script_path, 'failure_surface.csv')
#     if os.path.exists(file_path):
#         os.remove(file_path)

# # Define arrays
# n_values = 3
# values = np.linspace(-1.0, 1.0, n_values)

# # Initialize arrays
# sxx = values
# sxy = values
# sxz = values
# syz = values
# szz = values
# syy = values

# total_iterations = len(sxx) * len(sxy) * len(sxz) * len(syz) * len(szz) * len(syy)
# computation = 1

# for val_sxx in sxx:
#     for val_sxy in sxy:
#         for val_sxz in sxz:
#             for val_syz in syz:
#                 for val_szz in szz:
#                     for val_syy in syy:
#                         sig_mac = np.array([[val_sxx, val_sxy, val_sxz],
#                                              [val_sxy, val_syy, val_syz],
#                                              [val_sxz, val_syz, val_szz]])
#                         if np.linalg.norm(sig_mac) < 1.0e-3:  # if all entries are zero then no ev
#                             break

#                         try:
#                             comm.barrier()
#                             sig_vm_max = cube_function.run_simulation(sig_mac_param=sig_mac, comm=comm)
#                             sig_vm_c = 1.0
#                             sig_at_failure = sig_mac * sig_vm_c / sig_vm_max  # if linear
#                             principal_stress_at_failure = np.linalg.eigvals(sig_at_failure) # does not make sense to do this in main stresses since not isotropic?
                            
#                             if comm.Get_rank() == 0:
#                                 with open(os.path.join(script_path, 'failure_surface.csv'), 'a') as file:
#                                     if np.linalg.norm(principal_stress_at_failure) < 5.0 * sig_vm_c:
#                                         file.write(','.join(map(str, principal_stress_at_failure)) + '\n')
#                                 print("Running computation {} of {} total".format(computation, total_iterations))
#                                 sys.stdout.flush()

#                             computation += 1
#                             comm.barrier()
#                         except Exception as e:
#                             if comm.Get_rank() == 0:
#                                 print("An error occurred in computation " + str(computation) + " message:", e)
#                                 print("sigma is: \n")
#                                 print(sig_mac)
#                             computation += 1



if comm.Get_rank() == 0:
    
    
    def are_close(a, b, tol=1e-6):
        return abs(a - b) < tol

    def remove_duplicates(arr):
            unique_arrays = []
            for sub_arr in arr:
                is_duplicate = False
                for unique_sub_arr in unique_arrays:
                    if (are_close(sub_arr[0], unique_sub_arr[0]) and
                        are_close(sub_arr[1], unique_sub_arr[1]) and
                        are_close(sub_arr[2], unique_sub_arr[2])):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_arrays.append(sub_arr)
            return unique_arrays
    
    # Read data from the file for plotting
    points = np.genfromtxt(os.path.join(script_path, 'failure_surface.csv'), delimiter=',')
        
    points = np.array(remove_duplicates(points))
    
    mask = np.isnan(points).any(axis=1)
    
    points = points[~mask]

    # Compute convex hull
    hull = ConvexHull(points)

    # Plot convex hull
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'o')

    # Create triangulation of convex hull vertices
    triang = Triangulation(points[:, 0], points[:, 1], triangles=hull.simplices)

    # Plot smooth surface of convex hull
    ax.plot_trisurf(triang, points[:, 2], color='blue', alpha=0.5)

    # Save the plot as a screenshot
    plt.savefig(os.path.join(script_path, 'failure_surface.png'))





# import cube_function
# import numpy as np
# import scipy

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.spatial import ConvexHull
# from matplotlib.tri import Triangulation

# from tqdm import tqdm

# import os

# from mpi4py import MPI

# import sys

# comm = MPI.COMM_WORLD


# script_path = os.path.dirname(__file__)
# script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]

# # Define arrays
# n_values = 2
# values = np.linspace(-1.0, 1.0, n_values)

# # Initialize arrays
# sxx = values
# sxy = values
# sxz = values
# syz = values
# szz = values
# syy = values

# principal_stresses_at_failure = []
# total_iterations = len(sxx) * len(sxy) * len(sxz) * len(syz) * len(szz) * len(syy)
# computation = 1
# # with tqdm(total=total_iterations) as pbar:
# for val_sxx in sxx:
#         for val_sxy in sxy:
#             for val_sxz in sxz:
#                 for val_syz in syz:
#                     for val_szz in szz:
#                         for val_syy in syy: 
#                             sig_mac = np.array([[val_sxx, val_sxy, val_sxz],
#                                     [val_sxy, val_syy, val_syz],
#                                     [val_sxz, val_syz, val_szz]])
#                             if np.all(sig_mac == 0): # if all entries are zero then no ev
#                                 break
                            
#                             sig_vm_max = cube_function.run_simulation(sig_mac_param=sig_mac, comm=comm)
#                             sig_vm_c = 1.0
#                             sig_at_failure = sig_mac * sig_vm_c / sig_vm_max # if linear


#                             principal_stress_at_failure = np.linalg.eigvals(sig_at_failure)
                            
#                             if comm.Get_rank() == 0:
#                                 principal_stresses_at_failure.append(principal_stress_at_failure) # write everything to file first and read later for plot
#                                 print("Running computation {} of {} total".format(computation,total_iterations))
#                                 sys.stdout.flush()
                            
#                             computation += 1
#                                 # pbar.update(1) 
#                             # print("Principal Stresses:", principal_stress_at_failure)


# if comm.Get_rank() == 0:
#     def are_close(a, b, tol=1e-6):
#         return abs(a - b) < tol

#     def remove_duplicates(arr):
#         unique_arrays = []
#         for sub_arr in arr:
#             is_duplicate = False
#             for unique_sub_arr in unique_arrays:
#                 if (are_close(sub_arr[0], unique_sub_arr[0]) and
#                     are_close(sub_arr[1], unique_sub_arr[1]) and
#                     are_close(sub_arr[2], unique_sub_arr[2])):
#                     is_duplicate = True
#                     break
#             if not is_duplicate:
#                 unique_arrays.append(sub_arr)
#         return unique_arrays
    
#     points = np.array(remove_duplicates(principal_stresses_at_failure))
#     with open(os.path.join(script_path, 'failure_surface.csv'), 'w') as file:
#         for point in points:
#             file.write(','.join(map(str, point)) + '\n')
    
    
    
#     # Compute convex hull
#     hull = ConvexHull(points)

#     # Plot convex hull
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot points
#     ax.plot(points[:, 0], points[:, 1], points[:, 2], 'o')

    
#     # Create triangulation of convex hull vertices
#     triang = Triangulation(points[:, 0], points[:, 1], triangles=hull.simplices)

#     # Plot smooth surface of convex hull
#     ax.plot_trisurf(triang, points[:, 2], color='blue', alpha=0.5)
    
#     # # Plot convex hull
#     # for simplex in hull.simplices:
#     #     simplex = np.append(simplex, simplex[0])  # Close the loop
#     #     ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'r-')

#     # Save the plot as a screenshot
#     plt.savefig(os.path.join(script_path,'failure_surface.png'))











# print(sig_vm_max)
# print(sig_at_failure)

