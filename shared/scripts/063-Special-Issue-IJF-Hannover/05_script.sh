#!/bin/bash

# # Hardcoded folder path - change this to your input folder
FOLDER="/home/scripts/063-Special-Issue-IJF-Hannover/resources/dcb_var_bcpos_E_var/export"

# # # # Run the mesh conversion script
# python3 04_mesh2dlfxmesh.py "$FOLDER" 
# # # # # Run the phasefield script in parallel with 4 processes
# mpirun -np 4 python3 01_phasefield_dcb_whole_folder.py "$FOLDER" "vary" 1 20

# # Run the mesh conversion script
# Run the phasefield script in parallel with 4 processes
mpirun -np 4 python3 01_phasefield_dcb_whole_folder.py "$FOLDER" "fromfile" 18 20 

# Hardcoded folder path - change this to your input folder
FOLDER="/home/scripts/063-Special-Issue-IJF-Hannover/resources/dcb_var_bcpos_E_max/export"

# Run the mesh conversion script
python3 04_mesh2dlfxmesh.py "$FOLDER" 
# Run the phasefield script in parallel with 4 processes
mpirun -np 4 python3 01_phasefield_dcb_whole_folder.py "$FOLDER" "max" 1 20

# Hardcoded folder path - change this to your input folder
FOLDER="/home/scripts/063-Special-Issue-IJF-Hannover/resources/dcb_var_bcpos_E_min/export"

# Run the mesh conversion script
python3 04_mesh2dlfxmesh.py "$FOLDER" 
# Run the phasefield script in parallel with 4 processes
mpirun -np 4 python3 01_phasefield_dcb_whole_folder.py "$FOLDER" "min" 1 20