import argparse
import os

import gmsh
import meshio
import numpy as np
import pygmsh


script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]

parser = argparse.ArgumentParser(description="Create a unit-cell mesh with a circular inclusion.")
try:
    parser.add_argument("--dhole", type=float, required=True, help="Diameter of circular inclusion")
    parser.add_argument("--wsteg", type=float, required=True, help="Matrix ligament width around the inclusion")
    parser.add_argument("--e0", type=float, required=True, help="Element size")
    args = parser.parse_args()
    dinclusion = args.dhole
    wsteg = args.wsteg
    e0 = args.e0
except Exception:
    print("Could not parse arguments")
    dinclusion = 1.0
    wsteg = 0.25
    e0 = 0.02

w_cell = dinclusion + wsteg
h_cell = w_cell
radius = dinclusion / 2.0
filename = os.path.join(script_path, script_name_without_extension)
mesh_info = False

geom = pygmsh.occ.Geometry()
model = geom.__enter__()
model.characteristic_length_min = e0
model.characteristic_length_max = e0

x_center = 0.5 * w_cell
y_center = 0.0
matrix = model.add_rectangle([0.0, -0.5 * h_cell, 0.0], w_cell, h_cell, 0.0)
inclusion = model.add_disk([x_center, y_center, 0.0], radius)

# Keep both phases in the mesh by fragmenting instead of subtracting the disk.
model.boolean_fragments(matrix, inclusion, delete_first=True, delete_other=True)

model.synchronize()
model.generate_mesh(dim=2, verbose=True)
gmsh.write(filename + ".msh")
gmsh.clear()
model.__exit__()

mesh = meshio.read(filename + ".msh")
nodes = mesh.points[:, 0:2]
elems = mesh.get_cells_type("triangle")

centroids = nodes[elems].mean(axis=1)
distances = np.linalg.norm(centroids - np.array([x_center, y_center]), axis=1)
elem_data = np.where(distances <= radius, 1, 0).astype(np.int32)

if mesh_info:
    print("NODES:")
    print(nodes)
    print("ELEMENTS")
    print(elems)
    print("ELEMENT DATA")
    print(elem_data)

cell_mesh = meshio.Mesh(points=nodes, cells={"triangle": elems}, cell_data={"name_to_read": [elem_data]})
meshio.write(os.path.join(script_path, script_name_without_extension + ".xdmf"), cell_mesh)
