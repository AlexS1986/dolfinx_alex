import argparse
import os

import gmsh
import meshio
import numpy as np
import pygmsh


script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
parameter_path = os.path.join(script_path, "parameters.txt")

def append_parameters(filename, parameters):
    with open(filename, "a") as file:
        for key, value in parameters.items():
            file.write(f"{key}={value}\n")

parser = argparse.ArgumentParser(description="Create a fracture mesh with one row of circular inclusions.")
try:
    parser.add_argument("--nholes", type=int, required=True, help="Number of inclusions")
    parser.add_argument("--dhole", type=float, required=True, help="Diameter of circular inclusions")
    parser.add_argument("--wsteg", type=float, required=True, help="Matrix ligament width")
    parser.add_argument("--e0", type=float, required=True, help="Element size")
    args = parser.parse_args()
    Nholes = args.nholes
    dinclusion = args.dhole
    wsteg = args.wsteg
    e0 = args.e0
except Exception:
    print("Could not parse arguments")
    Nholes = 4
    dinclusion = 1.0
    wsteg = 0.25
    e0 = 0.02

w_cell = dinclusion + wsteg
h_cell = w_cell
radius = dinclusion / 2.0
l0 = (Nholes + 2) * w_cell
h0 = 2.0 * w_cell
filename = os.path.join(script_path, script_name_without_extension)
mesh_info = False

geom = pygmsh.occ.Geometry()
model = geom.__enter__()
model.characteristic_length_min = e0
model.characteristic_length_max = e0

p0 = model.add_point([0.0, -h0 / 2.0])
p1 = model.add_point([l0, -h0 / 2.0])
p2 = model.add_point([l0, h0 / 2.0])
p3 = model.add_point([0.0, h0 / 2.0])
outer = model.add_curve_loop([
    model.add_line(p0, p1),
    model.add_line(p1, p2),
    model.add_line(p2, p3),
    model.add_line(p3, p0),
])
fragmented_domain = model.add_plane_surface(outer)

inclusion_centers = []
for n in range(Nholes):
    x_center = w_cell + w_cell / 2.0 + n * w_cell
    y_center = 0.0
    inclusion_centers.append([x_center, y_center])

    cell_matrix = model.add_rectangle(
        [x_center - w_cell / 2.0, -h_cell / 2.0, 0.0], w_cell, h_cell, 0.0
    )
    fragmented_domain = model.boolean_fragments(
        fragmented_domain, cell_matrix, delete_first=True, delete_other=True
    )

    inclusion = model.add_disk([x_center, y_center, 0.0], radius)
    fragmented_domain = model.boolean_fragments(
        fragmented_domain, inclusion, delete_first=True, delete_other=True
    )

p8 = model.add_point([0.0, 0.0])
p9 = model.add_point([w_cell, 0.0])
crack = model.add_line(p8, p9)

model.synchronize()
surface_tags = [tag for dim, tag in gmsh.model.getEntities(2)]
gmsh.model.addPhysicalGroup(2, surface_tags, 1)
gmsh.model.setPhysicalName(2, 1, "domain")
model.add_physical(crack, "crack")
model.generate_mesh(dim=2, verbose=True)
gmsh.write(filename + ".msh")
gmsh.clear()
model.__exit__()

mesh = meshio.read(filename + ".msh")
nodes = mesh.points[:, 0:2]
elems = mesh.get_cells_type("triangle")
centroids = nodes[elems].mean(axis=1)

elem_data = np.zeros(len(elems), dtype=np.int32)
in_resolved_matrix = (
    (centroids[:, 0] >= w_cell)
    & (centroids[:, 0] <= (Nholes + 1) * w_cell)
    & (centroids[:, 1] >= -h_cell / 2.0)
    & (centroids[:, 1] <= h_cell / 2.0)
)
elem_data[in_resolved_matrix] = 1
for center in np.array(inclusion_centers):
    elem_data[np.linalg.norm(centroids - center, axis=1) <= radius] = 2

if mesh_info:
    print("NODES:")
    print(nodes)
    print("ELEMENTS")
    print(elems)
    print("ELEMENT DATA")
    print(elem_data)

cell_mesh = meshio.Mesh(points=nodes, cells={"triangle": elems}, cell_data={"name_to_read": [elem_data]})
meshio.write(os.path.join(script_path, script_name_without_extension + ".xdmf"), cell_mesh)

parameters_to_write = {
    "nholes": Nholes,
    "dinclusion": dinclusion,
    "dhole": dinclusion,
    "wsteg": wsteg,
    "cell_marker_effective": 0,
    "cell_marker_matrix": 1,
    "cell_marker_inclusion": 2,
}
append_parameters(parameter_path, parameters_to_write)
