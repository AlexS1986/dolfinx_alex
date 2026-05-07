import argparse
import math
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

parser = argparse.ArgumentParser(description="Create a fracture mesh with circular inclusions.")
try:
    parser.add_argument("--nholes", type=int, required=True, help="Number of inclusions per row")
    parser.add_argument("--dhole", type=float, required=True, help="Diameter of circular inclusions")
    parser.add_argument("--wsteg", type=float, required=True, help="Matrix ligament width")
    parser.add_argument("--e0", type=float, required=True, help="Fine element size")
    parser.add_argument("--e1", type=float, required=True, help="Coarse element size")
    args = parser.parse_args()
    Nholes = args.nholes
    dinclusion = args.dhole
    wsteg = args.wsteg
    e0 = args.e0
    e1 = args.e1
except Exception:
    print("Could not parse arguments")
    Nholes = 6
    dinclusion = 1.0
    wsteg = 0.25
    e0 = 0.02
    e1 = 0.8

w_cell = dinclusion + wsteg
h_cell = w_cell
radius = dinclusion / 2.0
l0 = (Nholes + 2) * w_cell
h0 = 20.0
n_rows = 3
filename = os.path.join(script_path, script_name_without_extension)
mesh_info = False

geom = pygmsh.occ.Geometry()
model = geom.__enter__()

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
for i in range(n_rows):
    for n in range(Nholes):
        x_center = w_cell + w_cell / 2.0 + n * w_cell
        y_center = -(n_rows // 2) * h_cell + i * h_cell
        inclusion_centers.append([x_center, y_center])

        cell_matrix = model.add_rectangle(
            [x_center - w_cell / 2.0, y_center - h_cell / 2.0, 0.0],
            w_cell,
            h_cell,
            0.0,
        )
        fragmented_domain = model.boolean_fragments(
            fragmented_domain, cell_matrix, delete_first=True, delete_other=True
        )

        inclusion = model.add_disk([x_center, y_center, 0.0], radius)
        fragmented_domain = model.boolean_fragments(
            fragmented_domain, inclusion, delete_first=True, delete_other=True
        )

p8 = model.add_point([0.0, 0.0])
p9 = model.add_point([l0, 0.0])
crack = model.add_line(p8, p9)

gmsh.model.mesh.field.add("Distance", 1)
inclusion_points = [model.add_point([x, y, 0.0], e0)._id for x, y in inclusion_centers]
crack_points = [model.add_point([i * l0 / 50.0, 0.0, 0.0], e0)._id for i in range(51)]
gmsh.model.mesh.field.setNumbers(1, "NodesList", inclusion_points + crack_points)

gmsh.model.mesh.field.add("Distance", 2)
gmsh.model.mesh.field.setNumbers(2, "EdgesList", [crack._id])

gmsh.model.mesh.field.add("Min", 3)
gmsh.model.mesh.field.setNumbers(3, "FieldsList", [1, 2])

gmsh.model.mesh.field.add("Threshold", 4)
gmsh.model.mesh.field.setNumber(4, "InField", 3)
gmsh.model.mesh.field.setNumber(4, "SizeMin", e0)
gmsh.model.mesh.field.setNumber(4, "SizeMax", e1)
gmsh.model.mesh.field.setNumber(4, "DistMin", radius)
gmsh.model.mesh.field.setNumber(4, "DistMax", w_cell)
gmsh.model.mesh.field.setAsBackgroundMesh(4)

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
nodes = mesh.points[:, :2]
elems = mesh.get_cells_type("triangle")

def filter_points_and_update_cells(cell_array, point_array):
    referenced_ids = sorted({point_id for triangle in cell_array for point_id in triangle})
    filtered_points = np.array([point_array[i] for i in referenced_ids])
    mapping = {old: new for new, old in enumerate(referenced_ids)}
    updated_cells = np.array([[mapping[pt] for pt in triangle] for triangle in cell_array], dtype=np.int64)
    return filtered_points, updated_cells

def classify_cells(point_array, cell_array):
    centroids = point_array[cell_array].mean(axis=1)
    tags = np.zeros(len(cell_array), dtype=np.int32)

    resolved_x_min = w_cell
    resolved_x_max = (Nholes + 1) * w_cell
    resolved_y_min = -(n_rows // 2) * h_cell - h_cell / 2.0
    resolved_y_max = (n_rows // 2) * h_cell + h_cell / 2.0
    in_resolved_matrix = (
        (centroids[:, 0] >= resolved_x_min)
        & (centroids[:, 0] <= resolved_x_max)
        & (centroids[:, 1] >= resolved_y_min)
        & (centroids[:, 1] <= resolved_y_max)
    )
    tags[in_resolved_matrix] = 1

    centers = np.array(inclusion_centers)
    for center in centers:
        distance = np.linalg.norm(centroids - center, axis=1)
        tags[distance <= radius] = 2

    return tags

nodes, elems = filter_points_and_update_cells(elems, nodes)
elem_data = classify_cells(nodes, elems)

cell_mesh = meshio.Mesh(points=nodes, cells={"triangle": elems}, cell_data={"name_to_read": [elem_data]})
meshio.write(filename + ".xdmf", cell_mesh)

params = {
    "nholes": Nholes,
    "dinclusion": dinclusion,
    "dhole": dinclusion,
    "wsteg": wsteg,
    "e0": e0,
    "e1": e1,
    "cell_marker_effective": 0,
    "cell_marker_matrix": 1,
    "cell_marker_inclusion": 2,
}
append_parameters(parameter_path, params)

if mesh_info:
    print("NODES:")
    print(nodes)
    print("ELEMENTS")
    print(elems)
    print("ELEMENT DATA")
    print(elem_data)
