import gmsh
import os

# -------------------------------------------------
# Save mesh next to this script
# -------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
mesh_path = os.path.join(script_dir, "porous_lattice.msh")

# -------------------------------------------------
# Initialize Gmsh
# -------------------------------------------------
gmsh.initialize()
gmsh.model.add("PorousCuboid_Buffered")

# -------------------------------------------------
# USER PARAMETERS
# -------------------------------------------------
nx, ny, nz = 3, 1 , 1          # number of pores
D = 1.0                      # pore diameter
gap = 0.5                    # surface-to-surface pore spacing

refinement_factor = 4       # center mesh refinement factor
h_max = 0.25                  # coarse mesh size

# -------------------------------------------------
# Derived parameters
# -------------------------------------------------
r = D / 2.0
h_min = h_max / refinement_factor

# Size of pore lattice (no buffer)
Lx_p = nx * D + (nx - 1) * gap
Ly_p = ny * D + (ny - 1) * gap
Lz_p = nz * D + (nz - 1) * gap

# Add buffer of width gap/2 on all sides
buffer = gap / 2.0

Lx = Lx_p + 2 * buffer
Ly = Ly_p + 2 * buffer
Lz = Lz_p + 2 * buffer

# -------------------------------------------------
# Create cuboidal matrix (with buffer)
# -------------------------------------------------
cube = gmsh.model.occ.addBox(0.0, 0.0, 0.0, Lx, Ly, Lz)

# -------------------------------------------------
# Create spherical pores
# -------------------------------------------------
dx = D + gap
dy = D + gap
dz = D + gap

x0 = buffer + r
y0 = buffer + r
z0 = buffer + r

spheres = []

for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            x = x0 + i * dx
            y = y0 + j * dy
            z = z0 + k * dz
            spheres.append(
                gmsh.model.occ.addSphere(x, y, z, r)
            )

# -------------------------------------------------
# Boolean cut (robust single operation)
# -------------------------------------------------
gmsh.model.occ.cut(
    [(3, cube)],
    [(3, s) for s in spheres],
    removeObject=True,
    removeTool=True
)
gmsh.model.occ.synchronize()

# -------------------------------------------------
# Physical volume
# -------------------------------------------------
volumes = gmsh.model.getEntities(dim=3)
gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes], 1)
gmsh.model.setPhysicalName(3, 1, "Solid")

# -------------------------------------------------
# Mesh refinement toward center pore row in y
# -------------------------------------------------
dist_field = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(
    dist_field,
    "FacesList",
    [s[1] for s in gmsh.model.getEntities(dim=2)]
)

th_field = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(th_field, "InField", dist_field)
gmsh.model.mesh.field.setNumber(th_field, "SizeMin", h_min)
gmsh.model.mesh.field.setNumber(th_field, "SizeMax", h_max)
gmsh.model.mesh.field.setNumber(th_field, "DistMin", D)
gmsh.model.mesh.field.setNumber(th_field, "DistMax", 2.5 * (D + gap))

gmsh.model.mesh.field.setAsBackgroundMesh(th_field)

# -------------------------------------------------
# Mesh options
# -------------------------------------------------
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

# -------------------------------------------------
# Generate mesh and save
# -------------------------------------------------
gmsh.model.mesh.generate(3)
gmsh.write(mesh_path)

gmsh.finalize()

print("Mesh written to:", mesh_path)


