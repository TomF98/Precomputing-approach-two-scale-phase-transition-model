import meshio

def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(points=mesh.points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    if prune_z:
        out_mesh.points = out_mesh.points[:, :2] # remove third dimension
    return out_mesh

res = 0.05

msh=meshio.read("MeshCreation/micro_domain_res_" + str(res) + ".msh")

triangle_mesh = create_mesh(msh, "triangle", True)
meshio.write("MeshCreation/micro_domain_res_" + str(res) + ".xdmf", triangle_mesh)