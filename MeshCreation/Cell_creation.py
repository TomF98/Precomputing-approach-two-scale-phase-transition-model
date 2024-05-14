import warnings
warnings.filterwarnings("ignore")
import gmsh
import numpy as np
import sys

### Build the cell domains (circles)
radius = 0.25
res = 0.05
show_mesh = True

gmsh.initialize()
gmsh.model.add("cell")

circle = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, radius, radius)

gmsh.model.occ.synchronize()

gmsh.model.addPhysicalGroup(2, [2, circle], 0, name="domain")
gmsh.option.setNumber('Mesh.MeshSizeMax', res)

gmsh.model.mesh.generate(2)
gmsh.write("MeshCreation/micro_domain_res_" + str(res) + ".msh")

# Launch the GUI to see the results:
if show_mesh and '-nopopup' not in sys.argv:
   gmsh.fltk.run()

# close gmsh
gmsh.finalize()