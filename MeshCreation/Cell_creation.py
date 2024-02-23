import warnings
warnings.filterwarnings("ignore")
import gmsh
import numpy as np
import sys

######################################################
r = 0.25
res = 0.7 * 1.0/64.0

gmsh.initialize()
gmsh.model.add("cell")

circle = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, r, r)

## Always_
gmsh.model.occ.synchronize()

gmsh.model.addPhysicalGroup(2, [2, circle], 0, name="domain")
gmsh.option.setNumber('Mesh.MeshSizeMax', res)

gmsh.model.mesh.generate(2)
gmsh.write('MeshCreation/micro_domain.msh')

# Launch the GUI to see the results:
if '-nopopup' not in sys.argv:
   gmsh.fltk.run()

# close gmsh
gmsh.finalize()