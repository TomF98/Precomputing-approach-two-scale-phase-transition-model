"""
Solve the helper problem, that defines the macroscopic diffusion.
The values are precomputed for cell sizes and later only looked up (interpolated)
"""
from dolfin import *
import numpy as np
import mshr

res = 256 # mesh resolution
N = 20 # number of precompute values

j = 0
if j == 0:
    e_j = Constant((1.0, 0.0))
else:
    e_j = Constant((0.0, 1.0))

results = np.zeros((N, 2)) # Entires contain: radius | value
radius_list = np.linspace(0.005, 0.495, len(results))
counter = 0

for radius in radius_list:
    ######################
    ### Domain and measure
    circle = mshr.Circle(Point(0.0, 0.0), radius)
    box = mshr.Rectangle(Point(-0.5, -0.5), Point(0.5, 0.5))
    domain_mesh = mshr.generate_mesh(box - circle, res)

    facet_markers = 0

    dx = Measure("dx", domain_mesh)

    #####################
    ### Functionspace stuff
    class PeriodicBC(SubDomain):

        def __init__(self):
            super().__init__()

        def inside(self, x, on_boundary):
            left, bottom = near(x[0], -0.5), near(x[1], -0.5)
            corner_1 = (near(x[1], 0.5) and left)
            corner_2 = (near(x[0], 0.5) and bottom)
            return on_boundary and (left or bottom) and not (corner_1 or corner_2)

        def map(self, x, y):
            right, top = near(x[0], 0.5), near(x[1], 0.5)
            y[0] = x[0]
            y[1] = x[1]
            if right:
                y[0] -= 1.0
            if top:
                y[1] -= 1.0


    V = FiniteElement('CG', domain_mesh.ufl_cell(), 1)
    R = FiniteElement('R', domain_mesh.ufl_cell(), 0)
    mixed = MixedElement([V, R])
    W = FunctionSpace(domain_mesh, mixed, constrained_domain=PeriodicBC())

    u, lamb = TrialFunctions(W)
    v, mu = TestFunctions(W)

    a = inner(grad(u), grad(v)) + inner(lamb, v) + inner(u, mu)
    a *= dx

    f = -inner(e_j, grad(v)) * dx


    w = Function(W)

    solve(a==f, w)

    u0, _ = w.split()

    volume = assemble(1*dx)
    print("volume", volume)
    diffusion_scale = assemble(inner((grad(u0) + e_j), Constant((1, 0)))*dx)
    print("entry "+ str(j+1) + ",1 :", diffusion_scale)
    print("entry "+ str(j+1) + ",2 :", assemble(inner((grad(u0) + e_j), Constant((0, 1)))*dx))

    # print("Other computaion")
    # diffusion_scale = assemble(inner((grad(u0) + e_j), (grad(u0) + e_j))*dx)
    # print("entry "+ str(j+1) + ",1 :", diffusion_scale)

    #print("Without volume")
    #print("entry "+ str(j+1) + ",1 :", assemble(inner((grad(u0) + e_j), Constant((1, 0, 0)))*dx))
    #print("entry "+ str(j+1) + ",2 :", assemble(inner((grad(u0) + e_j), Constant((0, 1, 0)))*dx))
    
    filev = File("TestResults/chi_" + str(radius) + ".pvd")
    filev << u0

    results[counter, 0] = radius
    results[counter, 1] = diffusion_scale
    counter += 1


np.save("Data/effective_conductivity_res_" + str(N) + ".npy", results)

## Show coefficient curve
import matplotlib.pyplot as plt
plt.plot(results[:, 0], results[:, 1])
plt.grid()
plt.xlabel("Radius")
plt.ylabel(r"$K_{11}$")
plt.show()
