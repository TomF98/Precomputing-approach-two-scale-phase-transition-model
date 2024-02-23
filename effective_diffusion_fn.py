from dolfin import *
import numpy as np
import mshr

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


def diffusion_compute(radius, res=64):
    e_j = Constant((1.0, 0.0)) # since we have circle, direction does not matter

    ######################
    ### Domain and measure
    circle = mshr.Circle(Point(0.0, 0.0), radius)
    box = mshr.Rectangle(Point(-0.5, -0.5), Point(0.5, 0.5))
    domain_mesh = mshr.generate_mesh(box - circle, res)

    dx = Measure("dx", domain_mesh)

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

    solve(a==f, w)#, solver_parameters={'linear_solver' : 'mumps'})

    u0, _ = w.split()

    diffusion_scale = assemble(inner((grad(u0) + e_j), Constant((1, 0)))*dx)
    return diffusion_scale
