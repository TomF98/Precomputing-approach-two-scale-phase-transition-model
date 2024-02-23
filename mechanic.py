from dolfin import *

# Problem parameters
L = 1; H = 0.5

C = as_tensor(
    [[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], 
     [[[0, 0], [1, 0]], [[0, 0], [0, 5]]]]
    )

alpha = 1.0

g = 9.81
rho = 0.1

f = Constant((0, -rho*g))

macro_res = 16

space_dim = 2

# Create mesh and define function space
test_mesh = RectangleMesh(Point(0, 0), Point(L, H), macro_res, macro_res)
V = VectorFunctionSpace(test_mesh, 'CG', 1)

# Define boundary condition
def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < DOLFIN_EPS

bc = DirichletBC(V, Constant((0, 0)), clamped_boundary)

# Define strain and stress
def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(eps):
    res_00 = inner(C[0, 0], eps)
    res_01 = inner(C[0, 1], eps)
    res_10 = inner(C[1, 0], eps)
    res_11 = inner(C[1, 1], eps)
    return as_tensor([[res_00, res_01], [res_10, res_11]])

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

a = inner(sigma(epsilon(u)), grad(v))*dx
L = dot(f, v)*dx 

# Compute solution
u = Function(V)
solve(a == L, u, bc)


fff = File("Results/displacement.pvd")
fff << u