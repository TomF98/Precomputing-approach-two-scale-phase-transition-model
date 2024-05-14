from dolfin import *
import numpy as np
import mshr
from scipy.sparse import csr_matrix, lil_matrix, bmat
import time 
from petsc4py import PETSc

from MicroProblems.cell_problem import CellProblem

### Problem parameters
kappa_macro = 0.1
kappa_micro = 0.1

theta_ref = 1.0 
start_temp = 0.0
f_macro_prod = Constant(1.0)
f_micro_prod = Constant(0.0)

## Effective conductivity
eff_cond_array = np.load("effective_conductivity.npy")

### Time interval
T_int = [0, 5.0]
dt = 0.02
t_n = T_int[0]

prod_stop = 0.5

### Domain parameters
macro_size_x = 1.0
macro_size_y = 1.0

micro_radius = 0.2
growth_speed = 0.1

macro_res = 2
micro_res = 2

## Genreate mesh
macro_mesh = RectangleMesh(Point(0, 0), Point(macro_size_x, macro_size_y), 
                           macro_res, macro_res)

#circle = mshr.Circle(Point(0.0, 0.0), micro_radius)
#micro_mesh = mshr.generate_mesh(circle, micro_res)

micro_mesh = RectangleMesh(Point(0, 0), Point(0.5, 0.5), 
                    micro_res, micro_res)

### Generate macro problem
V_macro = FunctionSpace(macro_mesh, "Lagrange", 1)
dof_map = V_macro.dofmap()

sol_function = Function(V_macro)
sol_function.assign(Constant(start_temp))

theta_old = Function(V_macro)
theta_old.assign(Constant(start_temp))

effective_cond_fn = Function(V_macro)
effective_density_fn = Function(V_macro)
old_effective_density_fn = Function(V_macro)

dofs_macro = V_macro.dim()

phi_macro = TestFunction(V_macro)
u_macro = TrialFunction(V_macro)

n_micro = FacetNormal(micro_mesh)

dx_macro = Measure("dx", V_macro)

### Macro problem
a_macro = inner(kappa_macro * effective_cond_fn * grad(u_macro), grad(phi_macro)) * dx_macro
a_macro += inner(effective_density_fn * u_macro, phi_macro) / dt * dx_macro

f_macro = (inner(old_effective_density_fn * theta_old, phi_macro) / dt \
           + inner(f_macro_prod, phi_macro)) * dx_macro

bc_cool = DirichletBC(V_macro, Constant(start_temp), "on_boundary and x[0] <= DOLFIN_EPS")

### Create micro problems
micro_problem_list = []

for _ in range(dofs_macro):
    micro_problem_list.append(CellProblem(micro_mesh, kappa_micro, f_micro_prod, 
                                          growth_speed, theta_ref, start_temp, dt, 
                                          micro_radius))

dofs_micro = micro_problem_list[0].V_micro.dim() # same for all cells

### Create vectors for writting the solution and rhs
petsc_vec = PETSc.Vec()
petsc_vec.create(PETSc.COMM_WORLD)
petsc_vec.setSizes(dofs_macro * (1 + dofs_micro))
petsc_vec.setUp()

u_sol_petsc = PETSc.Vec()
u_sol_petsc.create(PETSc.COMM_WORLD)
u_sol_petsc.setSizes(dofs_macro * (1 + dofs_micro))
u_sol_petsc.setUp()

### For saving
sol_file = File("Results/macro_solution.pvd")
macro_cond_scale_file = File("Results/macro_cond_scale.pvd")

radius_file = File("Results/radius_file.pvd")
radius_fn = Function(V_macro)
radius_fn.assign(Constant(micro_radius))

av_cell_temp_file = File("Results/average_cell_temp.pvd")
av_temp_fn = Function(V_macro)
av_temp_fn.assign(Constant(start_temp))

cell_temp_file = File("Results/example_cell_temp.pvd")
save_idx = dofs_macro - 8

### Save initial data
sol_file << (sol_function, t_n)
radius_file << (radius_fn, t_n)
av_cell_temp_file << (av_temp_fn, t_n)

Matrix_list = [[None for _ in range(dofs_macro + 1)] for _ in range(dofs_macro + 1)] 
rhs_vector_np = np.zeros(dofs_macro * (1 + dofs_micro))

# compute effective density once at the beginning
micro_cell_volume = np.pi * radius_fn.vector().get_local()**2
effective_density_fn.vector().set_local(1.0 - micro_cell_volume)

while t_n <= T_int[1] - dt/8.0:
    ### Update time variable and cells
    t_n += dt
    theta_old.assign(sol_function)

    if t_n > prod_stop:
        f_macro_prod.assign(0)

    print("Currently working on time step", t_n)

    ### Update effective density and conductivity on macro scale
    ## Conductivity with look up table
    new_values = np.interp(radius_fn.vector().get_local(), 
                           eff_cond_array[:, 0], eff_cond_array[:, 1])
    effective_cond_fn.vector().set_local(new_values)

    ## Compute eff. density by hand (since micro domain is a circle) and update
    ## the old one.
    old_effective_density_fn.assign(effective_density_fn)
    micro_cell_volume = np.pi * radius_fn.vector().get_local()**2
    effective_density_fn.vector().set_local(1.0 - micro_cell_volume)

    ### First we have to reconstruct the complete matrix.
    ### Start with diagonal blocks
    A_macro = PETScMatrix()
    F_macro = PETScVector()
    assemble_system(a_macro, f_macro, [bc_cool], A_tensor=A_macro, b_tensor=F_macro)

    ### Build macro matrix
    bi, bj, bv = A_macro.mat().getValuesCSR()
    M_macro = csr_matrix((bv, bj, bi))

    print("Start building matrix")
    start_time = time.time()
    for i in range(dofs_macro + 1):
        if i == 0:
            Matrix_list[i][i] = M_macro
            rhs_vector_np[:dofs_macro] = F_macro.get_local()
        else:
            # First compute micro values
            micro_problem_list[i-1].assemble_matrix_block()
            # Write into big matrix
            Matrix_list[i][i] = micro_problem_list[i-1].M_micro
            rhs_vector_np[dofs_macro+(i-1)*dofs_micro:dofs_macro+i*dofs_micro] = \
                                micro_problem_list[i-1].F_micro.get_local()

    M_complete = bmat(Matrix_list)
    #print("Size of complete system:", M_complete.shape)
    print("Took", time.time()- start_time)
    ## Build macro micro coupling
    M_lil = lil_matrix(M_complete)

    counter = 0
    print("Start coupling in matrix")
    start_time = time.time()
    for cell in cells(macro_mesh):
        # Find dofs that belong to this cell.
        # We indirectly set the ordering of the micro problems the same as the
        # macro dofs, e.g. dof=1 is the eqaul to the first micro block on the diagonal.
        dof_of_cell = dof_map.cell_dofs(cell.index())
        
        local_mass_matrix = assemble_local(inner(u_macro, phi_macro) * dx_macro,cell)
        
        for i in range(len(dof_of_cell)):
            row_dof = dof_of_cell[i]
            for j in range(len(dof_of_cell)):
                coloumn_dof = dof_of_cell[j]
                # Set heat exchange with micro cell
                M_lil[row_dof, dofs_macro + coloumn_dof * dofs_micro : 
                       dofs_macro + (coloumn_dof+1) * dofs_micro] += \
                            local_mass_matrix[i, j] * micro_problem_list[coloumn_dof].heat_exchange

            # Set BC for micro cell (theta_macro = theta_micro)
            M_lil[dofs_macro + row_dof * dofs_micro : dofs_macro + (row_dof+1) * dofs_micro, 
                   row_dof] = micro_problem_list[row_dof].BC_helper

    ## Show matrix        
    import matplotlib.pylab as plt
    # plt.spy(M_lil)
    # plt.show()
    print("Took", time.time()- start_time)
    ### Solve problem
    M_complete = csr_matrix(M_lil)

    petsc_mat = PETSc.Mat().createAIJ(size=M_complete.shape, 
                                    csr=(M_complete.indptr, M_complete.indices, M_complete.data))

    solver = PETSc.KSP().create()
    solver.setOperators(petsc_mat)
    solver.setType(PETSc.KSP.Type.GMRES) #PREONLY
    #solver.getPC().setType(PETSc.PC.Type.LU)

    petsc_vec.array[:] = rhs_vector_np
    start_time = time.time()
    print("Start solving")
    solver.solve(petsc_vec, u_sol_petsc)
    print("Solving is done")
    print("Took", time.time()- start_time)

    print("Extract macro solution")
    solution_array = u_sol_petsc.array
    sol_function.vector().set_local(solution_array[:dofs_macro])
    
    print("Update micro solution")
    start_time = time.time()
    for i in range(dofs_macro):
        ### First set the current solution and compute average
        micro_volume = assemble(1 * micro_problem_list[i].dx_micro)
        micro_problem_list[i].theta_old.vector().set_local(
            solution_array[dofs_macro+i*dofs_micro:dofs_macro+(i+1)*dofs_micro])

        micro_average = assemble(micro_problem_list[i].theta_old * 
                                 micro_problem_list[i].dx_micro) / micro_volume
        av_temp_fn.vector()[i] = micro_average

        ### Then update the cell
        micro_problem_list[i].update_cell(solution_array[i])
        radius_fn.vector()[i] = micro_problem_list[i].current_radius

        ### Save some micro solutions as well
        if i == save_idx:
            cell_temp_file << (micro_problem_list[i].theta_old, t_n)
    print("Took", time.time()- start_time)
    ### Save solutuion
    sol_file << (sol_function, t_n)
    radius_file << (radius_fn, t_n)
    av_cell_temp_file << (av_temp_fn, t_n)
    macro_cond_scale_file << (effective_cond_fn, t_n)