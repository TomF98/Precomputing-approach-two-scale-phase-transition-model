import sys
import petsc4py
petsc4py.init(sys.argv) # needs to be done to run in parallel...

import time
from scipy.sparse import csr_matrix
from mpi4py import MPI as pyMPI
import numpy as np

from dolfin import *

from MicroProblems.cell_problem_fixed import CellProblemFixed
from Utils.helper_functions import (create_scatter_idx, BuildMacroMatrix, BuildMicroMatrix, 
                                    SolveSystem, interpolate_conductivity, CreateSolver)
from Utils.heat_soruce import HeatSource
"""
The main implementation for the code examples.
"""

# %%
### Parallel parameters
comm = pyMPI.COMM_WORLD
rank = comm.Get_rank()
num_of_processes = comm.Get_size()

path_to_save_folder = "Results/Example"
path_to_data_folder = "InterpolationData"
path_to_mesh_folder = "MeshCreation"

### Problem parameters
kappa_macro = 0.1
kappa_micro = 0.1

rho_macro = 1.0
rho_micro = 1.0

theta_ref = 0.0
theta_start = theta_ref

T_int = [0, 10.0]
dt = 0.1
t_n = T_int[0]
prod_stop = 5.0

f_macro_prod = HeatSource(t_n, prod_stop, 0.75)
f_micro_prod = Constant(0.0)

latent_heat = 2*np.pi

### Domain parameters
macro_size_x = 1
macro_size_y = 1

micro_radius = 0.25 # original radius
micro_radius += dt * theta_start # extrapolate radius at first step
growth_speed = 0.1 # additional parameter to controll growth of cells

res_macro = 32
res_micro = 0.05

## Effective conductivity
cond_res = 320 # resolution of precompute
use_quadratic_interpolation = False # use qudratic interpolation (else piecewise constant)
use_linear_interpolation = True # use linear interpolation (else piecewise constant)
eff_cond_array = np.load(path_to_data_folder + "/higher_res_effective_conductivity_res_" + str(cond_res) + ".npy")

### Files to save solution data
energy_file             = File(path_to_save_folder + "/micro_energy_flow.pvd")
sol_file                = File(path_to_save_folder + "/macro_solution.pvd")
macro_cond_scale_file   = File(path_to_save_folder + "/macro_cond_scale.pvd")
radius_file             = File(path_to_save_folder + "/radius_file.pvd")
av_cell_temp_file       = File(path_to_save_folder + "/average_cell_temp.pvd")

cell_save_idx = [0, 100, res_macro**2-1] # values between 0 and res_macro**2-1 
cell_save_dic = {}

# %%
### Load/create mesh, function space and determine the splitting of dofs
### over all processecs
macro_mesh = UnitSquareMesh(MPI.comm_self, res_macro, res_macro) # is global

macro_coords = macro_mesh.coordinates()
for i in cell_save_idx:
    x_coord = f"{macro_coords[i][0]:0.2f}"
    y_coord = f"{macro_coords[i][1]:0.2f}"
    cell_path = f"/example_cell_temp_idx_{i}_x_{x_coord}_y_{y_coord}.pvd"
    cell_save_dic[i] = File(path_to_save_folder + cell_path) 


### Micro mesh and cell problems per process
micro_mesh = Mesh(MPI.comm_self)
with XDMFFile(MPI.comm_self, path_to_mesh_folder + "/micro_domain_res_" + str(res_micro) + ".xdmf") as infile:
    infile.read(micro_mesh)


## Dofs (only in case of linear function spaces)
dofs_micro = len(micro_mesh.coordinates())
dofs_macro = len(macro_mesh.coordinates())

print("Mesh size")
print("Macro", macro_mesh.hmax())
print("Micro", micro_mesh.hmax())

## Macro functions space and size
V_macro = FunctionSpace(macro_mesh, "Lagrange", 1)
dx_macro = Measure("dx", V_macro)

dof_split = int(dofs_macro / (num_of_processes - 1)) 

## Process 0 does not get any micro problems and the last one gets the remaining ones
## if we cant perfectly devide
if rank == 0:
    dofs_macro_local = dofs_macro
    print("Dofs macro domain", dofs_macro)
    print("Dofs of each micro cell", dofs_micro)
    print("Cells per process:")
elif rank < num_of_processes - 1: 
    dofs_macro_local = dof_split
    print(rank, dofs_macro_local)
elif rank == num_of_processes - 1:
    dofs_macro_local = dofs_macro  - (num_of_processes - 2) * dof_split
    print(rank, dofs_macro_local)

global_range_list, global_displacement_list = create_scatter_idx(
                    dofs_macro, dofs_micro, num_of_processes, dof_split)

macro_range_list, macro_displacement_list = create_scatter_idx(
                    dofs_macro, dofs_micro, num_of_processes, dof_split, False)

# %%
### Define the weak formulation on the macro domain and create the cell problems
## All macro parameters for rank 0
if rank == 0:
    ## Functions
    theta_old                   = Function(V_macro)
    effective_cond_fn           = Function(V_macro)
    effective_density_fn        = Function(V_macro)
    old_effective_density_fn    = Function(V_macro)
    sol_fn                      = Function(V_macro)
    radius_fn                   = Function(V_macro)
    average_temp_fn             = Function(V_macro)

    # Data arrays to save stuff.
    # Energy in form: time, expected energy, macro energy, micro energy
    energy_array = np.zeros((int(T_int[1]/dt)+1, 4)) 
    energy_idx = 0
    expected_energy = 0

    ## Weak form
    u_macro = TrialFunction(V_macro)
    phi_macro = TestFunction(V_macro)

    a_macro = 0.5*inner(kappa_macro * effective_cond_fn * grad(u_macro), 
                        grad(phi_macro)) * dx_macro
    a_macro += inner(rho_macro * effective_density_fn * u_macro, phi_macro) / dt * dx_macro

    a_macro += growth_speed * latent_heat * u_macro * phi_macro * radius_fn * dx_macro

    f_macro = (inner(rho_macro * old_effective_density_fn * theta_old, phi_macro) / dt \
            + inner(f_macro_prod, phi_macro)) * dx_macro \
            - 0.5*inner(kappa_macro * effective_cond_fn * grad(theta_old), 
                        grad(phi_macro)) * dx_macro

    ## dummy cell to determine cells dofs that should be coupled with the
    ## Dirichlet condition
    dummy_cell = CellProblemFixed(micro_mesh, kappa_micro, 
                                  f_micro_prod, growth_speed, 
                                  theta_ref, theta_start, dt, 
                                  micro_radius, rho_micro)

    dummy_dirichlet_helper = dummy_cell.BC_helper.get_local()
    num_bc_elements = len(np.nonzero(dummy_dirichlet_helper)[0])
    dummy_cell = None

    ## Setup everything for the first time step
    # initial temperature, radius, conductivity and 
    theta_old.assign(Constant(theta_start))
    average_temp_fn.assign(Constant(theta_start))
    radius_fn.assign(Constant(micro_radius))

    new_values = interpolate_conductivity(radius_fn.vector().get_local(), 
                                                eff_cond_array[:, 0], 
                                                eff_cond_array[:, 1], 
                                                use_linear_interpolation, 
                                                use_quadratic_interpolation) 
    effective_cond_fn.vector().set_local(new_values)

    micro_cell_volume = np.pi * radius_fn.vector().get_local()**2
    effective_density_fn.vector().set_local(1.0 - micro_cell_volume)
    old_effective_density_fn.assign(effective_density_fn)

    sol_file << (sol_fn, t_n)
    radius_file << (radius_fn, t_n)
    av_cell_temp_file << (average_temp_fn, t_n)
    macro_cond_scale_file << (effective_cond_fn, t_n)

## All other processecs handle the cell problems
else:
    ## Will need the mass matrix to construc the coupling
    u_macro = TrialFunction(V_macro)
    phi_macro = TestFunction(V_macro)
    global_mass_matrix = assemble(u_macro * phi_macro * dx_macro)

    ## Cell problems on each process
    micro_problem_list = []

    for _ in range(dofs_macro_local):
        micro_problem_list.append(
            CellProblemFixed(micro_mesh, kappa_micro, f_micro_prod, growth_speed, 
                             theta_ref, theta_start, dt, micro_radius, rho_micro))

    num_bc_elements = len(np.nonzero(micro_problem_list[0].BC_helper.get_local())[0])


print(rank, "waiting for macro- and cell-problem definition")
comm.barrier()
# %%
### Now do time stepping:
### 1) Build the matrix for the given time step in parallel for macro and cells
### 2) Solve the complete system
### 3) Update cell information
### 4) Update effective parameters on macro scale (density, conductivity, ...)
### 5) Save data

first_time_step = True # some stuff stays the same between iterations
dirichlet_coupling_data = None
while t_n <= T_int[1] - dt/8.0:
    t_n += dt
    ### 1) Build matrix
    ### Macro domain
    if rank == 0:
        print("Currently working on time step", t_n)
        f_macro_prod.t = t_n
        
        ## Update previous data functions
        theta_old.assign(sol_fn)

        ## Build matrix
        start_time = time.time()
        non_zero_rows, non_zero_cols, non_zero_data, rhs_vec, dirichlet_coupling_data = \
            BuildMacroMatrix(dofs_micro, dofs_macro, a_macro, f_macro, 
                             dummy_dirichlet_helper, first_time_step, 
                             dirichlet_coupling_data) 

    ### Cell problems
    else:
        ## Assemble cells on other process including coupling back
        ## to the macro scale
        non_zero_rows, non_zero_cols, non_zero_data, rhs_vec = \
            BuildMicroMatrix(rank, dofs_micro, dofs_macro, dof_split, dofs_macro_local, 
                             global_mass_matrix, micro_problem_list, first_time_step)

    # %%
    #print(rank, "waiting for block matrix creation")
    comm.barrier()

    ### Bring data to process 0 that will build the complete matrix
    if first_time_step:
        global_non_zero_rows = comm.gather(non_zero_rows, root=0)
        global_non_zero_cols = comm.gather(non_zero_cols, root=0)
    # Matrix data and rhs changes in each iteration
    global_non_zero_data = comm.gather(non_zero_data, root=0)
    global_rhs_vec = comm.gather(rhs_vec, root=0)

    # Afterwards we can delete local memory space
    non_zero_rows = None
    non_zero_cols = None
    non_zero_data = None
    rhs_vec = None

    if rank == 0:
        if first_time_step:
            global_non_zero_rows = np.concatenate(global_non_zero_rows)
            global_non_zero_cols = np.concatenate(global_non_zero_cols)
        
        global_non_zero_data = np.concatenate(global_non_zero_data)
        global_rhs_vec = np.concatenate(global_rhs_vec)

        # print(global_non_zero_rows.dtype, global_non_zero_cols.dtype)
        # print(global_non_zero_rows.shape, global_non_zero_cols.shape, global_non_zero_data.shape)
        M_macro = csr_matrix((global_non_zero_data, (global_non_zero_rows, global_non_zero_cols)),
                            shape=[dofs_macro*(1+dofs_micro), dofs_macro*(1+dofs_micro)])
        ##Show matrix
        # import matplotlib.pylab as plt
        # plt.spy(M_macro)
        # plt.show()
        # sparse.save_npz("matrix_p.npz", M_macro)
        # exit()

    comm.barrier()

    #%%
    ### 2) Start solving 
    if rank == 0:
        if first_time_step:
            petsc_vec, u_sol_petsc, solver = CreateSolver(dofs_micro, dofs_macro)
        u_sol_petsc = SolveSystem(petsc_vec, u_sol_petsc, solver, 
                                  global_rhs_vec, M_macro, start_time)
        solution_array = u_sol_petsc.array
        macro_array = solution_array[:dofs_macro]
    else:
        solution_array = None
        macro_array = None

    #print(rank, "waiting for solution")
    comm.barrier()

    #%%
    ### 3) Scatter solution data from 0 to the other processes
    ### and update cells
    local_solution_array = np.zeros(dofs_macro_local*dofs_micro)
    macro_solution_array = np.zeros(dofs_macro_local)

    comm.Scatterv([solution_array, global_range_list, global_displacement_list, pyMPI.DOUBLE],
                local_solution_array, root=0)

    #print(macro_displacement_list, macro_range_list)
    comm.Scatterv([macro_array, macro_range_list, macro_displacement_list, pyMPI.DOUBLE],
                macro_solution_array, root=0)
    #print("macro sol", len(macro_solution_array))
    if rank == 0:
        start_time = time.time()
        sol_fn.vector().set_local(macro_solution_array)

        data_for_macro_domain = np.zeros((1, 2)) # dummy
        print("Waiting for update of cells")
    else:
        ## Data to send to the macro domain:
        ## current temperature intergral, radius after cell movement
        data_for_macro_domain = np.zeros((dofs_macro_local, 2))
        for i in range(dofs_macro_local):
            ## First set the current solution and compute average temperature
            micro_problem_list[i].update_cell(
                macro_solution_array[i],
                local_solution_array[i*dofs_micro:(i+1)*dofs_micro])

            data_for_macro_domain[i, 0] = micro_problem_list[i].current_energy
            data_for_macro_domain[i, 1] = micro_problem_list[i].current_radius
            #print(cell_save_idx, i + read_idx_shift)
            if (i + (rank - 1) * dof_split) in cell_save_dic:
                cell_save_dic[i + (rank - 1) * dof_split] << (micro_problem_list[i].theta_old, t_n)

    comm.barrier()

    #%%
    ### Finally update the effective parameters and save solution
    effective_data = comm.gather(data_for_macro_domain, root=0)
    if rank == 0:
        effective_data = np.concatenate(effective_data[1:])

        average_temp_fn.vector().set_local(effective_data[:, 0] / rho_micro)
        radius_fn.vector().set_local(effective_data[:, 1])

        ## Compute energy and compare
        energy_array[energy_idx, 0] = t_n
        expected_energy += dt * assemble(f_macro_prod * dx_macro)
        macro_energy = assemble(rho_macro * effective_density_fn * sol_fn * dx_macro)
        micro_energy = assemble(rho_micro * average_temp_fn * dx_macro)
        #micro_energy += dt * assemble(energy_flow * dx_macro)
        print("Produced Energy:", expected_energy)
        num_energy = macro_energy + micro_energy
        print("Numeric Energy:", num_energy)
        #print("Rel diff:", np.abs(expected_energy - (num_energy)) / expected_energy)

        energy_array[energy_idx, 1] = expected_energy
        energy_array[energy_idx, 2] = macro_energy
        energy_array[energy_idx, 3] = micro_energy
        energy_idx += 1
        
        ## Update effective data
        new_values = interpolate_conductivity(radius_fn.vector().get_local(), 
                                                eff_cond_array[:, 0], 
                                                eff_cond_array[:, 1], 
                                                use_linear_interpolation,
                                                use_quadratic_interpolation) 
        effective_cond_fn.vector().set_local(new_values)

        old_effective_density_fn.assign(effective_density_fn)
        micro_cell_volume = np.pi * radius_fn.vector().get_local()**2
        effective_density_fn.vector().set_local(1.0 - micro_cell_volume)

        ## Save solution
        sol_file << (sol_fn, t_n)
        radius_file << (radius_fn, t_n)
        av_cell_temp_file << (average_temp_fn, t_n)
        macro_cond_scale_file << (effective_cond_fn, t_n)

        print("Updating cells and saving took:", time.time() - start_time)
    
    #print(rank, "waiting for saving")
    comm.barrier()

    first_time_step = False