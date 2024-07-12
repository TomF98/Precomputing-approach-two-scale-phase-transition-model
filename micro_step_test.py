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

# %%
### Parallel parameters
comm = pyMPI.COMM_WORLD
rank = comm.Get_rank()
num_of_processes = comm.Get_size()

path_to_save_folder = "Results/MicroResults"
path_to_data_folder = "InterpolationData/higher_res_"
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
prod_stop = 5.0

f_micro_prod = Constant(0.0)

latent_heat = 2*np.pi

### Domain parameters
macro_size_x = 1
macro_size_y = 1

micro_radius_start = 0.25
growth_speed = 0.1

res_macro = 32

H_micro_list = [0.004375, 0.00875, 0.0175, 0.025, 0.05, 0.1, 0.25, 0.5] # 0.0125/4.0, 0.0125/2.0, 0.0125, 
error_list = [[] for _ in range(len(H_micro_list) - 1)]
radius_error_list = [[] for _ in range(len(H_micro_list) - 1)]
grad_error_list = [[] for _ in range(len(H_micro_list) - 1)]
error_list_micro = [[] for _ in range(len(H_micro_list) - 1)]
grad_error_list_micro = [[] for _ in range(len(H_micro_list) - 1)]

fine_sol_list = []
fine_radius_list = []
fine_sol_list_micro = []

time_list = []

use_linear_interpolation = False # use linear interpolation 
use_quadratic_interpolation = True # use linear interpolation (else piecewise constant) 

macro_mesh = UnitSquareMesh(MPI.comm_self, res_macro, res_macro)
V_macro = FunctionSpace(macro_mesh, "Lagrange", 1)
dx_macro = Measure("dx", V_macro)

### Micro mesh and cell problems per process
micro_mesh_high = Mesh(MPI.comm_self)
with XDMFFile(MPI.comm_self, path_to_mesh_folder + "/micro_domain_res_" + str(H_micro_list[0]) + ".xdmf") as infile:
    infile.read(micro_mesh_high)

V_micro_heigh   = FunctionSpace(micro_mesh_high, "Lagrange", 1)
helper_fn_micro = Function(V_micro_heigh)
dx_micro_heigh  = Measure("dx", V_micro_heigh)
## Dofs (only in case of linear function spaces)
dofs_macro = len(macro_mesh.coordinates())

## Macro functions space and size
dof_split = int(dofs_macro / (num_of_processes - 1)) 

counter = -1

for micro_res in H_micro_list:
    counter += 1
    save_counter = 0
    t_n = T_int[0]
    f_macro_prod = HeatSource(t_n, prod_stop, 0.75)

    micro_radius = micro_radius_start
    micro_radius += dt * theta_start # extrapolate radius at first step

    ## Effective conductivity
    eff_cond_array = np.load(f"{path_to_data_folder}effective_conductivity_res_320.npy")

    ### Micro mesh and cell problems per process
    if counter == 0:
        micro_mesh = micro_mesh_high
    else:
        micro_mesh = Mesh(MPI.comm_self)
        with XDMFFile(MPI.comm_self, path_to_mesh_folder + "/micro_domain_res_" + str(micro_res) + ".xdmf") as infile:
            infile.read(micro_mesh)

    dofs_micro = len(micro_mesh.coordinates())
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
        helper_error_fn             = Function(V_macro)

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

        # sol_file << (sol_fn, t_n)
        # radius_file << (radius_fn, t_n)
        # av_cell_temp_file << (average_temp_fn, t_n)
        # macro_cond_scale_file << (effective_cond_fn, t_n)

    ## All other processecs handle the cell problems
    else:
        ## Will need the mass matrix to construc the coupling
        u_macro = TrialFunction(V_macro)
        phi_macro = TestFunction(V_macro)
        global_mass_matrix = assemble(u_macro * phi_macro * dx_macro)

        ## Cell problems on each process
        micro_problem_list = []

        for _ in range(dofs_macro_local):
            if counter == 0:
                micro_problem_list.append(
                    CellProblemFixed(micro_mesh, kappa_micro, f_micro_prod, growth_speed, 
                                    theta_ref, theta_start, dt, micro_radius, rho_micro, V_micro_heigh))
            else:
                micro_problem_list.append(
                    CellProblemFixed(micro_mesh, kappa_micro, f_micro_prod, growth_speed, 
                                    theta_ref, theta_start, dt, micro_radius, rho_micro))                

        num_bc_elements = len(np.nonzero(micro_problem_list[0].BC_helper.get_local())[0])


    #print(rank, "waiting for macro- and cell-problem definition")
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
    time_counter = 0
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
                
        comm.barrier()

        #%%
        ### Finally update the effective parameters and save solution
        effective_data = comm.gather(data_for_macro_domain, root=0)

        save_counter += 1
        if rank == 0:
            effective_data = np.concatenate(effective_data[1:])

            average_temp_fn.vector().set_local(effective_data[:, 0] / rho_micro)
            radius_fn.vector().set_local(effective_data[:, 1])
            
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

            if counter == 0:
                fine_sol_list.append(sol_fn.copy(True))
                fine_radius_list.append(radius_fn.copy(True))
                time_list.append(t_n)
                #print(len(time_list))
                
            elif save_counter:
                current_error = np.sqrt(assemble(inner(sol_fn - fine_sol_list[time_counter], 
                                                    sol_fn - fine_sol_list[time_counter])  * dx_macro))
                error_list[counter - 1].append(current_error)
                current_grad_error = np.sqrt(assemble(inner(grad(sol_fn - fine_sol_list[time_counter]), 
                                                grad(sol_fn - fine_sol_list[time_counter]))  * dx_macro))
                grad_error_list[counter - 1].append(current_grad_error)

                radius_error = np.sqrt(assemble(inner(radius_fn - fine_radius_list[time_counter], 
                                                    radius_fn - fine_radius_list[time_counter]) * dx))
                radius_error_list[counter - 1].append(radius_error)
                
                time_counter += 1

                error_from_micro_domain = np.zeros((1, 2)) # dummy

        else:
            # Save fine solution
            if counter == 0:
                fine_sol_list_micro.append([])
                for i in range(dofs_macro_local):
                    fine_sol_list_micro[-1].append(micro_problem_list[i].theta_old.copy(True))
                

            # Compute error
            else:
                error_from_micro_domain = np.zeros((dofs_macro_local, 2))
                for i in range(dofs_macro_local):
                    micro_problem_list[i].theta_old.set_allow_extrapolation(True)
                    helper_fn_micro = interpolate(micro_problem_list[i].theta_old, V_micro_heigh)
                    current_diff = helper_fn_micro - fine_sol_list_micro[time_counter][i]
                    error_from_micro_domain[i, 0] = \
                        assemble(inner(current_diff, current_diff) * dx_micro_heigh)

                    error_from_micro_domain[i, 1] = \
                        assemble(inner(grad(current_diff), 
                                       grad(current_diff)) * dx_micro_heigh)
                
                time_counter += 1


        comm.barrier()

        if counter > 0:
            micro_error = comm.gather(error_from_micro_domain, root=0)

            if rank == 0:
                micro_error = np.concatenate(micro_error[1:])

                # First function error:
                helper_error_fn.vector().set_local(micro_error[:, 0])
                error_list_micro[counter - 1].append(np.sqrt(assemble(helper_error_fn * dx_macro)))

                # Next gradient error:
                helper_error_fn.vector().set_local(micro_error[:, 1])
                grad_error_list_micro[counter - 1].append(np.sqrt(assemble(helper_error_fn * dx_macro)))

                print("Updating cells and saving took:", time.time() - start_time)

        comm.barrier()

        first_time_step = False
        
if rank==0:
    error_list = np.array(error_list)
    np.save(f"{path_to_save_folder}/H_error_micro", error_list_micro)
    np.save(f"{path_to_save_folder}/H_grad_error_micro", grad_error_list_micro)
    np.save(f"{path_to_save_folder}/H_error", error_list)
    np.save(f"{path_to_save_folder}/H_grad_error", grad_error_list)
    np.save(f"{path_to_save_folder}/time_list", np.array(time_list))
    np.save(f"{path_to_save_folder}/radius_error", np.array(radius_error_list))