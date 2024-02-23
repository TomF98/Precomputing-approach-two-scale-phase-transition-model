import sys
import petsc4py
from petsc4py import PETSc
petsc4py.init(sys.argv) # needs to be done to run in parallel...

import time
from scipy.sparse import csr_matrix
from mpi4py import MPI as pyMPI
import numpy as np

from dolfin import *

from cell_problem_fixed import CellProblemFixedImplicit

# %%
### Parallel parameters
comm = pyMPI.COMM_WORLD
rank = comm.Get_rank()
num_of_processes = comm.Get_size()

path_to_save_folder = "Results"
path_to_data_folder = "Data"
path_to_mesh_folder = "MeshCreation"

### Problem parameters
kappa_macro = 0.1
kappa_micro = 0.1

rho_macro = 1.0
rho_micro = 1.0

theta_ref = 0.0
theta_start = 0.0
f_macro_prod = Expression("2.0*x[0]", degree=1)
f_micro_prod = Constant(0.0)

T_int = [0, 2.0]
dt = 0.25
t_n = T_int[0]

prod_stop = 0.5
f_stopper = Constant(1.0) # at start 1, if 0 no production

### Domain parameters
macro_size_x = 1
macro_size_y = 1

micro_radius = 0.2
growth_speed = 0.1

res_macro = 32

## Effective conductivity
eff_cond_array = np.load(path_to_data_folder + "/effective_conductivity.npy")

### Files to save solution data
energy_file             = File(path_to_save_folder + "/micro_energy_flow.pvd")
sol_file                = File(path_to_save_folder + "/macro_solution.pvd")
macro_cond_scale_file   = File(path_to_save_folder + "/macro_cond_scale.pvd")
radius_file             = File(path_to_save_folder + "/radius_file.pvd")
av_cell_temp_file       = File(path_to_save_folder + "/average_cell_temp.pvd")
cell_temp_file          = File(path_to_save_folder + "/example_cell_temp.pvd") 

cell_save_idx = res_macro**2 - 3
# %%
### Load/create mesh, function space and determine the splitting of dofs
### over all processecs
macro_mesh = UnitSquareMesh(MPI.comm_self, res_macro, res_macro) # is global

### Micro mesh and cell problems per process
#micro_mesh = UnitSquareMesh(MPI.comm_self, res_macro, res_macro)
micro_mesh = Mesh(MPI.comm_self)
with XDMFFile(MPI.comm_self, path_to_mesh_folder + "/micro_domain.xdmf") as infile:
    infile.read(micro_mesh)

## Dofs (only in case of linear function spaces)
dofs_micro = len(micro_mesh.coordinates())
dofs_macro = len(macro_mesh.coordinates())

## Macro functions space and size
V_macro = FunctionSpace(macro_mesh, "Lagrange", 1)
dx_macro = Measure("dx", V_macro)

dof_split = int(dofs_macro / (num_of_processes - 1)) 

## Process 0 does not get any micro problems and the last one gets the remaining ones
## if we cant perfectly devide
if rank == 0:
    dofs_macro_local = dofs_macro
    print(rank, dofs_macro)
    print("dofs micro", dofs_micro)
elif rank < num_of_processes - 1: 
    dofs_macro_local = dof_split
    print(rank, dofs_macro_local)
elif rank == num_of_processes - 1:
    dofs_macro_local = dofs_macro  - (num_of_processes - 2) * dof_split
    print(rank, dofs_macro_local)

## For scattering later on, we define what data section each process gets
## back from process 0
def create_scatter_idx(df_macro, df_micro, n_processes, df_split, global_idx=True):
    last_process_dofs = df_macro  - (n_processes - 2) * df_split
    
    if global_idx:
        range_list = [df_split*df_micro] * n_processes  
        range_list[-1] = last_process_dofs*df_micro
    else:
        range_list = [df_split] * n_processes  
        range_list[-1] = last_process_dofs        
    range_list[0] = df_macro

    displacement_list = []
    for j in range(n_processes):
        if j == 0:
            displacement_list.append(0)
        else:
            if global_idx:
                displacement_list.append(df_macro + (j-1)*df_split*df_micro) 
            else:
                displacement_list.append((j-1)*df_split) 
    
    return range_list, displacement_list

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
    theta_iteration             = Function(V_macro)
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

    a_macro = inner(kappa_macro * effective_cond_fn * grad(u_macro), 
                    grad(phi_macro)) * dx_macro
    a_macro += inner(rho_macro * effective_density_fn * u_macro, phi_macro) / dt * dx_macro

    f_macro = (inner(rho_macro * old_effective_density_fn * theta_old, phi_macro) / dt \
            + f_stopper * inner(f_macro_prod, phi_macro)) * dx_macro

    ## dummy cell to determine cells dofs that should be coupled with the
    ## Dirichlet condition
    dummy_cell = CellProblemFixedImplicit(micro_mesh, kappa_micro, 
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

    new_values = np.interp(radius_fn.vector().get_local(), 
                           eff_cond_array[:, 0], eff_cond_array[:, 1])
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
            CellProblemFixedImplicit(micro_mesh, kappa_micro, 
                                     f_micro_prod, growth_speed, 
                                     theta_ref, theta_start, dt, 
                                     micro_radius, rho_micro))

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

tol = 1.e-4

while t_n <= T_int[1] - dt/8.0:
    t_n += dt
    current_error = 1

    ## Update previous data functions
    if rank == 0:
        theta_old.assign(sol_fn)
        theta_iteration.assign(sol_fn)

        print("Currently working on time step", t_n)
        if t_n > prod_stop:
            f_stopper.assign(0.0)
    
    iteration_counter = 0
    while current_error > tol:
        iteration_counter += 1
        ### 1) Build matrix
        ### Macro domain
        if rank == 0:
            ## Build matrix
            start_time = time.time()
            A_macro, F_macro = assemble_system(a_macro, f_macro)

            non_zero_rows, non_zero_cols = np.nonzero(A_macro.array())
            non_zero_data = A_macro.array()[non_zero_rows, non_zero_cols]
            #print(non_zero_data)

            dirichlet_coupling_idx = np.nonzero(dummy_dirichlet_helper)[0]
            len_coupling = len(dirichlet_coupling_idx)
            if first_time_step:
                dirichlet_coupling_rows = np.zeros(len_coupling * dofs_macro, dtype=np.int32)
                dirichlet_coupling_cols = np.zeros(len_coupling * dofs_macro, dtype=np.int32)
                dirichlet_coupling_data = np.zeros(len_coupling * dofs_macro)
                for dof in range(dofs_macro):
                    dirichlet_coupling_rows[dof * len_coupling : (dof+1) * len_coupling] = \
                        dirichlet_coupling_idx + dof * dofs_micro + dofs_macro
                    dirichlet_coupling_cols[dof * len_coupling : (dof+1) * len_coupling] = dof
                    dirichlet_coupling_data[dof * len_coupling : (dof+1) * len_coupling] = -1

            if first_time_step:
                non_zero_rows = np.concatenate((non_zero_rows, dirichlet_coupling_rows), dtype=np.int32)
                non_zero_cols = np.concatenate((non_zero_cols, dirichlet_coupling_cols), dtype=np.int32)
            non_zero_data = np.concatenate((non_zero_data, dirichlet_coupling_data))

            rhs_vec = F_macro.get_local() 

        ### Cell problems
        else:
            ## Assemble cells on other process including coupling back
            ## to the macro scale
            read_idx_shift = dofs_macro + (rank - 1) * dof_split
            idx_shift = dofs_macro + (rank - 1) * dof_split * dofs_micro

            micro_problem_list[0].assemble_matrix_block()
            heat_exchange_idx = np.nonzero(micro_problem_list[0].heat_exchange)[0]

            non_zero_rows = []
            non_zero_cols = []
            
            # this has to be compute in each iteration
            non_zero_data = []
            rhs_vec = []

            for counter in range(dofs_macro_local):
                # Diagonal block for cell problem 

                micro_problem_list[counter].assemble_matrix_block()

                if counter == 0: # only once at the start
                    cell_non_zero_rows, cell_non_zero_cols = \
                        np.nonzero(micro_problem_list[counter].M_micro.array())

                if first_time_step:
                    non_zero_rows.append(cell_non_zero_rows + idx_shift + counter * dofs_micro)
                    non_zero_cols.append(cell_non_zero_cols + idx_shift + counter * dofs_micro)
                non_zero_data.append(
                    micro_problem_list[counter].M_micro.array()[cell_non_zero_rows, 
                                                                cell_non_zero_cols])
                
                ## rhs of cell problem
                rhs_vec.append(micro_problem_list[counter].F_micro.get_local())

                ## Coupling from micro to macro
                coupling_dofs, mass_values = global_mass_matrix.getrow(
                                                read_idx_shift + counter - dofs_macro)

                dof_counter = 0
                for couple_dof in coupling_dofs:
                    if first_time_step:
                        non_zero_rows.append(np.ones_like(heat_exchange_idx) * couple_dof)
                        non_zero_cols.append(heat_exchange_idx + idx_shift + counter * dofs_micro)
                    non_zero_data.append(
                        mass_values[dof_counter] * micro_problem_list[counter].heat_exchange[heat_exchange_idx])

                    dof_counter += 1

            if first_time_step:
                non_zero_rows = np.concatenate(non_zero_rows, dtype=np.int32)
                non_zero_cols = np.concatenate(non_zero_cols, dtype=np.int32)
            non_zero_data = np.concatenate(non_zero_data)
            rhs_vec = np.concatenate(rhs_vec)

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


        #print(rank, "building matrix, waiting...")
        comm.barrier()

        #%%
        ### 2) Start solving 
        if rank == 0:
            ### Building is easier with scipy (docu not clear with PETSc)
            ### but solving is faster with PETSc. (Transformation is also fast)
            petsc_vec = PETSc.Vec()
            petsc_vec.create(PETSc.COMM_SELF)
            petsc_vec.setSizes(dofs_macro * (1 + dofs_micro))
            petsc_vec.setUp()

            u_sol_petsc = PETSc.Vec()
            u_sol_petsc.create(PETSc.COMM_SELF)
            u_sol_petsc.setSizes(dofs_macro * (1 + dofs_micro))
            u_sol_petsc.setUp()

            petsc_mat = PETSc.Mat().createAIJ(size=M_macro.shape, 
                    csr=(M_macro.indptr, M_macro.indices, M_macro.data), 
                    comm=PETSc.COMM_SELF)

            print("building matrix took", time.time() - start_time)

            ## Select solver
            solver = PETSc.KSP().create(PETSc.COMM_SELF)
            solver.setOperators(petsc_mat)
            solver.setType(PETSc.KSP.Type.GMRES) #PREONLY
            #solver.getPC().setType(PETSc.PC.Type.LU)

            petsc_vec.array[:] = global_rhs_vec

            print("Start solving")
            start_time = time.time()
            solver.solve(petsc_vec, u_sol_petsc)
            print("Solving is done, took", time.time()- start_time)

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
            theta_iteration.assign(sol_fn)
            sol_fn.vector().set_local(macro_solution_array)

            data_for_macro_domain = np.zeros((1, 3))

            error = assemble(inner(theta_iteration-sol_fn, 
                                    theta_iteration-sol_fn)*dx_macro)

            data_for_macro_domain[0, 2] = error
            print("waiting for update of cells")
        else:
            ## Data to send to the macro domain:
            ## current temperature intergral, radius after cell movement, iteration diff
            data_for_macro_domain = np.zeros((dofs_macro_local, 3))
            for i in range(dofs_macro_local):
                ## First set the current solution and compute average temperature
                cell_diff = micro_problem_list[i].update_iteration(
                                macro_solution_array[i],
                                local_solution_array[i*dofs_micro:(i+1)*dofs_micro])

                data_for_macro_domain[i, 0] = micro_problem_list[i].current_energy
                data_for_macro_domain[i, 1] = micro_problem_list[i].iteration_radius
                data_for_macro_domain[i, 2] = cell_diff


        comm.barrier()

        #%%
        ### Finally update the effective parameters and save solution
        effective_data = comm.gather(data_for_macro_domain, root=0)
        if rank == 0:
            effective_data = np.concatenate(effective_data[1:])

            average_temp_fn.vector().set_local(effective_data[:, 0])
            radius_fn.vector().set_local(effective_data[:, 1])
            
            ## Update effective data
            new_values = np.interp(radius_fn.vector().get_local(), 
                                    eff_cond_array[:, 0], eff_cond_array[:, 1])
            effective_cond_fn.vector().set_local(new_values)

            micro_cell_volume = np.pi * radius_fn.vector().get_local()**2
            effective_density_fn.vector().set_local(1.0 - micro_cell_volume)

            ## compute iteration difference:
            current_error = np.sqrt(np.mean(effective_data[:, 2]))
            print("In iteration", iteration_counter, 
                  "had difference", current_error)
        
        #print(rank, "waiting for saving")
        comm.barrier()

        current_error = comm.bcast(current_error, root=0)
        first_time_step = False
    
    ### When iteration converged -> save solution and set up next time step
    if rank == 0:
        old_effective_density_fn.assign(effective_density_fn)

        ## Compute energy and compare
        energy_array[energy_idx, 0] = t_n
        expected_energy += dt * assemble(f_stopper * f_macro_prod * dx_macro)
        macro_energy = assemble(rho_macro * effective_density_fn * sol_fn * dx_macro)
        micro_energy = assemble(rho_micro * average_temp_fn * dx_macro)
        #micro_energy += dt * assemble(energy_flow * dx_macro)
        print("Expected Energy:", expected_energy)
        num_energy = macro_energy + micro_energy
        print("Numeric Energy:", num_energy)
        print("Rel diff:", np.abs(expected_energy - (num_energy)) / expected_energy)

        energy_array[energy_idx, 1] = expected_energy
        energy_array[energy_idx, 2] = macro_energy
        energy_array[energy_idx, 3] = micro_energy
        energy_idx += 1
        ## Save solution
        sol_file << (sol_fn, t_n)
        radius_file << (radius_fn, t_n)
        av_cell_temp_file << (average_temp_fn, t_n)
        macro_cond_scale_file << (effective_cond_fn, t_n)
        print("waiting for saving")
    else:
        for i in range(dofs_macro_local):
            cell_diff = micro_problem_list[i].update_cell()
            if i + (rank - 1) * dof_split == cell_save_idx:
                cell_temp_file << (micro_problem_list[i].theta_old, t_n)
    
    comm.barrier()