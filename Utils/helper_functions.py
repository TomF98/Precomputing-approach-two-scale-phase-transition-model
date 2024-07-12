import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
import time
from petsc4py import PETSc
from dolfin import *

def create_scatter_idx(df_macro, df_micro, n_processes, df_split, global_idx=True):
    """
    For scattering data from process 0 to other processes. Returns what data section
    each process gets.

    Parameters
    ==========
    df_macro : int
        Number of macro dofs.
    df_micro : int
        Number of micro dofs.
    n_processes : int
        Number of processes.
    df_split : int
        How many cells each process handles.
    global_idx : bool
        Process 0 needs to not shift its data.
    """
    # last process has more cells:
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


def interpolate_conductivity(current_radius, radius_values, cond_values, linear_interpolation : bool,
                             quadratic_interpolation : bool):
    """ Does the interpolation for the precomputing.

    Parameters
    ==========
    current_radius : np.array
        The current local radius values of the micro cells.
    radius_values : np.array
        The array containing the radius-values used for the precomputing.
    cond_values : np.array
        The coressponding effective conducitivity values for the given radius.
    linear_interpolation : bool
        Wheter linear interpolation should be used.
    quadratic_interpolation : bool
        Wheter quadratic interpolation should be used. If neither linear nor 
        quadratic we use piece wise continuous.
    """
    if linear_interpolation:
        return np.interp(current_radius, radius_values, cond_values)
    
    if quadratic_interpolation:
        f = interp1d(radius_values, cond_values, kind="quadratic", bounds_error=False,
                    fill_value=(cond_values[0], cond_values[-1]))
    else:
        f = interp1d(radius_values, cond_values, kind="zero", bounds_error=False,
                    fill_value=(cond_values[0], cond_values[-1]))
    return f(current_radius)


def BuildMacroMatrix(dofs_micro, dofs_macro, a_macro, f_macro, dummy_dirichlet_helper, first_time_step,
                     dirichlet_coupling_data=None):
    """
    Build the macroscopic matrix block in each time step.

    Parameters
    ==========
    dofs_macro : int
        Number of macro dofs.
    dofs_micro : int
        Number of micro dofs.  
    a_macro : form
        The bilinear form of the macro domain.
    f_macro : form
        The rhs of the macro domain.
    dummy_dirichlet_helper : vector
        The microscopic boundary nodes.
    first_time_step : bool
        At the first step we need to compute what entries have non zero values.
        This does not change later on, and needs to be computed only once.   
    """
    A_macro, F_macro = assemble_system(a_macro, f_macro)

    non_zero_rows, non_zero_cols = np.nonzero(A_macro.array())
    non_zero_data = A_macro.array()[non_zero_rows, non_zero_cols]
    #print(non_zero_data)

    # Find Dirichelt coupling indicies on micro mesh
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

    return non_zero_rows, non_zero_cols, non_zero_data, F_macro.get_local(), dirichlet_coupling_data


def BuildMicroMatrix(rank, dofs_micro, dofs_macro, dof_split, dofs_macro_local, 
                     global_mass_matrix, micro_problem_list, first_time_step):
    read_idx_shift = dofs_macro + (rank - 1) * dof_split
    idx_shift = dofs_macro + (rank - 1) * dof_split * dofs_micro

    micro_problem_list[0].assemble_matrix_block()
    heat_exchange_idx = np.nonzero(micro_problem_list[0].heat_exchange)[0]

    non_zero_rows = []
    non_zero_cols = []
        
    # this has to be computed in each iteration
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

    return non_zero_rows, non_zero_cols, non_zero_data, rhs_vec


def CreateSolver(dofs_micro, dofs_macro):
    petsc_vec = PETSc.Vec()
    petsc_vec.create(PETSc.COMM_SELF)
    petsc_vec.setSizes(dofs_macro * (1 + dofs_micro))
    petsc_vec.setUp()

    u_sol_petsc = PETSc.Vec()
    u_sol_petsc.create(PETSc.COMM_SELF)
    u_sol_petsc.setSizes(dofs_macro * (1 + dofs_micro))
    u_sol_petsc.setUp()  

    solver = PETSc.KSP().create(PETSc.COMM_SELF)
    solver.setType(PETSc.KSP.Type.GMRES) #PREONLY
    #solver.getPC().setType(PETSc.PC.Type.LU)

    return petsc_vec, u_sol_petsc, solver


def SolveSystem(petsc_vec, u_sol_petsc, solver, global_rhs_vec, M_macro, start_time):
    ### Building is easier with scipy.
    ### But solving is faster with PETSc. (Transformation is fast)
    petsc_mat = PETSc.Mat().createAIJ(size=M_macro.shape, 
                csr=(M_macro.indptr, M_macro.indices, M_macro.data), 
                comm=PETSc.COMM_SELF)

    print("building matrix took", time.time() - start_time)

    solver.setOperators(petsc_mat)
    petsc_vec.array[:] = global_rhs_vec

    print("Start solving")
    start_time = time.time()
    solver.solve(petsc_vec, u_sol_petsc)
    print("Solving is done, took", time.time()- start_time)
    return u_sol_petsc