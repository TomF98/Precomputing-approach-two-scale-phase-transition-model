from dolfin import *
import numpy as np
import mshr
from scipy.sparse import csr_matrix, lil_matrix, bmat
from scipy import sparse
import time 
from petsc4py import PETSc

from cell_problem import CellProblem
from effective_diffusion_fn import diffusion_compute

for c_dt in [0.05]:
    for c_res in [8]: 
        expected_energy = 0
        micro_energy = 0
        ### Problem parameters
        kappa_macro = 0.1
        kappa_micro = 0.1

        theta_ref = 0.0
        theta_start = 0.0
        f_macro_prod = Expression("2.0*x[0]", degree=1)
        f_micro_prod = Constant(0.0)

        ## Effective conductivity
        eff_cond_array = np.load("Data/effective_conductivity.npy")

        ### Time interval
        T_int = [0, 2.0]
        dt = c_dt
        t_n = T_int[0]

        prod_stop = 0.5
        f_stopper = Constant(1.0)
        ### Domain parameters
        macro_size_x = 1
        macro_size_y = 1

        micro_radius = 0.2
        growth_speed = 0.1

        macro_res = c_res
        micro_res = c_res

        ## Genreate mesh
        macro_mesh = RectangleMesh(Point(0, 0), Point(macro_size_x, macro_size_y), 
                                   macro_res, macro_res)

        circle = mshr.Circle(Point(0.0, 0.0), micro_radius)
        micro_mesh = mshr.generate_mesh(circle, micro_res)

        print("dofs micro", len(micro_mesh.coordinates()))
        # micro_mesh = Mesh(MPI.comm_self)
        # with XDMFFile(MPI.comm_self, "MeshCreation/micro_domain.xdmf") as infile:
        #     infile.read(micro_mesh)

        #micro_mesh = RectangleMesh(Point(0, 0), Point(0.5, 0.5), 
        #                   micro_res, micro_res)

        ### Generate macro problem
        V_macro = FunctionSpace(macro_mesh, "Lagrange", 1)
        dof_map = V_macro.dofmap()

        sol_function = Function(V_macro)
        sol_function.assign(Constant(theta_start))

        theta_old = Function(V_macro)
        theta_old.assign(Constant(theta_start))

        energy_flow = Function(V_macro)
        energy_file = File("Results/micro_energy_flow.pvd")

        energy_diff_micro = Function(V_macro)
        energy_diff_micro_file = File("Results/energy_diff_micro_file.pvd")

        #prev_micro_energy = Function(V_macro)

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
                + f_stopper * inner(f_macro_prod, phi_macro)) * dx_macro
        ### Create micro problems
        micro_problem_list = []

        global_mass_matrix = assemble(u_macro * phi_macro * dx_macro)
        #row_sum_mass_matrix = np.sum(global_mass_matrix.array(), axis=1)

        for i in range(dofs_macro):
            micro_problem_list.append(CellProblem(micro_mesh, kappa_micro, f_micro_prod, 
                                                growth_speed, theta_ref, theta_start, dt, 
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
        if (c_dt <= 0.03 and c_res == 16) or (c_dt >= 0.08 and c_res == 4):
            sol_file = File("Results/macro_solution_dt_" + str(dt) + "_Res_" + str(c_res) + ".pvd")
            macro_cond_scale_file = File("Results/macro_cond_scale_dt_" + str(dt) + "_Res_" + str(c_res) + ".pvd")

            radius_file = File("Results/radius_file_dt_" + str(dt) + "_Res_" + str(c_res) + ".pvd")
            radius_fn = Function(V_macro)
            radius_fn.assign(Constant(micro_radius))

            av_cell_temp_file = File("Results/average_cell_temp_dt_" + str(dt) + "_Res_" + str(c_res) + ".pvd")
            av_temp_fn = Function(V_macro)
            av_temp_fn.assign(Constant(theta_start))

            cell_temp_file = File("Results/example_cell_temp_dt_" + str(dt) + "_Res_" + str(c_res) + ".pvd")
        else:
            sol_file = File("Results/macro_solution.pvd")
            macro_cond_scale_file = File("Results/macro_cond_scale.pvd")

            radius_file = File("Results/radius_file.pvd")
            radius_fn = Function(V_macro)
            radius_fn.assign(Constant(micro_radius))

            av_cell_temp_file = File("Results/average_cell_temp.pvd")
            av_temp_fn = Function(V_macro)
            av_temp_fn.assign(Constant(theta_start))

            cell_temp_file = File("Results/example_cell_temp.pvd")          
        save_idx = dofs_macro - 8

        ### Save initial data
        sol_file << (sol_function, t_n)
        radius_file << (radius_fn, t_n)
        av_cell_temp_file << (av_temp_fn, t_n)

        Matrix_list = [[None for _ in range(dofs_macro + 1)] for _ in range(dofs_macro + 1)] 
        rhs_vector_np = np.zeros(dofs_macro * (1 + dofs_micro))

        # compute effective density
        micro_cell_volume = np.pi * radius_fn.vector().get_local()**2
        effective_density_fn.vector().set_local(1.0 - micro_cell_volume)

        energy_array = np.zeros((int(T_int[1]/dt)+1, 4)) # time, expected energy, macro en, micro en
        energy_idx = 0

        while t_n <= T_int[1] - dt/8.0:
            ### Update time variable and cells
            t_n += dt
            theta_old.assign(sol_function)

            if t_n > prod_stop:
                f_stopper.assign(0.0)

            print("Currently working on time step", t_n)

            ### Update effective density and conductivity on macro scale
            ## Conductivity with look up table
            # new_values = np.interp(radius_fn.vector().get_local(), 
            #                     eff_cond_array[:, 0], eff_cond_array[:, 1])
            # effective_cond_fn.vector().set_local(new_values)

            ## Compute eff. density by hand (since micro domain is a circle) and update
            ## the old one.
            old_effective_density_fn.assign(effective_density_fn)
            micro_cell_volume = np.pi * radius_fn.vector().get_local()**2
            effective_density_fn.vector().set_local(1.0 - micro_cell_volume)

            ### First we have to reconstruct the complete matrix.
            ### Start with diagonal blocks
            A_macro = PETScMatrix()
            F_macro = PETScVector()
            assemble_system(a_macro, f_macro, A_tensor=A_macro, b_tensor=F_macro)

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

            ## Build macro micro coupling
            M_lil = lil_matrix(M_complete)

            counter = 0

            print("Start coupling in matrix")
            for i in range(dofs_macro):
                coupling_dofs, mass_values = global_mass_matrix.getrow(i)

                dof_counter = 0
                for dof in coupling_dofs:
                    M_lil[dof, dofs_macro + i * dofs_micro : 
                            dofs_macro + (i+1) * dofs_micro] = \
                            mass_values[dof_counter] * micro_problem_list[i].heat_exchange
                    dof_counter += 1

                M_lil[dofs_macro + i * dofs_micro : 
                      dofs_macro + (i+1) * dofs_micro, 
                      i] = micro_problem_list[i].BC_helper

            # Show matrix        
            # import matplotlib.pylab as plt
            # plt.spy(M_lil)
            # plt.show()

            ### Solve problem
            M_complete = csr_matrix(M_lil)
            # sparse.save_npz("matrix_s.npz", M_complete )
            # exit()
            M_complete.eliminate_zeros()

            
            petsc_mat = PETSc.Mat().createAIJ(size=M_complete.shape, 
                                            csr=(M_complete.indptr, M_complete.indices, M_complete.data))

            print("Building took", time.time() - start_time)
            solver = PETSc.KSP().create()
            solver.setOperators(petsc_mat)
            solver.setType(PETSc.KSP.Type.GMRES) #PREONLY
            #solver.getPC().setType(PETSc.PC.Type.LU)

            petsc_vec.array[:] = rhs_vector_np

            print("Start solving")
            start_time = time.time()
            solver.solve(petsc_vec, u_sol_petsc)
            print("Solving is done, took", time.time()- start_time)

            print("Extract macro solution")
            solution_array = u_sol_petsc.array
            sol_function.vector().set_local(solution_array[:dofs_macro])

            print("Update micro solution")
            start_time = time.time()
            for i in range(dofs_macro):
                s_time = time.time()
                ### First set the current solution and compute average
                #micro_volume = assemble(1 * micro_problem_list[i].dx_micro)
                micro_problem_list[i].theta_old.vector().set_local(
                    solution_array[dofs_macro+i*dofs_micro:dofs_macro+(i+1)*dofs_micro])

                micro_average = assemble(micro_problem_list[i].theta_old * 
                                         micro_problem_list[i].dx_micro)# / micro_volume
                av_temp_fn.vector()[i] = micro_average
                #print("Extract micro sol took", time.time() - s_time)
                ### Then update the cell
                s_time = time.time()
                micro_problem_list[i].update_cell(solution_array[i])
                radius_fn.vector()[i] = micro_problem_list[i].current_radius
                #print("Update cell took", time.time() - s_time)
                ### Save some micro solutions as well
                s_time = time.time()
                if i == save_idx:
                    cell_temp_file << (micro_problem_list[i].theta_old, t_n)
        
                energy_flow.vector()[i] = kappa_micro * assemble(inner(
                    grad(micro_problem_list[i].theta_old), 
                    micro_problem_list[i].n_micro) * micro_problem_list[i].ds_micro)

                energy_diff_micro.vector()[i] = micro_average - micro_problem_list[i].old_energy
                #prev_micro_energy.vector()[i] = micro_problem_list[i].old_energy
                #print("Energy computation took", time.time() - s_time)
                effective_cond_fn.vector()[i] = diffusion_compute(micro_problem_list[i].current_radius)

            print("took", time.time() - start_time)
            ### Save solutuion
            sol_file << (sol_function, t_n)
            radius_file << (radius_fn, t_n)
            av_cell_temp_file << (av_temp_fn, t_n)
            macro_cond_scale_file << (effective_cond_fn, t_n)
            energy_file << (energy_flow, t_n)
            energy_diff_micro_file << (energy_diff_micro, t_n)

            ### Compute energy and compare
            energy_array[energy_idx, 0] = t_n
            expected_energy += dt * assemble(f_stopper * f_macro_prod * dx_macro)
            macro_energy = assemble(effective_density_fn * sol_function * dx_macro)
            micro_energy = assemble(av_temp_fn * dx_macro)
            #micro_energy += dt * assemble(energy_flow * dx_macro)
            print("Expected Energy:", expected_energy)
            num_energy = macro_energy + micro_energy
            print("Numeric Energy:", num_energy)
            print("Rel diff:", np.abs(expected_energy - (num_energy)) / expected_energy)
            energy_array[energy_idx, 1] = expected_energy
            energy_array[energy_idx, 2] = macro_energy
            energy_array[energy_idx, 3] = micro_energy
            energy_idx += 1

        # np.save("Energy/MicroLinearExtensionCorrection/Data_dt_" + str(dt) + "_Res_" + str(macro_res), 
        #         energy_array)