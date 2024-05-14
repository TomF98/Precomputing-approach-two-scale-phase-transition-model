from dolfin import *
import numpy as np
from scipy.sparse import csr_matrix

class Displacement(UserExpression):

    def __init__(self, speed, ref_temp, time_step, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.speed = speed
        self.ref_temp = ref_temp
        self.current_temp = ref_temp
        self.tau = time_step
        self.current_radius = 1.0

    def eval(self, values, x):
        # x_len = np.sqrt(x[0]**2 + x[1]**2)
        # if x_len <= 0.0001:
        #     values[0] = 0.0
        #     values[1] = 0.0
        factor = self.tau * self.speed/self.current_radius * (self.current_temp - self.ref_temp)
        values[0] = x[0] * factor
        values[1] = x[1] * factor

    def compute_current_growth(self):
        speed = self.tau * self.speed * (self.current_temp - self.ref_temp)
        if abs(speed) < 1.e-5:
            return 0
        return speed

    def value_shape(self):
        return (2,)


class CellProblem():

    def __init__(self, initial_mesh, kappa, f, v, 
                 ref_temp, start_temp, time_step, initial_radius, 
                 mass_scaling=1.0):
        """
        initial_mesh : The cell mesh at the start
        kappa : heat conductivity
        f : heat source
        v : speed of movement
        ref_temp : Reference temperature for growing 
        start_temp : Initial temperature
        time_step : size of the current time step
        initial_radius : radius at the start
        """
        self.cell_mesh = Mesh(initial_mesh)
        self.kappa = Constant(kappa)
        self.f = f
        self.current_radius = initial_radius
        self.time_step = time_step
        self.displacement_field = Displacement(v, ref_temp, time_step)

        self.V_micro = FunctionSpace(self.cell_mesh, "CG", 1)
        self.theta_old = Function(self.V_micro)
        self.theta_old.assign(Constant(start_temp))

        self.dx_micro = Measure("dx", self.V_micro) 
        self.ds_micro = Measure("ds", self.V_micro) 
        self.n_micro = FacetNormal(self.cell_mesh)

        ### Micro problem
        phi_micro = TestFunction(self.V_micro)
        u_micro = TrialFunction(self.V_micro)

        density_scale = Constant(mass_scaling/self.time_step)

        self.a_micro = (density_scale * inner(u_micro, phi_micro) +
             inner(self.kappa * grad(u_micro), grad(phi_micro))) * self.dx_micro
        
        self.f_micro = (inner(self.f, phi_micro) 
                        + density_scale * inner(self.theta_old, phi_micro)) * self.dx_micro

        # Only dummy bc to set the entries in the matrix to 1 and the rhs to 0
        # and build a vector that contains 1 and the boundary dofs for the macro
        # coupling
        self.bc_micro_zero = DirichletBC(self.V_micro, Constant(0.0), "on_boundary")
        self.bc_micro_lhs = DirichletBC(self.V_micro, Constant(-1.0), "on_boundary")

        # For heat transfer to macro domain
        flow_scale = Constant(self.kappa/mass_scaling)
        self.micro_transfer = inner(flow_scale * grad(phi_micro), 
                                    self.n_micro) * self.ds_micro
        
        # For the Dirichlet coupling with the macro domain
        self.BC_helper = PETScVector()
        assemble(Constant(0.0)*phi_micro*self.dx_micro, self.BC_helper)
        self.bc_micro_lhs.apply(self.BC_helper)
        self.first_step = True


    def assemble_matrix_block(self):
        # only compute everything again if the cell had any change
        # and also at the first step
        # if (self.displacement_field.compute_current_growth() < 0.00001
        #     and not self.first_step):
        #     return
        self.old_energy = assemble(self.theta_old * self.dx_micro)


        A_micro = PETScMatrix()
        self.F_micro = PETScVector()

        assemble_system(self.a_micro, self.f_micro, 
                        A_tensor=A_micro, b_tensor=self.F_micro)

        self.bc_micro_lhs.apply(A_micro)
        self.bc_micro_zero.apply(self.F_micro)

        bi, bj, bv = A_micro.mat().getValuesCSR()
        self.M_micro = csr_matrix((bv, bj, bi))

        self.heat_exchange = assemble(self.micro_transfer).get_local()

        self.first_step = False
    

    def update_cell(self, macro_temp):
        old_energy = assemble(self.theta_old * self.dx_micro)
        #self.theta_old.vector().set_local(solution_data)

        self.displacement_field.current_radius = self.current_radius
        self.displacement_field.current_temp = macro_temp
        new_growth = self.displacement_field.compute_current_growth()
        if new_growth > 0:
            self.current_radius += new_growth

            ## Move mesh
            ALE.move(self.cell_mesh, self.displacement_field)

            ## Set energy to the same value
            new_energy = assemble(self.theta_old * self.dx_micro)
            self.theta_old.vector()[:] *= old_energy / new_energy
            
        #######################################################
        ### Or just extend/cut function linearly
        # # Before ALE
        # self.cell_mesh.bounding_box_tree().build(self.cell_mesh)
        # old_mesh = Mesh(self.cell_mesh)
        # old_mesh.bounding_box_tree().build(old_mesh)
        # V_old = FunctionSpace(old_mesh, "CG", 1)
        # w_old = Function(V_old)
        # w_old = interpolate(self.theta_old, V_old)
        # w_old.set_allow_extrapolation(True)
        # # After ALE
        # self.theta_old.assign(interpolate(w_old, self.V_micro))

        return
    

class CellProblemParallel():

    def __init__(self, initial_mesh, kappa, f, v, 
                 ref_temp, start_temp, time_step, initial_radius, 
                 rho_micro, mass_scaling=1.0):
        """
        initial_mesh : The cell mesh at the start
        kappa : heat conductivity
        f : heat source
        v : speed of movement
        ref_temp : Reference temperature for growing 
        start_temp : Initial temperature
        time_step : size of the current time step
        initial_radius : radius at the start
        rho_micro : Density of cell
        """
        self.cell_mesh = Mesh(initial_mesh)
        self.kappa = Constant(kappa)
        self.f = f
        self.current_radius = initial_radius
        self.time_step = time_step
        self.displacement_field = Displacement(v, ref_temp, time_step)

        self.V_micro = FunctionSpace(self.cell_mesh, "CG", 1)
        self.theta_old = Function(self.V_micro)
        self.theta_old.assign(Constant(start_temp))

        self.dx_micro = Measure("dx", self.V_micro) 
        self.ds_micro = Measure("ds", self.V_micro) 
        self.n_micro = FacetNormal(self.cell_mesh)

        ### Micro problem
        phi_micro = TestFunction(self.V_micro)
        u_micro = TrialFunction(self.V_micro)

        density_scale = Constant(rho_micro * mass_scaling/self.time_step)

        self.a_micro = (density_scale * inner(u_micro, phi_micro) +
             inner(self.kappa * grad(u_micro), grad(phi_micro))) * self.dx_micro
        
        self.f_micro = (inner(self.f, phi_micro) 
                        + density_scale * inner(self.theta_old, phi_micro)) * self.dx_micro

        # Only dummy bc to set the entries in the matrix to 1 and the rhs to 0
        # and build a vector that contains 1 and the boundary dofs for the macro
        # coupling
        self.bc_micro_zero = DirichletBC(self.V_micro, Constant(0.0), "on_boundary")
        self.bc_micro_lhs = DirichletBC(self.V_micro, Constant(-1.0), "on_boundary")

        # For heat transfer to macro domain
        flow_scale = Constant(self.kappa/mass_scaling)
        self.micro_transfer = inner(flow_scale * grad(phi_micro), 
                                    self.n_micro) * self.ds_micro
        
        # For the Dirichlet coupling with the macro domain
        self.BC_helper = PETScVector(MPI.comm_self)
        assemble(Constant(0.0)*phi_micro*self.dx_micro, self.BC_helper)
        self.bc_micro_lhs.apply(self.BC_helper)
        self.first_step = True



    def assemble_matrix_block(self):
        self.old_energy = assemble(self.theta_old * self.dx_micro)


        A_micro = PETScMatrix(MPI.comm_self)
        self.F_micro = PETScVector(MPI.comm_self)

        assemble_system(self.a_micro, self.f_micro, 
                        A_tensor=A_micro, b_tensor=self.F_micro)

        self.bc_micro_lhs.apply(A_micro)
        self.bc_micro_zero.apply(self.F_micro)

        #bi, bj, bv = A_micro.mat().getValuesCSR()
        self.M_micro = A_micro #csr_matrix((bv, bj, bi))

        self.heat_exchange = assemble(self.micro_transfer).get_local()

        self.first_step = False
    

    def update_cell(self, macro_temp):
        #self.theta_old.vector().set_local(solution_data)
        old_energy = assemble(self.theta_old * self.dx_micro)
        self.displacement_field.current_radius = self.current_radius
        self.displacement_field.current_temp = macro_temp
        new_growth = self.displacement_field.compute_current_growth()
        if new_growth > 0:
            self.current_radius += new_growth

            ### Move mesh
            ALE.move(self.cell_mesh, self.displacement_field)

            ### Set energy to the same value
            new_energy = assemble(self.theta_old * self.dx_micro)
            self.theta_old.vector()[:] *= old_energy / new_energy
            
        #######################################################
        ### Or just extend/cut function linearly
        # # Before ALE
        # self.cell_mesh.bounding_box_tree().build(self.cell_mesh)
        # old_mesh = Mesh(self.cell_mesh)
        # old_mesh.bounding_box_tree().build(old_mesh)
        # V_old = FunctionSpace(old_mesh, "CG", 1)
        # w_old = Function(V_old)
        # w_old = interpolate(self.theta_old, V_old)
        # w_old.set_allow_extrapolation(True)
        # # After ALE
        # self.theta_old.assign(interpolate(w_old, self.V_micro))

        return