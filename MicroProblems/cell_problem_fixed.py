from dolfin import *
import warnings

class TransformConvection(UserExpression):
    """
    Circle growth direction. Since the center is at (0, 0) the movement is 
    equal to the position.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval(self, values, x):
        values[0] = x[0] 
        values[1] = x[1] 

    def value_shape(self):
        return (2,)


class CellProblemFixed():

    def __init__(self, initial_mesh, kappa, f, v, 
                 ref_temp, start_temp, time_step, initial_radius, rho_micro, V_micro=None):
        """
        A class to represent one cell problem at a macroscopic position. Here,
        the transformation to a fixed domain is applied.

        Parameters
        ==========
        initial_mesh : Mesh
            The cell mesh. 
        kappa : float
            The heat conductivity inside the cells.
        f : function, float
            The heat source.
        v : float
            A scalar to scale the speed of the boundary movement.
        ref_temp : float
            The reference temperature for growing. 
        start_temp : float 
            The initial temperature.
        time_step : float 
            The size of the current time step.
        initial_radius : float
            The radius of domain at the start.
        rhi_micro : float
            The denisity of the
        V_micro : FunctionSpace, optional
            The function space that should be used on the cell domain.
        """
        self.cell_mesh = initial_mesh #Mesh(initial_mesh)
        self.kappa = Constant(kappa)
        self.f = f
        self.current_radius = initial_radius
        self.initial_radius = initial_radius
        self.time_step = time_step
        self.speed = v
        self.ref_temp = ref_temp
        self.growth_factor = 0
        self.rho_micro = rho_micro

        if V_micro:
            self.V_micro = V_micro
        else:
            self.V_micro = FunctionSpace(self.cell_mesh, "CG", 1)
        self.theta_old = Function(self.V_micro)
        self.theta_old.assign(Constant(start_temp))

        self.dx_micro = Measure("dx", self.V_micro) 
        self.ds_micro = Measure("ds", self.V_micro) 
        self.n_micro = FacetNormal(self.cell_mesh)

        V_micro_vec = VectorFunctionSpace(self.cell_mesh, "CG", 1)
        identity_fn = interpolate(TransformConvection(), V_micro_vec)

        ### Micro problem
        phi_micro = TestFunction(self.V_micro)
        u_micro = TrialFunction(self.V_micro)

        ## Tranform to reference circle:
        self.density_scale = Constant((self.current_radius/self.initial_radius)**2)
        self.old_density_scale = Constant((self.current_radius/self.initial_radius)**2)
        self.convection_scale = Constant(self.growth_factor * self.current_radius/(self.initial_radius**2))
        
        self.a_micro = self.rho_micro * self.density_scale * inner(u_micro, phi_micro) * 1/self.time_step
        self.a_micro += inner(grad(u_micro), identity_fn) * phi_micro * self.convection_scale
        self.a_micro += 0.5 * inner(self.kappa * grad(u_micro), grad(phi_micro)) 
        self.a_micro *= self.dx_micro
        
        self.f_micro = self.density_scale * inner(self.f, phi_micro) 
        self.f_micro += self.rho_micro * self.old_density_scale/self.time_step * inner(self.theta_old, phi_micro) 
        self.f_micro -= 0.5 * inner(self.kappa * grad(self.theta_old), grad(phi_micro))
        self.f_micro *= self.dx_micro

        # A dummy BC to set entries in the systm matrix to 1 and the rhs to 0.
        # Additionally, build a vector that contains 1 and the boundary dofs for the macro
        # coupling.
        self.bc_micro_zero = DirichletBC(self.V_micro, Constant(0.0), "on_boundary")
        self.bc_micro_lhs = DirichletBC(self.V_micro, Constant(-1.0), "on_boundary")

        # For heat transfer to macro domain, we need the flow at the boundary:
        flow_scale = Constant(self.kappa)
        self.micro_transfer = inner(flow_scale * grad(phi_micro), self.n_micro) * self.ds_micro
        self.heat_exchange = assemble(self.micro_transfer).get_local()

        # For the Dirichlet coupling with the macro domain
        self.BC_helper = PETScVector(MPI.comm_self)
        assemble(Constant(0.0)*phi_micro*self.dx_micro, self.BC_helper)
        self.bc_micro_lhs.apply(self.BC_helper)

        self.first_step = True


    def assemble_matrix_block(self):
        A_micro = PETScMatrix(MPI.comm_self)
        self.F_micro = PETScVector(MPI.comm_self)

        assemble_system(self.a_micro, self.f_micro, 
                        A_tensor=A_micro, b_tensor=self.F_micro)

        self.bc_micro_lhs.apply(A_micro)
        self.bc_micro_zero.apply(self.F_micro)

        #bi, bj, bv = A_micro.mat().getValuesCSR()
        self.M_micro = A_micro #csr_matrix((bv, bj, bi))

        self.first_step = False
    

    def update_cell(self, macro_temp, micro_temp):
        self.old_density_scale.assign((self.current_radius/self.initial_radius)**2)
        
        self.theta_old.vector().set_local(micro_temp)
        self.current_energy = self.rho_micro * assemble(self.old_density_scale*self.theta_old * self.dx_micro)
        
        # Radius is explicity given by current macro temperature
        self.growth_factor = self.speed * (macro_temp - self.ref_temp)
        self.current_radius += self.time_step * self.growth_factor
        
        if self.current_radius <= 0 or self.current_radius >= 0.5:
            warnings.warn(f"Radius is {self.current_radius}, system is not well defined!")

        # print(self.current_radius / self.initial_radius)
        # update all scales:
        self.density_scale.assign(
            (self.current_radius/self.initial_radius)**2)
        self.convection_scale.assign(
            self.growth_factor * self.current_radius/(self.initial_radius**2))
        return
    


# class CellProblemFixedImplicit():

#     def __init__(self, initial_mesh, kappa, f, v, 
#                  ref_temp, start_temp, time_step, initial_radius, rho_micro):
#         """
#         initial_mesh : The cell mesh at the start
#         kappa : heat conductivity
#         f : heat source
#         v : speed of movement
#         ref_temp : Reference temperature for growing 
#         start_temp : Initial temperature
#         time_step : size of the current time step
#         initial_radius : radius at the start
#         """
#         self.cell_mesh = Mesh(initial_mesh)
#         self.kappa = Constant(kappa)
#         self.f = f
#         self.current_radius = initial_radius
#         self.initial_radius = initial_radius
#         self.iteration_radius = initial_radius
#         self.iteration_growth_factor = 0
#         self.time_step = time_step
#         self.speed = v
#         self.ref_temp = ref_temp
#         self.growth_factor = 0
#         self.rho_micro = rho_micro

#         self.V_micro = FunctionSpace(self.cell_mesh, "CG", 1)
#         self.theta_old = Function(self.V_micro)
#         self.theta_iterate = Function(self.V_micro)
#         self.theta_iterate_old = Function(self.V_micro)
#         self.theta_old.assign(Constant(start_temp))
#         self.theta_iterate.assign(Constant(start_temp))
#         self.theta_iterate_old.assign(Constant(start_temp))


#         self.dx_micro = Measure("dx", self.V_micro) 
#         self.ds_micro = Measure("ds", self.V_micro) 
#         self.n_micro = FacetNormal(self.cell_mesh)

#         V_micro_vec = VectorFunctionSpace(self.cell_mesh, "CG", 1)
#         identity_fn = interpolate(TransformConvection(), V_micro_vec)
#         ### Micro problem
#         phi_micro = TestFunction(self.V_micro)
#         u_micro = TrialFunction(self.V_micro)

#         self.density_scale = Constant((self.current_radius/self.initial_radius)**2)
#         self.old_density_scale = Constant((self.current_radius/self.initial_radius)**2)
#         self.convection_scale = Constant(self.growth_factor * self.current_radius/(self.initial_radius**2))
        
#         self.a_micro = self.rho_micro * self.density_scale * inner(u_micro, phi_micro) * 1/self.time_step
#         self.a_micro += inner(grad(u_micro), identity_fn) * phi_micro * self.convection_scale
#         self.a_micro += 0.5 * inner(self.kappa * grad(u_micro), grad(phi_micro)) 
#         self.a_micro *= self.dx_micro
        
#         self.f_micro = self.density_scale * inner(self.f, phi_micro) 
#         self.f_micro += self.rho_micro * self.old_density_scale/self.time_step \
#                              * inner(self.theta_old, phi_micro) 
#         self.f_micro -= 0.5 * inner(self.kappa * grad(self.theta_old), grad(phi_micro))
#         self.f_micro *= self.dx_micro

#         # Only dummy bc to set the entries in the matrix to 1 and the rhs to 0
#         # and build a vector that contains 1 and the boundary dofs for the macro
#         # coupling
#         self.bc_micro_zero = DirichletBC(self.V_micro, Constant(0.0), "on_boundary")
#         self.bc_micro_lhs = DirichletBC(self.V_micro, Constant(-1.0), "on_boundary")

#         # For heat transfer to macro domain
#         flow_scale = Constant(self.kappa)
#         self.micro_transfer = inner(flow_scale * grad(phi_micro), 
#                                     self.n_micro) * self.ds_micro

#         self.heat_exchange = assemble(self.micro_transfer).get_local()

#         # For the Dirichlet coupling with the macro domain
#         self.BC_helper = PETScVector(MPI.comm_self)

#         assemble(Constant(0.0)*phi_micro*self.dx_micro, self.BC_helper)

#         self.bc_micro_lhs.apply(self.BC_helper)

#         self.first_step = True


#     def assemble_matrix_block(self):
#         A_micro = PETScMatrix(MPI.comm_self)
#         self.F_micro = PETScVector(MPI.comm_self)

#         assemble_system(self.a_micro, self.f_micro, 
#                         A_tensor=A_micro, b_tensor=self.F_micro)

#         self.bc_micro_lhs.apply(A_micro)
#         self.bc_micro_zero.apply(self.F_micro)

#         #bi, bj, bv = A_micro.mat().getValuesCSR()
#         self.M_micro = A_micro #csr_matrix((bv, bj, bi))

#         self.first_step = False
    
#     def update_iteration(self, macro_temp, micro_temp):
#         self.theta_iterate_old.vector().set_local(
#             self.theta_iterate.vector().get_local()
#         )
#         self.theta_iterate.vector().set_local(micro_temp)

#         self.iteration_growth_factor = self.speed * (macro_temp - self.ref_temp)
#         self.iteration_radius = self.current_radius + self.time_step * self.growth_factor
        
#         self.density_scale.assign(
#             (self.iteration_radius/self.initial_radius)**2)
#         self.convection_scale.assign(
#             self.iteration_growth_factor * self.iteration_radius/(self.initial_radius**2))

#         self.current_energy = self.rho_micro \
#             *assemble(self.density_scale*self.theta_iterate * self.dx_micro)

#         # compute error 
#         error = assemble(inner(self.theta_iterate-self.theta_iterate_old, 
#                                self.theta_iterate-self.theta_iterate_old)
#                                * self.dx_micro)
#         return error

#     def update_cell(self):
#         self.growth_factor = self.iteration_growth_factor
#         self.current_radius = self.iteration_radius

#         self.old_density_scale.assign(
#             (self.current_radius/self.initial_radius)**2)
        
#         self.theta_old.vector().set_local(
#                             self.theta_iterate.vector().get_local())
    
#         return