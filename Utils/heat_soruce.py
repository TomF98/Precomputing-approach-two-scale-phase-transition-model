from dolfin import * 
"""
Moving heat source used in the examples.
"""

class HeatSource(UserExpression):
    def __init__(self, t, final_time, heat_value, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = t
        self.t_end = final_time
        self.heat = heat_value

    def eval(self, value, x):
        value[0] = 0.0

        if self.t <= self.t_end:
            left_border = 0.1 + self.t/self.t_end * 0.6
            right_border = 0.3 + self.t/self.t_end * 0.6

            # inner rectanlge
            if x[1] > 0.6 and x[1] < 0.8:
                if x[0] >= left_border and x[0] <= right_border:
                    value[0] = self.heat
                elif x[0] < left_border and x[0] > (left_border - 0.1):
                    value[0] = self.heat * (x[0] - left_border + 0.1) * 10.0
                elif x[0] > right_border and x[0] < (right_border + 0.1):
                    value[0] = self.heat * (right_border + 0.1 - x[0]) * 10.0

            # outer square
            elif x[1] > 0.5 and x[1] < 0.9 and x[0] > (left_border - 0.1) and x[0] < (right_border + 0.1):
                value[0] = self.heat
                
                if x[1] < 0.6:
                    value[0] *= (x[1] - 0.5) * 10.0
                elif x[1] > 0.8:
                    value[0] *= (0.9 - x[1]) * 10.0

                if x[0] < left_border:
                    value[0] *= (x[0] - left_border + 0.1) * 10.0
                elif x[0] > right_border:
                    value[0] *= (right_border + 0.1 - x[0]) * 10.0

            # smooth in time
            if self.t > self.t_end - 1:
                value[0] *= (self.t_end - self.t)


    def value_shape(self):
        return []
    

# macro_mesh_high = UnitSquareMesh(MPI.comm_self, 64, 64)
# V_macro_high = FunctionSpace(macro_mesh_high, "Lagrange", 1)
# dx_macro_high = Measure("dx", V_macro_high)

# dt = 0.1
# t = 0

# f = HeatSource(t, 5, 1)
# fff = File("Results/TestHeatSource/f.pvd")
# while t < 10.0:
#     t += dt
#     f.t = t
#     f_int = interpolate(f, V_macro_high)
#     f_int.rename("f", "f")
#     fff << (f_int, t)
