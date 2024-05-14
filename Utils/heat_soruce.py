from dolfin import * 

class HeatSource(UserExpression):
    def __init__(self, t, final_time, heat_value, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = t
        self.t_end = final_time
        self.heat = heat_value

    def eval(self, value, x):
        value[0] = 0.0
        if self.t < self.t_end:
            if x[1] > 0.6 and x[1] < 0.8:
                if x[0] >= 0.1 + self.t/self.t_end * 0.6 and x[0] <= 0.3 + self.t/self.t_end * 0.6:
                    value[0] = self.heat