import numpy as np
import numpy.polynomial as poly
import cvxpy as cp 
polynomial = poly.Polynomial


x = cp.Variable()
y = cp.Variable()

# Create two constraints.
constraints = [x + y == 1,
               x - y >= 1]

# Form objective.
obj = cp.Minimize((x - y)**2)

# Form and solve problem.
prob = cp.Problem(obj, constraints)
prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value, y.value)
# class Trajectory:
#     def __init__(self,cond_pos,cond_vel,cond_acc):
#       pass
        