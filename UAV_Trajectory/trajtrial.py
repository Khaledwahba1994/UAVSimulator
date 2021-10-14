import numpy as np
import cvxpy as cp
from scipy import linalg as la
import numpy.polynomial as poly
import matplotlib.pyplot as plt
import math

polynomial = poly.Polynomial
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

init_cond = np.array([1,0,0,0])
fin_cond  = np.array([1.5,0,0,0])
hk = 2
sqrtQ = np.array([0, 0, 0, 0, 24, 120*hk, 360*hk**2, 840*hk**3]).reshape((8,1))

C = np.zeros((8,8))
for i in range(8):
    if i < 4:
        pass
    else:
        C[i,i] = math.factorial(i)/math.factorial(i-4)

T_mat = np.zeros((8,8), dtype=float)
for i in range(4,8):
    for j in range(4,8):
        T_mat[i,j] = 1.0/((i-4)+(j-4)+1) * (hk)**((i-4)+(j-4)+1)

Q = np.matmul(C, np.matmul(T_mat, C))
print(Q)
A_eq = np.zeros((8,8))

A0 = np.eye(4)
Hk = np.array([[hk**4    , hk**5    , hk**6      , hk**7],
             [4*hk**3  , 5*hk**4  ,   6*hk**5  , 7*hk**6],
             [12*hk**2 , 20*hk**3 , 30*hk**4   , 42*hk**5],
             [24*hk    , 60*hk**2 , 120*hk**3  , 210*hk**4]]).reshape(4,4)

A_eq = la.block_diag(A0,Hk)

A0[2,2] = 2
A0[3,3] = 6
b0 = init_cond
b1  = np.array([fin_cond[0] - init_cond[0] - init_cond[1]*hk - 0.5*init_cond[2]*hk**2  - (1/6)*init_cond[3]*hk**3,
                    fin_cond[1] - init_cond[1] - init_cond[2]*hk - 0.5*init_cond[3]*hk**2,
                            fin_cond[2] - init_cond[2] - init_cond[3]*hk,
                                fin_cond[3] - init_cond[3]])
b = np.array([[b0],[b1]]).reshape(8,)


coefs = cp.Variable(8)
objective = cp.quad_form(coefs,Q)
constraints = [A_eq @ coefs == b]

problem = cp.Problem(cp.Minimize(objective),constraints)
print("prob is DCP:", problem.is_dcp())
print("A_eq@ coefs == b is DCP:", (constraints[0]).is_dcp())
problem.solve(solver='OSQP',eps_dual_inf=1e-6)
print(problem.is_qp())
print("\nThe optimal value is", problem.value)
print("coefs is")
print(coefs.value)
print(type(coefs.value))

pos = polynomial(np.array([coefs.value]).reshape(8,))
vel = pos.deriv(m=1)
acc = pos.deriv(m=2)
jer = pos.deriv(m=2)

time = np.linspace(0,hk,100,endpoint=True)
pos_vec = pos(time)
vel_vec = vel(time)
acc_vec = acc(time)
jer_vec = jer(time)

fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(autoscale_on=True)
ax.plot(time,pos_vec)
ax.plot(time,vel_vec)
ax.plot(time,acc_vec)
plt.show()
