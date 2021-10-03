import numpy as np
import numpy.polynomial as poly
import cvxpy as cp
from scipy import linalg as la
import matplotlib.pyplot as plt
polynomial = poly.Polynomial

# fig = plt.figure(figsize=(10,10))
# ax  = fig.add_subplot(autoscale_on=True,projection="3d")
# ax.plot(data[1,:],data[2,:],data[3,:])
# plt.show()

# Problem data.
# WayPoints for cirle trajetory
numRobots = 1
r = 0.3
height = 0.7
w = 2 * np.pi / numRobots
T = 2 * 2 * np.pi / w
time = np.linspace(0,T,100)

data = np.empty((4,time.size))
data[0,:]  = time
data[1,:]  = r * np.cos(w * time)
data[2,:]  = r * np.sin(w * time)
data[3,:]  = height


pieces = 4
n_waypoints = pieces + 1 #number of waypoints
hk = time[-1]/pieces #time per piece [0,hk]

## Construct the Hessian Matrix Q
Qx = np.eye(8*pieces)
Qy = np.eye(8*pieces)
Qz = np.eye(8*pieces)

for i in range(0,8*pieces,8):
    sqrtQx = np.array([0, 0, 0, 0, 24, 120*hk, 360*hk**2, 840**hk**3]).reshape((1,8))
    Qx[i:i+8,i:i+8] = sqrtQx.T @ sqrtQx
    sqrtQy = np.array([0, 0, 0, 0, 24, 120*hk, 360*hk**2, 840**hk**3]).reshape((1,8))
    Qy[i:i+8,i:i+8] = sqrtQy.T @ sqrtQy
    sqrtQz = np.array([0, 0, 0, 0, 24, 120*hk, 360*hk**2, 840**hk**3]).reshape((1,8))
    Qz[i:i+8,i:i+8] = sqrtQz.T @ sqrtQz

## Extract equidistant way points from the given data:
pk   = np.zeros((3,n_waypoints))
vk   = np.zeros((3,n_waypoints))
ak   = np.zeros((3,n_waypoints))
step = round((len(data.T))/pieces)

for i in range(0,pieces):
    index  = step*(i)
    pk[:,i] = data[1:,index]
pk[:,-1] = data[1:,-1]

## If mid conditions are not given:
## Enforce Continuity By finding the mid conditions for vel_k, acc_k, jerk_k
## through equating coefficients c4,k+1 = [.].T*[c4k,..,c7k] from the span condition

h1_vec = np.array([1, 5*hk, 15*hk**2, 35*hk**3]).reshape((1,4))
Hk = np.array([[hk**4    , hk**5    , hk**6      , hk**7],
               [4*hk**3  , 5*hk**4  ,   6*hk**5  , 7*hk**6],
               [12*hk**2 , 20*hk**3 , 30*hk**4   , 42*hk**5],
               [24*hk    , 60*hk**2 , 120*hk**3  , 210*hk**4]]).reshape((4,4))

hk_tri = np.array([[1, hk, 0.5*hk**2],
                   [0, 1,  hk],
                   [0, 0, 1]]).reshape((3,3))

invHk = la.inv(Hk)
invHk = invHk[1:,1:].reshape(3,3)
h1_v = h1_vec[0,1:].reshape((1,3))

A_v = np.zeros((pieces-1,3*(pieces-1)))
b_v = np.zeros((pieces-1,1))
print(A_v.shape)

for i in range(0,(3*(pieces-1))-1,3):
    z = i+3
    A_vdiag =((h1_v @ invHk) + (invHk[0,:] @ hk_tri)).reshape(1,3)
    np.copyto(A_v[i:z,i:z], A_vdiag)
    print(i,z)
    # A_v[i:z,i:z] = A_vdiag.copy()
    print(A_v[i:z,i:z])

print(A_v[:,:])
## Construct the Equality Constraints
Ax_eq = np.zeros((8*pieces,8*pieces))
Ay_eq = np.zeros((8*pieces,8*pieces))
Az_eq = np.zeros((8*pieces,8*pieces))

# # Construct the problem.
# n = 8 * pieces
# x = cp.Variable(n)
# objective = cp.Minimize(cp.sum_squares(A @ x - b))
# constraints = [0 <= x, x <= 1]
# prob = cp.Problem(objective, constraints)

# # The optimal objective is returned by prob.solve().
# result = prob.solve()
# # The optimal value for x is stored in x.value.
# print(x.value)
# # The optimal Lagrange multiplier for a constraint
# # is stored in constraint.dual_value.
# print(constraints[0].dual_value)









