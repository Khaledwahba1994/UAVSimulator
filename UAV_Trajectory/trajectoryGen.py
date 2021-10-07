import numpy as np
import numpy.polynomial as poly
import cvxpy as cp
from scipy import linalg as la
import matplotlib.pyplot as plt
polynomial = poly.Polynomial
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

# 

# Problem data.
# WayPoints for cirle trajetory
r = 0.3
height = 0.7
w = 4 * np.pi
T = 0.3 * (2 * np.pi / w)
time = np.linspace(0,T,600)

data = np.empty((4,time.size))
data[0,:]  = time
data[1,:]  = r * np.cos(w * time)
data[2,:]  = r * np.sin(w * time)
data[3,:]  = 1+time/10
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(autoscale_on=True,projection="3d")
ax.plot(data[1,:],data[2,:],data[3,:])


pieces = 10
n_waypoints = pieces + 1 #number of waypoints
hk = time[-1]/pieces #time per piece [0,hk]

## Extract equidistant way points from the given data:
pk   = np.zeros((3,n_waypoints))
vk   = np.zeros((3,n_waypoints))
ak   = np.zeros((3,n_waypoints))
jk   = np.zeros((3,n_waypoints))
step = round((len(data.T))/pieces)

for i in range(0,pieces):
    index  = step*(i)
    pk[:,i] = data[1:,index]

pk[:,-1] = data[1:,-1]


## If mid conditions are not given:
## Enforce Continuity By finding the mid conditions for vel_k, acc_k, jerk_k
## through equating coefficients c4,k+1 = [.].T*[c4k,..,c7k] from the span condition
## Solution for x-y-z axes
for axis in range(0,3):
    h1_vec = np.array([1, 5*hk, 15*hk**2, 35*hk**3]).reshape((1,4))
    Hk = np.array([[hk**4    , hk**5    , hk**6      , hk**7],
                 [4*hk**3  , 5*hk**4  ,   6*hk**5  , 7*hk**6],
                 [12*hk**2 , 20*hk**3 , 30*hk**4   , 42*hk**5],
                 [24*hk    , 60*hk**2 , 120*hk**3  , 210*hk**4]]).reshape(4,4)
    hk_tri = np.array([[0, hk, 0.5*hk**2, (1/6)*hk**3],
                    [0, 1,     hk    ,  0.5*hk**2],
                    [0, 0,     1     ,         hk],
                    [0, 0,     0     ,          1]]).reshape(4,4)
    invHk = la.inv(Hk)
    one_vec = np.array([1,0,0,0]).reshape(1,4)
    vars0 = np.array([0,vk[axis,0],ak[axis,0],jk[axis,0]]).reshape(4,1)
    varsN = np.array([0,vk[axis,-1],ak[axis,-1],jk[axis,-1]]).reshape(4,1)
    A_v = np.zeros((pieces-1,3*(pieces-1)))
    b_v = np.zeros((pieces-1,1))
    b_v[0,0] = (h1_vec @ invHk @ hk_tri @ vars0)
    b_v[-1,0] =  np.array([1,0,0,0]) @ invHk @varsN
    j,k = 0,3 
    for i in range(0,pieces-1):
        b_v[i,0] += one_vec @ invHk @ ((pk[axis,i+2] - pk[axis,i+1])*one_vec.T) \
                  - h1_vec @ invHk @ ((pk[axis,i+1] - pk[axis,i])*one_vec.T) 
        Adiag  = (h1_vec @ invHk) + (one_vec @ invHk @ hk_tri) 
        A_v[i,j:j+3] = Adiag[:,1:]
        if i >= 1 and i < pieces-1:
            As1 = - h1_vec @ invHk @ hk_tri  
            A_v[i,j-3:j] = As1[:,1:]
        j += 3
        if k+3 <= 3*(pieces-1):
            As0 = -one_vec @ invHk
            A_v[i,k:k+3] = As0[:,1:]
            k += 3   
    mid_conditions = la.pinv(A_v) @  b_v
    ind = 1
    for midIndex in range(0,len(mid_conditions),3):
        vk[axis,ind] = mid_conditions[midIndex]
        ak[axis,ind] = mid_conditions[midIndex+1]
        jk[axis,ind] = mid_conditions[midIndex+2]
        ind += 1

# print('vk = ') / print(vk) /print('ak' ) /print(ak) / print('jk ') /print(jk)
## Construct the Hessian Matrix Q
Qx = np.eye(8*pieces)
Qy = np.eye(8*pieces)
Qz = np.eye(8*pieces)
for i in range(0,8*pieces,8):
    sqrtQx = np.array([0, 0, 0, 0, 24, 120*hk, 360*hk**2, 840*hk**3]).reshape((1,8))
    Qx[i:i+8,i:i+8] = sqrtQx.T @ sqrtQx
    sqrtQy = np.array([0, 0, 0, 0, 24, 120*hk, 360*hk**2, 840*hk**3]).reshape((1,8))
    Qy[i:i+8,i:i+8] = sqrtQy.T @ sqrtQy
    sqrtQz = np.array([0, 0, 0, 0, 24, 120*hk, 360*hk**2, 840*hk**3]).reshape((1,8))
    Qz[i:i+8,i:i+8] = sqrtQz.T @ sqrtQz

## Construct the Equality Constraints Aeq matrix and beq for x-y-z axes
Ax_eq = np.zeros((8*pieces,8*pieces))
Ay_eq = np.zeros((8*pieces,8*pieces))
Az_eq = np.zeros((8*pieces,8*pieces))
bx = np.zeros((8*pieces,1)).reshape(8*pieces,1)
by = np.zeros((8*pieces,1)).reshape(8*pieces,1)
bz = np.zeros((8*pieces,1)).reshape(8*pieces,1)

A0 = np.eye(4)
A0[2,2] = 2 
A0[3,3] = 6

for axis in range(0,3):
    for i in range(0,pieces):
        b0 = np.array([[pk[axis,i], vk[axis,i], ak[axis,i], jk[axis,i]]]).reshape(4,1);

        b1  = np.array([[pk[axis,i+1] - pk[axis,i] - vk[axis,i]*hk - 0.5*ak[axis,i]*hk**2  - (1/6)*jk[axis,i]*hk**3],
                            [vk[axis,i+1] - vk[axis,i] - ak[axis,i]*hk - 0.5*jk[axis,i]*hk**2],
                                    [ak[axis,i+1] - ak[axis,i] - jk[axis,i]*hk],
                                        [jk[axis,i+1] - jk[axis,i]]]).reshape(4,1)
        b = np.array([[b0],[b1]]).reshape(8,1)
        if axis == 0:
            Ax_eq[8*i:8*(i+1),8*i:8*(i+1)] = la.block_diag(A0,Hk)
            bx[8*i:8*(i+1)] = b
        elif axis == 1:
            Ay_eq[8*i:8*(i+1),8*i:8*(i+1)] = la.block_diag(A0,Hk)
            by[8*i:8*(i+1)] = b
        else: 
            Az_eq[8*i:8*(i+1),8*i:8*(i+1)] = la.block_diag(A0,Hk)
            bz[8*i:8*(i+1)] = b

# Construct the problem.
coefsx = np.zeros((pieces-1,8))
coefsy = np.zeros((pieces-1,8))
coefsz = np.zeros((pieces-1,8))

j = 0
n = 8 * pieces
for i in range(0,n,8):
    x = cp.Variable((8,1))
    y = cp.Variable((8,1))
    z = cp.Variable((8,1))

    objectivex = cp.quad_form(x, Qx[i:i+8,i:i+8])
    objectivey = cp.quad_form(y, Qy[i:i+8,i:i+8]) 
    objectivez = cp.quad_form(z, Qz[i:i+8,i:i+8])
    constraintsx = [Ax_eq[i:i+8,i:i+8]@x == bx[i:i+8]]
    constraintsy = [Ay_eq[i:i+8,i:i+8]@y == by[i:i+8]]
    constraintsz = [Az_eq[i:i+8,i:i+8]@z == bz[i:i+8]]

    probx = cp.Problem(cp.Minimize(objectivex),constraints=constraintsx)
    proby = cp.Problem(cp.Minimize(objectivey),constraints=constraintsy)
    probz = cp.Problem(cp.Minimize(objectivez),constraints=constraintsz)
    print("probx is DCP:", probx.is_dcp())
    print("proby is DCP:", proby.is_dcp())
    print("probz is DCP:", probz.is_dcp())
    print("Ax_eq@x == bx is DCP:", (constraintsx[0]).is_dcp())
    print("Ay_eq@x == by is DCP:", (constraintsy[0]).is_dcp())
    print("Az_eq@x == bz is DCP:", (constraintsz[0]).is_dcp())
    probx.solve(solver='OSQP',enforce_dpp=True)
    print("\nThe optimal value is", probx.value)
    print("A solution x is")
    print(x.value)
# probx.solve(solver='OSQP',enforce_dpp=True)
# print("\nThe optimal value is", probx.value)
# print("A solution x is")
# print(x.value)

# print(y.value)

# probz.solve()
# plt.show()
# y = cp.Variable(n)
# proby = cp.Problem(cp.Minimize(cp.quad_form(y, Qy)),  [Ay_eq @ y == by])
# print("proby is DCP:", proby.is_dcp())
# print("Ay_eq@y == by is DCP:", (Ay_eq@y == by).is_dcp())
# z = cp.Variable(n)
# probz = cp.Problem(cp.Minimize(cp.quad_form(z, Qz)),  [Az_eq @ z == bz])
# print("probz is DCP:", probz.is_dcp())
# print("Az_eq@z == bz is DCP:", (Az_eq@z == bz).is_dcp())
# for i in range(0,8*pieces,8):
    
#     Qxi = Qx[i:i+8,i:i+8].reshape(8,8)
#     Axi_eq = Ax_eq[i:i+8,i:i+8].reshape(8,8)
#     bxi = bx[i:i+8].reshape(8,1)

#     x = cp.Variable((n,1))
#     probx = cp.Problem(cp.Minimize(cp.quad_form(x, Qxi)),  [Axi_eq @ x == bxi])
#     # probx.solve()
#     # coefsx[j,0:8] = x.value.reshape(n,)
#     print("probx is DCP:", probx.is_dcp())
#     # print("Axi_eq@x == bxi is DCP:", (Axi_eq@x == bxi).is_dcp())

#     Qyi = Qy[i:i+8,i:i+8].reshape(8,8)
#     Ayi_eq = Ay_eq[i:i+8,i:i+8].reshape(8,8)
#     byi = by[i:i+8].reshape(8,1)

#     y = cp.Variable((n,1))
#     proby = cp.Problem(cp.Minimize(cp.quad_form(y, Qyi)), [Ayi_eq @ y == byi])
#     print("proby is DCP:", proby.is_dcp())
#     # print("Ayi_eq@x == byi is DCP:", (Ayi_eq@x == byi).is_dcp())
#     # proby.solve()
#     # coefsy[j,0:8] = y.value.reshape(n,)

#     Qzi = Qz[i:i+8,i:i+8].reshape(8,8)
#     Azi_eq = Az_eq[i:i+8,i:i+8].reshape(8,8)
#     bzi = bz[i:i+8].reshape(8,1)

#     z = cp.Variable((n,1))
#     probz = cp.Problem(cp.Minimize(cp.quad_form(z, Qzi)), [Azi_eq @ z == bzi])
#     print("probz is DCP:", probz.is_dcp())
#     # print("Azi_eq@x == bzi is DCP:", (Azi_eq@x == bzi).is_dcp())
#     # probz.solve()
#     # coefsz[j,0:8] = z.value.reshape(n,)
#     j += 1
# print(coefsx)
# print(coefsy)
# print(coefsz)
# print("probx is DCP:", probx.is_dcp())
# print("Ax_eq@x == bx is DCP:", (Ax_eq@x == bx).is_dcp())

# prob1.solve()
# print("\nThe optimal value is", prob1.value)
# print(x.value)
# y = cp.Variable(n)
# objective = cp.Minimize(cp.quad_form(y, Qy))
# constraints = [Ay_eq@y == by]
# proby = cp.Problem(objective, constraints)

# n =  8 * pieces 
# z = cp.Variable(n)

# objective = cp.Minimize(cp.quad_form(z, Qz))
# constraints = [Az_eq@z == bz]
# probz = cp.Problem(objective, constraints)
# # The optimal objective is returned by prob.solve().
# coefsx = probx.solve()
# coefsy = proby.solve()
# coefsz = probz.solve()

# The optimal value for x is stored in x.value.
# print(x.value)
# print(y.value)
# print(z.value)
# # The optimal Lagrange multiplier for a constraint
# # is stored in constraint.dual_value.
# print(constraints[0].dual_value)









