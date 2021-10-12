import numpy as np
import numpy.polynomial as poly
import cvxpy as cp
from scipy import linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
polynomial = poly.Polynomial
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

#

# Problem data.
# WayPoints for cirle trajetory
r = 0.3
height = 0.7
w = 2 * np.pi
T = 4 * (2 * np.pi / w)
time = np.linspace(0,T,600,endpoint=True)

data = np.empty((4,time.size))
data[0,:]  = time
data[1,:]  = r * np.cos(w * time)
data[2,:]  = r * np.sin(w * time)
data[3,:]  = 1+time/10

## Define the number of splines
pieces = 30
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
    
    A_v = np.zeros((pieces-1,3*(pieces-1)))
    b_v = np.zeros((pieces-1,1))

    b_v[0,0]  = h1_vec @ invHk @ hk_tri @ vars0
    j,k = 0,3
    for i in range(0,pieces-1):
        b_v[i,0] +=   (one_vec @ invHk @ ((pk[axis,i+2] - pk[axis,i+1])*one_vec.T)) \
                     - (h1_vec @ invHk @ ((pk[axis,i+1] -   pk[axis,i])*one_vec.T))
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


## Construct the Equality Constraints Aeq matrix and beq for x-y-z axes
Ax_eq = np.zeros((8*pieces,8*pieces))
Ay_eq = np.zeros((8*pieces,8*pieces))
Az_eq = np.zeros((8*pieces,8*pieces))
A0 = np.eye(4)
A0[2,2] = 2
A0[3,3] = 6

bx = np.zeros((8*pieces,))
by = np.zeros((8*pieces,))
bz = np.zeros((8*pieces,))

for axis in range(0,3):
    for i in range(0,pieces):
        b0 = np.array([[pk[axis,i], vk[axis,i], ak[axis,i], jk[axis,i]]]).reshape(4,1)

        b1  = np.array([[pk[axis,i+1] - pk[axis,i] - vk[axis,i]*hk - 0.5*ak[axis,i]*hk**2  - (1/6)*jk[axis,i]*hk**3],
                            [vk[axis,i+1] - vk[axis,i] - ak[axis,i]*hk - 0.5*jk[axis,i]*hk**2],
                                    [ak[axis,i+1] - ak[axis,i] - jk[axis,i]*hk],
                                        [jk[axis,i+1] - jk[axis,i]]]).reshape(4,1)

        b = np.array([[b0],[b1]]).reshape(8,)
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
coefsx = la.inv(Ax_eq) @ bx 
coefsy = la.inv(Ay_eq) @ by
coefsz = la.inv(Az_eq) @ bz

n = 8 * pieces
postraj  = np.zeros((4,step*pieces))
veltraj  = np.zeros((4,step*pieces))
acctraj  = np.zeros((4,step*pieces))
jerktraj = np.zeros((4,step*pieces))

postraj[0,:]  = time[:(step*pieces)]
veltraj[0,:]  = time[:(step*pieces)]
acctraj[0,:]  = time[:(step*pieces)]
jerktraj[0,:] = time[:(step*pieces)]

stepInd = 0
for i in range(0,n,8):
    xpoly = polynomial(np.array([coefsx[i:i+8]]).reshape(8,))
    ypoly = polynomial(np.array([coefsy[i:i+8]]).reshape(8,))
    zpoly = polynomial(np.array([coefsz[i:i+8]]).reshape(8,))
    vxpoly, vypoly, vzpoly = xpoly.deriv(m=1),ypoly.deriv(m=1),zpoly.deriv(m=1)
    axpoly, aypoly, azpoly = xpoly.deriv(m=2),ypoly.deriv(m=2),zpoly.deriv(m=2)
    jxpoly, jypoly, jzpoly = xpoly.deriv(m=3),ypoly.deriv(m=3),zpoly.deriv(m=3)

    time_hk = np.linspace(0, hk,step)

    postraj[1:,stepInd:stepInd+step]  = np.array([xpoly(time_hk),ypoly(time_hk),zpoly(time_hk)]).reshape(3,len(time_hk))
    veltraj[1:,stepInd:stepInd+step]  = np.array([vxpoly(time_hk),vypoly(time_hk),vzpoly(time_hk)]).reshape(3,len(time_hk))
    acctraj[1:,stepInd:stepInd+step]  = np.array([axpoly(time_hk),aypoly(time_hk),azpoly(time_hk)]).reshape(3,len(time_hk))
    jerktraj[1:,stepInd:stepInd+step] = np.array([jxpoly(time_hk),jypoly(time_hk),jzpoly(time_hk)]).reshape(3,len(time_hk))

    stepInd += step

fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(autoscale_on=True,projection="3d")
trajspline = ax.plot(postraj[1,:],postraj[2,:],postraj[3,:],'k',lw=1,label="7th order Spline Trajectory")
refTraj = ax.plot(data[1,:],data[2,:],data[3,:],'r',lw=1,label='Reference Trajectory')
wayp = ax.plot(pk[0,:],pk[1,:],pk[2,:],'*b',lw=2,label='waypoints')
ax.legend()
plt.grid()
plt.show()


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

cffsx = cp.Variable(n)
obj   = cp.Minimize(cp.quad_form(cffsx, Qx))
constraints = [Ax_eq @ cffsx == bx]
problem = cp.Problem(objective=obj,constraints=constraints)
problem.solve()