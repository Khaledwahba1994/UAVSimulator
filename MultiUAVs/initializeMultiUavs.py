import numpy as np
from rowan import from_matrix, to_matrix 
from utils import Rx, Ry, Rz


dt = 0.01 #time step
g = 9.81 #gravitational constant [m/s^2]
t0 = 0
tf = 8

# Initialize the time t vector given:
# Initial time: t0
# Final time: tf
# time step: dt 
samples = int((tf-t0)/dt)
t = np.linspace(t0, tf+dt, num=samples)
# Initialize the Pose
initPos = np.array([[1,1,1]])
initR = np.eye(3)
# initialize Rotation matrix about Roll-Pitch-Yaw
angle = [175,0,0]
initR   = Rz((np.pi/180)*angle[2]) @ Ry((np.pi/180)*angle[1]) @ Rx((np.pi/180)*angle[0])
initq = from_matrix(initR)
#Initialize Twist
initTwist = np.zeros((6,))
### State = [x, y, z, xdot, ydot, zdot, q1, q2, q3, q4, wx, wy, wz] ###
initState = np.zeros((13,))
initState[0:3]  = initPos  # position: x,y,z
initState[3:6]  = initTwist[0:3]  # linear velocity: xdot, ydot, zdot
initState[6:10] = initq# quaternions: [q1, q2, q3, q4]
initState[10::] = initTwist[3::] # angular velocity: wx, wy, wz

initState0 = initState.copy()
initState1 = initState.copy() 
initState2 = initState.copy()
initState3 = initState.copy()

initState1[0:2] = np.array([-1,1])
initState2[0:2] = np.array([-1,-1])
initState3[0:2] = np.array([1,-1])

teamNum = 4
initStates   = {'0':initState0, '1':initState1, '2':initState2, '3':initState3}