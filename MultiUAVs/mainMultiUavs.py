import os
import sys 
path = os.getcwd()
sys.path.insert(0, path)
sys.path.insert(0, path+'/SingleUAV')
sys.path.insert(0, path+'/ControllerUAV')
sys.path.insert(0, path+'/Utilities')
sys.path.insert(0, path+'/UAV_Trajectory')
import uav
import controller
import numpy as np
from rowan import from_matrix, to_matrix 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
import matplotlib.animation as animation
from UavTrajectory import DesiredTrajInfinity, DesiredTrajHelix, Hover
from initializeMultiUavs import initStates, teamNum, t0, tf, dt, t
from Animate_single import PlotandAnimate
import time
import multiUavs

k = [10,5,70,10]
# Initialize a 'multirobots' object given initial states and number of vehicles
multirobots  = multiUavs.MultiRobots(teamNum, dt, initStates)
refTraj      = {}  # Each loop reference traj
fullstateSt  = {}  # Full states stack
refstateSt   = {}  # Reference states stack

# Initialize both stacks 
for robot in multirobots.robots.keys():
    fullstateSt[robot] = np.zeros((1,13))
    refstateSt[robot] = np.zeros((1,13))

## Initialize Reference Hover Pose:
for robot in multirobots.robots.keys():
    refTraj[robot] = Hover(initStates[robot])

for i in range(0,len(t)):
    for robot in multirobots.robots.keys():
        # refTraj[robot] = DesiredTrajHelix(t[i], initStates[robot][0], initStates[robot][1], initStates[robot][2])
        uavRobot       = multirobots.robots[robot]
        state          = uavRobot.state
        uavcontr       = multirobots.uavcontrollers[robot]
        f_th, qref     = uavcontr.largeAngleController(state, refTraj[robot][0],refTraj[robot][1])
        state          = uavRobot.states_evolution(f_th)
        # Stack the full State
        fullstateSt[robot] = np.concatenate((fullstateSt[robot], state.reshape(1,13))) 
        # Stack the Reference State
        currRef         = np.zeros((1,13))
        currRef[0,0:3]  = refTraj[robot][0][0:3]
        currRef[0,3:6]  = refTraj[robot][1][3:6]
        currRef[0,6:10] =  qref
        refstateSt[robot]  = np.concatenate((refstateSt[robot], currRef.reshape(1,13)))

# Remove the first row of zeros used in initialization step   
for robot in multirobots.robots.keys():
    fullstateSt[robot] = np.delete(fullstateSt[robot], 0, 0)
    refstateSt[robot]  = np.delete(refstateSt[robot], 0, 0)      

## Plot reference and actual states of the whole team
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(autoscale_on=True,projection="3d")
ax.view_init(25,35)
dataDict = {}
sample   = 10
for robot in multirobots.robots.keys():
    uavrobot = multirobots.robots[robot]
    reference_state = refstateSt[robot]
    full_state      = fullstateSt[robot]
    dataDict[robot] = (full_state[::sample], reference_state[::sample])

multirobots.setData(dataDict)

animateAndSave = True
if animateAndSave:
    videoname = path+'/Videos/UpsideDownTeam.gif'
    show = False
    t_sampled  = t[::sample]
    dt_sampled = t_sampled[1] - t_sampled[0]
    print("Converting Animation to Video. \nPlease wait...")
    now = time.time()
    multirobots.startAnimation(fig, ax, videoname, show, dt_sampled)
    end = time.time()
    print("Run time:  {:.3f}s".format((end - now)))