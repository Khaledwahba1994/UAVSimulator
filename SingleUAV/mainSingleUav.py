import os
import sys 
path = os.getcwd()
sys.path.insert(0, path)
sys.path.insert(0, path+'/ControllerUAV/')
sys.path.insert(0, path+'/Utilities/')
sys.path.insert(0, path+'/UAV_Trajectory/')
import uav
import controller
import numpy as np
from rowan import from_matrix, to_matrix 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
import matplotlib.animation as animation
from UavTrajectory import _DesiredTrajInfinity, DesiredTrajInfinity, Hover
from initialize import initState, t0, tf, dt, t
from AnimateSingleUav import PlotandAnimate
import time
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)
# Initialize UAV object with a given initState and a step time dt
uavModel = uav.UavModel(dt, initState)
# Given the controller gains, initialize a controller object
k = [10,5,0.05,0.001]
controller = controller.Controller(uavModel,kpp=k[0],kdp=k[1],kpo=k[2],kdo=k[3])
# Logging the state data
full_traj = DesiredTrajInfinity()
full_state = np.zeros((1,13))
ref_state  = np.zeros((1,13))
traj_choice = 1

for i in range(0,len(t)):
# Generate the reference flat outputs trajectory for 3 different trajectories
    RefTraj  = np.array([full_traj[0][0][i],full_traj[0][1][i],full_traj[0][2][i],full_traj[0][3][i]])
    RefTwist = full_traj[1][:]
    # Start Trajectory Tracking Algorithm    
    f_th, qref = controller.largeAngleController(RefTraj, RefTwist)
    state      = uavModel.states_evolution(f_th)
    #  Reference State at step i
    currRefSt         = np.zeros((1,13))
    currRefSt[0,0:3]  = RefTraj[0:3]
    currRefSt[0,3:6]  = RefTwist[3:6]
    currRefSt[0,6:10] =  qref
    full_state = np.concatenate((full_state, state.reshape(1,13)))
    ref_state  = np.concatenate((ref_state, currRefSt.reshape(1,13)))

full_state = np.delete(full_state, 0, 0)
ref_state  = np.delete(ref_state, 0, 0)
sample     = 10

fig     = plt.figure(figsize=(10,10))
ax      = fig.add_subplot(autoscale_on=True,projection="3d")
animate = PlotandAnimate(fig, ax, uavModel, full_state[::sample,:], ref_state[::sample,:])

animateAndSave = True
if animateAndSave:
    videoname  = path+'/Videos/infinitytraj.mp4' 
    t_sampled  = t[::sample]
    dt_sampled = t_sampled[1] - t_sampled[0]
    show       = True
    save       = False
    if show:
        print("Showing animation.")
    if save:
        print("Converting Animation to Video. \nPlease wait...")
    now = time.time()
    startanimation = animate.startAnimation(videoname,show,save,dt_sampled)
    end = time.time()
    print("Run time:  {:.3f}s".format((end - now)))
