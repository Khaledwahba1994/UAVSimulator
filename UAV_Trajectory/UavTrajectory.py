import numpy as np
from pathlib import Path
import os
def _DesiredTrajInfinity(t):
    p                  = 1
    desiredFlatOutputs = np.zeros((4,))
    desiredTwist       = np.zeros((6,))
    wt                      = ((2*np.pi)/p) * t
    desiredFlatOutputs[0] = 2 * np.sin(wt)
    desiredFlatOutputs[1] = np.sin(2*wt)
    desiredFlatOutputs[2] = 0.6 * np.sin(wt) + np.sin(2*wt) 
    desiredFlatOutputs[3] = 0 # np.pi * np.sin(wt)
    return desiredFlatOutputs, desiredTwist

def DesiredTrajInfinity():
    filename = "/UAVSimulator/trajectoriesCSV/infinity8.csv"
    fpath = Path(os.getcwd())
    fpathParent = str(fpath.parent) + filename
    postraj = np.genfromtxt(fpathParent, delimiter=',')
    desiredFlatOutputs = np.zeros((4,len(postraj.T)))
    desiredTwist       = np.zeros((6,))
    desiredFlatOutputs[0:3,:] = postraj[1:]
    desiredFlatOutputs[3] = 0
    return desiredFlatOutputs, desiredTwist

def Hover(p):
    desiredFlatOutputs = np.zeros((4,))
    desiredTwist       = np.zeros((6,))
    desiredFlatOutputs[0] = p[0]
    desiredFlatOutputs[1] = p[1]
    desiredFlatOutputs[2] = p[2]
    desiredFlatOutputs[3] = p[3]
    return desiredFlatOutputs, desiredTwist

