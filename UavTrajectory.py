import numpy as np

def DesiredTrajInfinity(t):
    p                  = 1
    desiredFlatOutputs = np.zeros((4,))
    desiredTwist       = np.zeros((6,))
    wt                      = ((2*np.pi)/p) * t
    desiredFlatOutputs[0] = 2 * np.sin(wt)
    desiredFlatOutputs[1] = np.sin(2*wt)
    desiredFlatOutputs[2] = 0.6 * np.sin(wt) + np.sin(2*wt) 
    desiredFlatOutputs[3] = 0 # np.pi * np.sin(wt)
    return desiredFlatOutputs, desiredTwist

def DesiredTrajHelix(t, xinit, yinit,zinit):
    desiredFlatOutputs = np.zeros((4,))
    desiredTwist       = np.zeros((6,))
    desiredFlatOutputs[0] = 0.1 * np.cos((t)) + (xinit - 0.1)
    desiredFlatOutputs[1] = 0.1 * np.sin((t)) + yinit
    desiredFlatOutputs[2] = zinit#1 + (t/5)
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

