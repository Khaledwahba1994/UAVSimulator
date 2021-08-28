import numpy as np
from rowan import from_matrix, to_matrix, to_euler
from scipy import  integrate, linalg
from utils import skew, veeMap

class Controller:
    def __init__(self, uavModel, kpp=10, kdp=5, kpo=70, kdo=10):
        self.kpp    = kpp
        self.kdp    = kdp
        self.kpo    = kpo
        self.kdo    = kdo
        self.m      = uavModel.m  
        self.I      = uavModel.I
        self.d      = uavModel.d   
        self.cft    = uavModel.cft 
        self.invI   = uavModel.invI
        self.all    = uavModel.all  
        self.invAll = uavModel.invAll
        self.g_comp = -(1/self.m) * uavModel.grav.reshape(3,1)

    def desiredStates(self, desiredFlatOutputs, desiredTwist):
        desired_pos = desiredFlatOutputs[0:3].reshape(3,1)
        desired_yaw = desiredFlatOutputs[3]
        desired_vel = desiredTwist[0:3].reshape(3,1)
        desired_w   = desiredTwist[3::].reshape(3,1)
        return desired_pos, desired_yaw, desired_vel, desired_w

    def currentStates(self, state):
        curr_pos  = state[0:3].reshape(3,1)  # position: x,y,z
        curr_vel  = state[3:6].reshape(3,1)  # lin velocity: xdot, ydot, zdot
        curr_q    = state[6:10].reshape(4,1) # quaternions: [q1, q2, q3, q4]
        curr_w    = state[10::].reshape(3,1) # ang velocity: wx, wy, wz
        return curr_pos, curr_vel, curr_q, curr_w

    def linearErrors(self, curr_pos, desired_pos, curr_vel, desired_vel):
            return curr_pos - desired_pos, curr_vel - desired_vel

    def desiredRotation(self, a_des, desired_yaw):
        Rd_IB      = np.eye(3) 
        norm_a_des = linalg.norm(a_des,ord='fro')
        if norm_a_des > 0: 
             zb_des = a_des / norm_a_des
        else:  
            zb_des = np.array([[0],[0],[1]])      
        # Compute the desired heading direction of the x-axis of the body frame from the desired yaw
        xb_des = np.array([[np.cos(desired_yaw)], [np.sin(desired_yaw)], [0]])  
        # Compute the desired direction of the y-axis of the body frame 
        norm_zbxb  = linalg.norm(skew(zb_des) @ xb_des ,ord='fro')
        if norm_zbxb > 0:
            yb_des =  (skew(zb_des) @ xb_des) / norm_zbxb
        else:
            yb_des = np.array([[0], [1], [0]])
        # Compute the desired rotation matrix Rd_IB
        Rd_IB[:,0]  =  skew(yb_des) @ zb_des.reshape(3,)
        Rd_IB[:,1]  = yb_des.reshape(3,)
        Rd_IB[:,2]  = zb_des.reshape(3,)
        return Rd_IB
        
    def angularErrors(self, Rd_IB, R_IB, curr_w, desired_w):
        # Orientation Error
        er = 0.5 * veeMap( Rd_IB.T @ R_IB - R_IB.T @ Rd_IB)
        # Angular velocity error
        ew = curr_w - R_IB.T @ Rd_IB @ desired_w 
        return er, ew

    def largeAngleController(self, state, desiredFlatOutputs, desiredTwist):
        u_inp      = np.zeros((4,))
         # Extract Current States 
        curr_pos, curr_vel, curr_q, curr_w = self.currentStates(state)
        # Extract Desired States
        desired_pos, desired_yaw, desired_vel, desired_w = self.desiredStates(desiredFlatOutputs, desiredTwist)
        # Position and Velocity error:
        ep, ev = self.linearErrors(curr_pos, desired_pos, curr_vel, desired_vel)
        # Desired accelerations in Inertial Frame
        a_des =  -self.kpp * ep - self.kdp * ev + self.g_comp
        # Desired collective thrust in the z-axis of the body frame
        R_IB  = to_matrix(curr_q.reshape(4,))
        fzb   = self.m * a_des.T @ R_IB @ np.array([[0],[0],[1]])   
        # Compute the desired rotation matrix
        Rd_IB = self.desiredRotation(a_des, desired_yaw)        
        q_des =  from_matrix(Rd_IB)
        # Orientation and Angular Velocity Error
        er, ew = self.angularErrors(Rd_IB, R_IB, curr_w, desired_w)
        # The Control Inputs
        Kp = self.kpo * self.I
        Kd = self.kdo * self.I
        tau = (-Kp @ er - Kd @ ew + skew(curr_w) @ self.I @ curr_w)
        u_inp[0] = fzb
        u_inp[1::] = tau.reshape(3,)
        #Compute the thrust for each motor
        f_th = self.invAll @ u_inp 
        return f_th, q_des