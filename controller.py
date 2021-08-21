import numpy as np
from rowan import from_matrix, to_matrix, to_euler
from scipy import  integrate, linalg
from utils import skew, veeMap

class Controller:
    def __init__(self,kpp=10,kdp=5, kpo=70, kdo=10):
        self.kpp    = kpp
        self.kdp    = kdp
        self.kpo    = kpo
        self.kdo    = kdo
        self.m      = 28 * 10**(-3)  
        self.I      = np.array([[16.571710, 0, 0],[0, 16.571710, 0],[0, 0, 29.261652]])* 10 ** (-6) 
        self.d      = 4 * 10**(-3) 
        self.cft    = 0.005964552 
        self.invI   = linalg.inv(self.I)
        self.all    = np.array([[1, 1, 1, 1],[0, -self.d, 0 , self.d],[self.d, 0 , -self.d, 0],[-self.cft, self.cft, -self.cft, self.cft]])
        self.invAll = linalg.pinv(self.all)
        self.g_comp = np.array([[0],[0],[9.81]])
    
    def LargeAngleController(self, state,desiredFlatOutputs, desiredTwist):
        tau_inp      = np.zeros((4,))
        Rd_IB        = np.eye(3) 
        # Extract Desired and Current States 
        curr_pos  = state[0:3].reshape(3,1)  # position: x,y,z
        curr_vel  = state[3:6].reshape(3,1)  # lin velocity: xdot, ydot, zdot
        curr_q    = state[6:10].reshape(4,1) # quaternions: [q1, q2, q3, q4]
        curr_w    = state[10::].reshape(3,1)  # ang velocity: wx, wy, wz
        # Extract Desired States
        desired_pos = desiredFlatOutputs[0:3].reshape(3,1)
        desired_yaw = desiredFlatOutputs[3]
        desired_vel = desiredTwist[0:3].reshape(3,1)
        desired_w   = desiredTwist[3::].reshape(3,1)
        # Position and Velocity error:
        ep = curr_pos - desired_pos  
        ev = curr_vel - desired_vel
        # Desired accelerations in Inertial Frame
        a_des =  -self.kpp * ep - self.kdp * ev + self.g_comp
        # Desired collective thrust in the z-axis of the body frame
        R_IB  = to_matrix(curr_q.reshape(4,))
        fzb   = self.m * a_des.T @ R_IB @ np.array([[0],[0],[1]])   
        # Compute the desired direction of the z-axis of the body frame
        norm_a_des = linalg.norm(a_des,ord='fro')
        if norm_a_des > 0:
            zb_des = a_des / norm_a_des
        else:
            zb_des = np.array([[0],[0],[1]])      
        # Compute the desired heading direction of the x-axis of the body frame from the desired yaw
        xb_des = np.array([[np.cos(desired_yaw)], [np.sin(desired_yaw)], [0]])  
        # Compute the desired direction of the y-axis of the body frame
        cr_zbdes_xbdes = skew(zb_des) @ xb_des 
        norm_zbxb      = linalg.norm(cr_zbdes_xbdes,ord='fro')
        if norm_zbxb > 0:
            yb_des =  cr_zbdes_xbdes / norm_zbxb
        else:
            yb_des = np.array([[0], [1], [0]])
        # Compute the desired rotation matrix Rd_IB
        Rd_IB[:,0]  =  skew(yb_des) @ zb_des.reshape(3,)
        Rd_IB[:,1]  = yb_des.reshape(3,)
        Rd_IB[:,2]  = zb_des.reshape(3,)
        q_des       =  from_matrix(Rd_IB)
        # Orientation Error
        Rtd_IB = Rd_IB.T
        Rt_IB  =  R_IB.T
        er = 0.5 * veeMap( Rtd_IB @ R_IB - Rt_IB @ Rd_IB)
        # Angular velocity error
        ew = curr_w - Rt_IB @ Rd_IB @ desired_w 
        # The Control Moments
        Kp = self.kpo * self.I
        Kd = self.kdo * self.I
        Jw = self.I @ curr_w
        tau = (-Kp @ er - Kd @ ew + skew(curr_w) @ Jw)
        #Compute the thrust for each motor
        tau_inp[0] = fzb
        tau_inp[1::] = tau.reshape(3,)
        f_th = self.invAll @ tau_inp 
        return f_th, q_des