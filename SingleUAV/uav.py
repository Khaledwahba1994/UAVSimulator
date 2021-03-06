import numpy as np
from rowan.calculus import integrate as quat_integrate
from rowan import from_matrix, to_matrix, to_euler
from scipy import  integrate, linalg
from utils import skew



class UavModel:
        """initialize an instance of UAV object with the following physical parameters:
        m = 0.028 [kg]  -------------------------------------> Mass of the UAV
        I =   (16.571710 0.830806 0.718277
               0.830806 16.655602 1.800197    -----------------> Moment of Inertia 
               0.718277 1.800197 29.261652)*10^-6 [kg.m^2]"""

        def __init__(self,dt ,state):
            self.m      = 28e-3  
            self.I      = np.array([[16.571710, 0, 0],[0, 16.571710, 0],[0, 0, 29.261652]])* 1e-6
            self.invI   = linalg.inv(self.I)
            self.d      = 4e-3 
            self.cft    = 0.005964552 
            self.all    = np.array([[1, 1, 1, 1],[0, -self.d, 0 , self.d],[self.d, 0 , -self.d, 0],[-self.cft, self.cft, -self.cft, self.cft]])
            self.invAll = linalg.pinv(self.all)
            self.grav  = np.array([0,0,-self.m*9.81])
             ### State initialized with the Initial values ###
             ### state = [x, y, z, xdot, ydot, zdot, q1, q2, q3, q4, wx, wy, wz]
            self.state = state
            self.dt    = dt
           
        def __str__(self):
          return "\nUAV object with physical parameters defined as follows: \n \n m = {} kg,\n \n{} {}\n I = {}{} [kg.m^2] \n {}{}\n\n Initial State = {}".format(self.m,'     ',self.I[0,:],' ',self.I[1,:],'     ',self.I[2,:], self.state)
          
        def getNextAngularState(self, curr_w, curr_q, tau):
           wdot  = self.invI @ (tau - skew(curr_w) @ self.I @ curr_w)
           wNext = wdot * self.dt + curr_w
           qNext = quat_integrate(curr_q, curr_w, self.dt)
           return qNext, wNext

        def getNextLinearState(self, curr_vel, curr_position, q ,fz):
            R_IB = to_matrix(q)
            a =  (1/self.m) * (self.grav + R_IB @ np.array([0,0,fz]))
            velNext = a * self.dt + curr_vel
            posNext = curr_vel * self.dt + curr_position
            return posNext, velNext

        def states_evolution(self, f_th):
            """this method generates the 6D states evolution for the UAV given for each time step:
                the control input: f_th = [f1, f2, f3, f4] for the current step"""
            tau_inp = self.all @ f_th
            fz      = tau_inp[0]
            tau_i   = tau_inp[1::]

            curr_pos  = self.state[0:3]  # position: x,y,z
            curr_vel  = self.state[3:6]  # linear velocity: xdot, ydot, zdot
            curr_q    = self.state[6:10] # quaternions: [q1, q2, q3, q4]
            curr_w    = self.state[10::]  # angular velocity: wx, wy, wz
            
            posNext, velNext = self.getNextLinearState(curr_vel, curr_pos, curr_q, fz)
            qNext, wNext     = self.getNextAngularState(curr_w, curr_q, tau_i)
            
            self.state[0:3]  = posNext  # position: x,y,z
            self.state[3:6]  = velNext  # linear velocity: xdot, ydot, zdot
            self.state[6:10] = qNext# quaternions: [q1, q2, q3, q4]
            self.state[10::] = wNext # angular velocity: wx, wy, wz
            return self.state
           



       
       