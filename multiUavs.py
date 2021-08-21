import uav
import controller
import numpy as np
from utils import Rz, RotatedCylinder
from rowan import from_matrix, to_matrix 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
import matplotlib.animation as animation


class MultiRobots:
    def __init__(self,teamNum,dt,initStates):
        self.robots = {}
        self.uavcontrollers = {}
        self.dataDict = {}

        for i in range(0,teamNum):
            name  = str(i)
            state = initStates[name] 
            self.robots[name] = uav.UavModel(dt, state)
            self.uavcontrollers[name] = controller.Controller()
        
    def setData(self,dataDict):
        self.dataDict = dataDict
        arm_length    = self.robots['0'].d * 10**(2)
        self.armb1    = np.array([[arm_length], [0] ,[0]])
        self.armb1    = Rz(0) @ self.armb1
        self._armb1   = -self.armb1.copy()
        self.armb2    = Rz(np.pi/2) @ (self.armb1.reshape(3,))
        self._armb2   = Rz(np.pi/2) @ (self._armb1.reshape(3,))
 
    def startAnimation(self,fig,ax,videoname,show,dt):
        self.fig = fig
        self.ax  = ax
        self.ani = animation.FuncAnimation(self.fig, self.animate, frames=len(self.dataDict['0'][0]), interval=dt*1000,blit=True)
        self.ani.save(videoname)
        if show:
            plt.show()

    def animate(self,i):
        self.ax.cla()
        self.setlimits()
        for robot in self.robots.keys():
           self.full_state              = self.dataDict[robot][0]
           self.reference_state         = self.dataDict[robot][1] 
           x, y, z, q                   = self.getCurrState(i)
           xref,yref,zref, qref         = self.getRefState(i) 
           armI1, armI2, _armI1, _armI2 = self.getArmpos(x[i],y[i],z[i],q)
           self.drawQuivers(x[i],y[i],z[i], q, xref[i], yref[i], zref[i], qref)
           self.drawActvsRefTraj(x, y, z, xref, yref, zref)
           self.drawQuadrotorArms(x[i], y[i], z[i], armI1, armI2, _armI1, _armI2)

           Xb,Yb,Zb = RotatedCylinder(0,0,0.1,0.1,q) 
           self.drawPropellers(Xb, Yb, Zb,armI1, armI2, _armI1, _armI2)
        self.line, = self.ax.plot([0,0.0001], [0,0.0001], [0,0.0001], 'b--', lw=0.1)   
        return self.line,

    def setlimits(self):
        self.ax.set_xlim3d([-2, 2])
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d([-2, 2])
        self.ax.set_ylabel('Y')
        self.ax.set_zlim3d([-2, 2])
        self.ax.set_zlabel('Z')

    def drawQuivers(self, x, y, z, q, xref, yref, zref, qref):
        R_i = to_matrix(q)
        u = R_i[:,0]
        v = R_i[:,1]
        w = R_i[:,2]
        Rd_i = to_matrix(qref)
        ud = Rd_i[:,0]
        vd = Rd_i[:,1]
        wd = Rd_i[:,2]
        self.vec1  = self.ax.quiver(x,y,z, u[0], u[1] ,u[2],color = 'r', length = 0.2)
        self.vec2  = self.ax.quiver(x,y,z, v[0], v[1] ,v[2],color = 'g', length = 0.2)
        self.vec3  = self.ax.quiver(x,y,z, w[0], w[1] ,w[2],color = 'b', length = 0.2)
        self.vec1r = self.ax.quiver(xref,yref,zref, ud[0], ud[1] ,ud[2],color = 'r', length = 0.5)
        self.vec2r = self.ax.quiver(xref,yref,zref, vd[0], vd[1] ,vd[2],color = 'g', length = 0.5)
        self.vec3r = self.ax.quiver(xref,yref,zref, wd[0], wd[1] ,wd[2],color = 'b', length = 0.5)
        
    def getCurrState(self, i):
        x = self.full_state[:i+1,0]
        y = self.full_state[:i+1,1]
        z = self.full_state[:i+1,2]
        q = self.full_state[i,6:10].reshape(4,)
        return x, y, z, q

    def getRefState(self, i):
        xref = self.reference_state[:i+1,0]
        yref = self.reference_state[:i+1,1]
        zref = self.reference_state[:i+1,2]
        qref = self.reference_state[i,6:10].reshape(4,)
        return xref, yref, zref, qref

    def getArmpos(self, x, y, z, q):
        R_i      = to_matrix(q)
        position = np.array([x, y, z]) 
        armI1    = position + R_i @ (self.armb1.reshape(3,))
        _armI1   = position + R_i @ (self._armb1.reshape(3,))
        armI2    = position + R_i @(self.armb2.reshape(3,))
        _armI2   = position + R_i @ (self._armb2.reshape(3,))
        return armI1, armI2, _armI1, _armI2

    def drawActvsRefTraj(self, x, y, z, xref, yref, zref):
            self.ax.plot3D(x, y, z, 'b--',lw=1.5)
            self.ax.plot3D(xref, yref ,zref,'g--',lw=2)

    def drawQuadrotorArms(self, x, y, z, armI1, armI2, _armI1, _armI2):
        self.ax.plot3D(np.linspace(x, armI1[0]), np.linspace(y, armI1[1]), np.linspace(z, armI1[2]),'k',lw=2)
        self.ax.plot3D(np.linspace(x, _armI1[0]), np.linspace(y, _armI1[1]), np.linspace(z, _armI1[2]),'k',lw=2)
        
        self.ax.plot3D(np.linspace(x, armI2[0]), np.linspace(y, armI2[1]), np.linspace(z, armI2[2]),'k',lw=2)
        self.ax.plot3D(np.linspace(x, _armI2[0]), np.linspace(y, _armI2[1]), np.linspace(z, _armI2[2]),'k',lw=2)

    def drawPropellers(self, Xb, Yb, Zb,armI1, armI2, _armI1, _armI2):
        self.ax.plot_surface(Xb+armI1[0], Yb+armI1[1], Zb+armI1[2], alpha=1)
        self.ax.plot_surface(Xb+_armI1[0], Yb+_armI1[1], Zb+_armI1[2], alpha=1)
        self.ax.plot_surface(Xb+armI2[0], Yb+armI2[1], Zb+armI2[2], alpha=1)
        self.ax.plot_surface(Xb+_armI2[0], Yb+_armI2[1], Zb+_armI2[2], alpha=1)
 