from __future__ import print_function,division
import matplotlib.pyplot as plt
import numpy as np
import math
from klampt.math import so2
from plotting import *
from code.control.examples.pendulum import Pendulum

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def PD(x,dx,KP,KD):
    u = -np.dot(KP,x)-np.dot(KD,dx)
    return u

def PID(x,I,dx,KP,KI,KD):
    u = -np.dot(KP,x)-np.dot(KI,I)-np.dot(KD,dx)
    return u

#create a pendulum instance -- can change parameters in the constructor or below
pendulum = Pendulum()
pendulum.umin = -20.0
pendulum.umax = 20.0

#initial state: swinging from the bottom
q0=2*math.pi-math.pi/2

#initial state: near the top, with a 0.2 radian disturbance
#q0=math.pi/2+0.2

dq0=0.0
qdes = math.pi/2
KP=10.0
KD=2.0

def pdcontrol(t,q,dq):
    """Standard PD controller"""
    return PD(so2.diff(q,qdes),dq,KP,KD)

res1 = pendulum.simulate(q0,dq0,pdcontrol,dt=0.001,T=8)
pendulum.umin = -5.0
pendulum.umax = 5.0
def clamp(x,a,b):
    return min(b,max(a,x))
rc('xtick', labelsize=16) 
rc('ytick', labelsize=16) 
plotflow([0,math.pi*2],[-6,6],100,
         lambda x:[x[1],pendulum.dynamics(x[0],x[1],pdcontrol(0,x[0],x[1]))],
         lambda x,u:pdcontrol(0,x[0],x[1]),linewidth=1,cmap=plt.cm.Set1)
plt.xlabel(r'$\theta$',fontsize=20)
plt.ylabel(r'$\dot{\theta}$',fontsize=20)
plt.plot(res1['q'],res1['dq'],color='k')
plt.scatter([math.pi/2,math.pi*3/2], [0,0], s=80, c=['r','g'], marker=(5, 1))
plt.tight_layout()
plt.show()

plotflow([0,math.pi*2],[-6,6],100,
         lambda x:[x[1],pendulum.dynamics(x[0],x[1],-5.0)],
          colorfunc=lambda x,u:-5.0,linewidth=1)
plt.scatter([math.pi/2,math.pi*3/2], [0,0], s=80, c=['r','g'], marker=(5, 1))
plt.xlabel(r'$\theta$',fontsize=20)
plt.ylabel(r'$\dot{\theta}$',fontsize=20)
plt.tight_layout()
plt.show()
plotflow([0,math.pi*2],[-6,6],100,
         lambda x:[x[1],pendulum.dynamics(x[0],x[1],5.0)],
          colorfunc=lambda x,u:5.0,linewidth=1)
plt.scatter([math.pi/2,math.pi*3/2], [0,0], s=80, c=['r','g'], marker=(5, 1))
plt.xlabel(r'$\theta$',fontsize=20)
plt.ylabel(r'$\dot{\theta}$',fontsize=20)
plt.tight_layout()
plt.show()

res2 = pendulum.simulate(q0,dq0,pdcontrol,dt=0.001,T=5)
q0 = math.pi/2-0.5
res3 = pendulum.simulate(q0,dq0,pdcontrol,dt=0.001,T=5)
#res1['qdes'] = [[qdes]]*len(res1['t'])
#plotmulti([res1,res1,res1],['theta','dtheta','theta des','u'],['x','x','qdes','u'],[0,1,0,None],['black','green','red','blue'])
plt.figure(figsize=(6,6))
plt.xlabel('$\theta$')
plt.ylabel('$\dot{\theta}$')
plt.ylim(-10,10)
plotxy([res1,res2,res3],['umax=20','umax=5','umax=5 stabilizing'],['x','x','x'],[0,1],['black','green','red','blue'])
plt.show()
