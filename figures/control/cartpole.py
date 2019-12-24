from __future__ import print_function,division
import matplotlib.pyplot as plt
import numpy as np
import math
from klampt.math import so2
from code.control.examples.cartpole import Cartpole
from plotting import *

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


#create a cartpole instance -- can change parameters in the constructor or below
cartpole = Cartpole()
#cartpole.viscosity = np.array([0.1,0.1])

#initial state: swinging from the bottom
q0=np.array([0.,2*math.pi-math.pi/2])

#initial state: near the top, with a 0.2 radian disturbance
#q0=np.array([0.,math.pi/2+0.2])

dq0=np.array([0.,0.])
qdes = np.array([0.,math.pi/2])
KP=np.array([[3,-0.8],
             [0.,0.]])
KD=np.array([[0.8,-2.0],
             [0.,0.]])

def pdcontrol(t,q,dq):
    """Standard PD controller for cartpole problem"""
    return PD([q[0]-qdes[0],so2.diff(q[1],qdes[1])],dq,KP,KD)

def sinusoidal_accel(t,q,dq):
    #3 swings results
    numswings = 3
    #no spin
    #period = 1.11
    #spin
    period = 1.12
    period = 1.288
    #no spin
    period = 1.29
    period = 1.31
    #spin again
    period = 1.32
    period = 1.39
    #no spin
    period = 1.4
    #very out of phase
    period = 1.6
    period = 1.288
    """
    #2 swings results
    numswings = 2.0
    #no spin
    #period = 1.21
    #spin on backward swing -- probably an integration artifact
    #period = 1.22
    #spin on forward swing
    #period = 1.23
    period = 1.25
    period = 1.313
    #no swing
    #period = 1.32
    """
    amp = 0.5
    if t < numswings*period/2:
        return [amp*(2*math.pi/period)**2*math.cos(t*2*math.pi/period)]
    return [0,0]

def startstop_accel(t,q,dq):
    #no swing
    #period = 0.30
    #big swing
    #period = 0.325
    #no swing
    #period = 0.35
    period = 0.375
    strength = 7
    if t < 1*period:
        return [strength,0]
    elif t < 3*period:
        return [-strength,0]
    elif t < 5*period:
        return [strength,0]
    elif t < 6*period:
        return [-strength,0]
    else:
        return [0,0]
    
def startstop(t,q,dq):
    #partial feedback linearization
    #u = Bq''+CG
    #q'' = B^-1 (u-CG)
    #want to set x with u[0]=x, u[1]=0 so that q''[0] is taken on as input
    #let B^-1 = [[a,b],[c,d]]
    #q''[0] = a u[0]-[a,b]*CG
    global cartpole
    #ddq = startstop_accel(t,q,dq)
    ddq = sinusoidal_accel(t,q,dq)
    B,C,G = cartpole.dynamics_terms(q,dq)
    Binv = np.linalg.inv(B)
    #natural dynamics
    ddq_nat = np.dot(Binv,-C-G)
    u=np.array([0.,0.])
    u[0] = (1.0/Binv[0,0])*(ddq[0]-ddq_nat[0])
    return u

#res1 = cartpole.simulate(q0,dq0,pdcontrol,dt=0.001,T=10)
res1 = cartpole.simulate(q0,dq0,startstop,dt=0.001,T=5.0)
res1['qdes'] = [qdes]*len(res1['t'])
plt.figure(figsize=(4,3))
plt.xlabel('t')
plotmulti([res1,res1,res1],['x',r'$\theta$',r'$\theta$ up'],['q','q','qdes'],[0,1,1],['black','green','red','blue'])
plt.tight_layout()
plt.show()
plotxy([res1],['plot'],['q'],[0,1],['black','green','red','blue'])
#plotxy([res1,res1],['x','u'],['x','u'],[0,1],['black','green','red','blue'])
plt.show()

