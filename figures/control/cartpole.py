import matplotlib.pyplot as plt
import numpy as np
import math
from klampt.math import so2
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

class Cartpole:
    """Defines a cart-pole system with the first element x translation and the
    second element returning the angle of the pole w.r.t to the x axis."""
    def __init__(self,m=[1.,1.],L=1.,g=9.8):
        self.m = m[:]
        self.L = L
        self.g = g
        self.viscosity = None
        self.umin = None
        self.umax = None
        
    def dynamics_terms(self,q,dq):
        """Returns elements B,C,G of the dynamics equation for a given state (q,dq)"""
        #unpack elements
        x,theta=q[0],q[1]
        dx,dtheta=dq[0],dq[1]
        m1,m2=self.m
        L=self.L
        s,c = math.sin(theta),math.cos(theta)
        B = np.array([[m1+m2, -m2*L/2*s],
                      [-m2*L/2*s, m2*L*L/4]])
        C = np.array([-m2*L/2*dtheta**2*c,0])
        G = np.array([0,m2*self.g*c])
        return B,C,G

    def dynamics(self,q,dq,u):
        """Returns forward dynamics q''=f(q,q',u).  Note that u is the
        torque applied to both the cart and the pole, so if you just want to
        control the cart, set the second term of u to 0."""
        B,C,G = self.dynamics_terms(q,dq)
        visc = np.zeros(2)
        if self.viscosity is not None:
            visc = -self.viscosity*dq
        ddq = np.dot(np.linalg.inv(B),u+visc-G-C)
        return ddq

    def inverse_dynamics(self,q,dq,ddq):
        """Returns inverse dynamics u=g(q,q',q'').  Note that u is the
        torque applied to both the cart and the pole, so if you just want to
        control the cart, set the second term of u to 0."""
        B,C,G = self.dynamics_terms(q,dq)
        visc = np.zeros(2)
        if self.viscosity is not None:
            visc = -self.viscosity*dq
        u = np.dot(B,ddq)+G+C-visc
        return u

    def simulate(self,q0,dq0,ufunc,dt=1e-3,T=1):
        """Returns a simulation trace of the cart pole problem using Euler
        integration.  ufunc is a policy u(t,q,dq)"""
        q = q0.copy()
        dq = dq0.copy()
        res = dict((idx,[]) for idx in ['t','q','dq','u','ddq'])
        t = 0
        while t < T:
            u = ufunc(t,q,dq)
            if self.umin is not None:
                raise NotImplementedError("handling bounds not done yet")
            #print t,q,dq,u
            ddq = self.dynamics(q,dq,u)
            res['t'].append(t)
            res['q'].append(q)
            res['dq'].append(dq)
            res['ddq'].append(ddq)
            res['u'].append(u)
            q = q+dt*dq
            dq = dq+dt*ddq
            q[1]=q[1]%(math.pi*2)
            t += dt
        return res    

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
    """Standard PD controller"""
    return PD([so2.diff(q[0],qdes[0]),q[1]-qdes[1]],dq,KP,KD)

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

