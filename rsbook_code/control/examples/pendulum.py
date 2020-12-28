from __future__ import print_function,division
import math
import numpy as np

class Pendulum:
    """Defines a pendulum state space.  The state x is (theta,theta') and the
    control is u=torque.
    """
    def __init__(self,m=1.,L=1.,g=9.8):
        self.m = m
        self.L = L
        self.g = g
        self.viscosity = None
        self.umin = None
        self.umax = None
    
    def derivative(self,x,u):
        """Returns x' = f(x,u)"""
        assert len(x) == 2
        assert isinstance(u,(int,float)) or len(u) == 1
        q = x[0]
        dq = x[1]
        if hasattr(u,'__iter__'):
            u = u[0]
        ddq = self.dynamics(q,dq,u)
        return np.array([dq,ddq])
    
    def next_state(self,x,u,dt,timestep=1e-3):
        """Simulates forward for time dt and returns the resulting state x"""
        assert len(x) == 2
        assert callable(u) or isinstance(u,(int,float)) or len(u) == 1
        q = x[0]
        dq = x[1]
        if not callable(u):
            if hasattr(u,'__iter__'):
                u = u[0]
            u = lambda t,q,dq:u
        return self.simulate(q,dq,u,dt,timestep)['x'][-1]
        
    def dynamics(self,theta,dtheta,u):
        """Returns forward dynamics theta''=f(theta,theta',u).  u is the
        torque applied to the system."""
        c = math.cos(theta)
        s = math.sin(theta)
        m = self.m
        L = self.L
        visc=0.0
        if self.viscosity is not None:
            visc = -self.viscosity*dtheta
        ddtheta = (u+visc-m*c*self.g)/(L*m)
        return ddtheta

    def inverse_dynamics(self,q,dq,ddq):
        """Returns inverse dynamics u=g(theta,theta',theta'')"""
        c = math.cos(theta)
        s = math.sin(theta)
        m = self.m
        visc=0.0
        if self.viscosity is not None:
            visc = -self.viscosity*dtheta
        u = L*m*(ddtheta-visc+m*c*self.g)
        return u

    def simulate(self,q0,dq0,ufunc,dt=1e-3,T=1):
        """Returns a simulation trace of the pendulum problem using Euler
        integration.  ufunc is a policy u(t,q,dq)"""
        q = q0
        dq = dq0
        res = dict((idx,[]) for idx in ['t','x','q','dq','u','ddq'])
        t = 0
        while t < T:
            u = ufunc(t,q,dq)
            if self.umin is not None:
                u = max(u,self.umin)
            if self.umax is not None:
                u = min(u,self.umax)
            #print t,q,dq,u
            ddq = self.dynamics(q,dq,u)
            res['t'].append(t)
            res['q'].append(q)
            res['dq'].append(dq)
            res['ddq'].append(ddq)
            res['x'].append(np.array([q,dq]))
            res['u'].append(u)
            q = q+dt*dq
            dq = dq+dt*ddq
            q=q%(math.pi*2)
            t += dt
        return res    
