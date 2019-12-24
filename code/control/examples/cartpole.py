from __future__ import print_function,division
import math
import numpy as np

class Cartpole:
    """Defines a cart-pole system with the state space (x,theta,x',theta')
    where x is the translation and theta is the angle of the pole
    w.r.t to the x axis.
    
    The control u=(fcart,tpole) is a force applied to the cart and a torque
    applied to the pole.
    """
    def __init__(self,m=[1.,1.],L=1.,g=9.8):
        self.m = m[:]
        self.L = L
        self.g = g
        self.viscosity = None
        self.umin = None
        self.umax = None
    
    def derivative(self,x,u):
        """Returns x' = f(x,u)"""
        assert len(x) == 4
        assert len(u) == 2
        q = x[:2]
        dq = x[2:]
        ddq = self.dynamics(q,dq,u)
        return np.hstack((dq,ddq))
    
    def next_state(self,x,u,dt,timestep=1e-3):
        """Simulates forward for time dt and returns the resulting state x"""
        assert len(x) == 4
        assert len(u) == 2
        q = x[:2]
        dq = x[2:]
        if not callable(u):
            u = lambda t,q,dq:u
        res = self.simulate(q,dq,u,dt,timestep)
        return np.hstack((res['q'][-1],res['dq'][-1]))
        
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

    def simulate(self,q0,dq0,ufunc,T=1,dt=1e-3):
        """Returns a simulation trace of the cart pole problem using Euler
        integration.  ufunc is a policy u(t,q,dq)"""
        q = np.asarray(q0).copy()
        dq = np.asarray(dq0).copy()
        res = dict((idx,[]) for idx in ['t','q','dq','u','ddq'])
        t = 0
        while t < T:
            u = ufunc(t,q,dq)
            assert len(u) == 2
            if self.umin is not None:
                u = [max(a,min(b,v)) for (a,b,v) in zip(self.umin,self.umax,u)]
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
    