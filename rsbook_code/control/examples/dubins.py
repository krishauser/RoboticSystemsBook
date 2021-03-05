from __future__ import print_function,division
import math
from klampt.math import so2,vectorops
from ..dynamics import Dynamics,ControlSpace

def cmp(x,y):
    if x < y: return -1
    elif x > y: return 1
    return 0
    
class DubinsCar(Dynamics):
    """Defines a first-order Dubins car state space with x = (tx,ty,theta) 
    and u = (fwd_velocity,turnRate).
    """
    def __init__(self,turnRateMin=-1,turnRateMax=1):
        self.turnRateRange = [turnRateMin,turnRateMax]
        self.distanceRange = [-float('inf'),float('inf')]
    
    def stateDimension(self):
        return 3
    
    def controlDimension(self):
        return 2

    def derivative(self,x,u):
        """Returns x' = f(x,u)"""
        assert len(x) == 3
        assert len(u) == 2
        pos = [x[0],x[1]]
        fwd = [math.cos(x[2]),math.sin(x[2])]
        right = [-fwd[1],fwd[0]]
        phi = u[1]
        d = u[0]
        return [fwd[0]*d,fwd[1]*d,phi]
        

class DubinsCarIntegrator(ControlSpace):
    def __init__(self,T=1,dt=1e-2):
        self.T = T
        self.dt = dt

    def nextState(self,x,u):
        assert len(x) == 3
        assert len(u) == 2
        pos = [x[0],x[1]]
        fwd = [math.cos(x[2]),math.sin(x[2])]
        right = [-fwd[1],fwd[0]]
        phi = u[1]
        d = u[0]*self.T
        if abs(phi)<1e-8:
            newpos = vectorops.madd(pos,fwd,d)
            return newpos + [x[2]]
        else:
            #rotate about a center of rotation, with radius 1/phi
            cor = vectorops.madd(pos,right,1.0/phi)
            sign = cmp(d*phi,0)
            d = abs(d)
            phi = abs(phi)
            theta=0
            thetaMax=d*phi
            newpos = vectorops.add(so2.apply(sign*thetaMax,vectorops.sub(pos,cor)),cor)
            return newpos + [so2.normalize(x[2]+sign*thetaMax)]
    
    def trajectory(self,x0,ufunc):
        """Returns a simulation trace of the Dubins problem using Euler
        integration.  ufunc is a policy u(t,x)"""
        x = x0
        res = dict((idx,[]) for idx in ['t','x','u','dx'])
        t = 0
        while t < self.T:
            u = ufunc(t,x)
            assert len(u) == 2
            u = list(u)
            u[0] = max(self.distanceRange[0],min(self.distanceRange[1],u[0]))
            u[1] = max(self.turnRateRange[0],min(self.turnRateRange[1],u[1]))
            #print t,x,u
            dx = self.derivative(x,u)
            res['t'].append(t)
            res['x'].append(x)
            res['dx'].append(dx)
            res['u'].append(u)
            uprime = [u[0]*self.dt,u[1]]
            xn = self.nextState(x,uprime)
            x = xn
            t += self.dt
        return res