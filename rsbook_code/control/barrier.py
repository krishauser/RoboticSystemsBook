from .objective import ObjectiveFunction
from ..utilities import differences
import numpy as np

class BarrierPenalty(ObjectiveFunction):
    """Objective function that uses a barrier on a distance function
    for all states: J(x,u) = sum alpha(d(x[i])) + alpha(d[x[T]).
    
    The distance function should be positive outside of the constraint
    and negative inside of it.
    """
    def __init__(self,dfunc,dfunc_gradient=None,dfunc_hessian=None,
                 barriertype='quadratic',dmax=1):
        self.dfunc = dfunc
        self.dfunc_gradient = dfunc_gradient
        self.dfunc_hessian = dfunc_hessian
        self.barriertype = barriertype
        if barriertype not in ['log','inv','quadratic']:
            raise ValueError("Barrier type must be either log, inv or quadratic")
        self.dmax = dmax
    def __str__(self):
        return self.barriertype + self.__class__.__name__
    def incremental(self,x,u=None):
        return self.barrier(self.dfunc(x))
    def terminal(self,x):
        return self.barrier(self.dfunc(x))
    def incremental_gradient(self,x,u):
        return self.terminal_gradient(x),np.zeros(len(u))
    def incremental_hessian(self,x,u):
        return self.terminal_hessian(x),np.zeros((len(x),len(u))),np.zeros((len(u),len(u)))
    def terminal_gradient(self,x):
        d = self.dfunc(x)
        if self.dfunc_gradient:
            gd = self.dfunc_gradient(x)
        else:
            if hasattr(d,'__iter__'):
                gd = differences.jacobian_forward_difference(self.dfunc,x,1e-4)
            else:
                gd = differences.gradient_forward_difference(self.dfunc,x,1e-4)
        return self.barrier_gradient(d,gd)
    def terminal_hessian(self,x):
        d = self.dfunc(x)
        if hasattr(d,'__iter__') or not self.dfunc_hessian:
            return self.terminal_hessian_diff(x)
        if self.dfunc_hessian:
            Hd = self.dfunc_hessian(x)
        if self.dfunc_gradient:
            gd = self.dfunc_gradient(x)
        else:
            gd = differences.gradient_forward_difference(self.dfunc,x,1e-4)
        return self.barrier_hessian(d,gd,Hd)  
        
    def barrier(self,d):
        if hasattr(d,'__iter__'):
            return sum(self.barrier(di) for di in d)
        if self.barriertype == 'quadratic':
            return max(self.dmax-d,0)**2
        elif self.barriertype == 'log':
            if d < 0:
                return np.inf
            return -np.log(d)
        elif self.barriertype == 'inv':
            if d < 0:
                return np.inf
            if d >= self.dmax:
                return 0
            return 1/d-2/self.dmax + d/self.dmax**2
        else:
            raise ValueError("Invalid barrier type?")
    
    def barrier_derivative(self,d):
        if self.barriertype == 'quadratic':
            if self.dmax <= d:
                return 0
            return 2*(d-self.dmax)
        elif self.barriertype == 'log':
            if d < 0:
                return 0
            return -1/d
        elif self.barriertype == 'inv':
            if d < 0:
                return 0
            if d >= self.dmax:
                return 0
            return -1/d**2 + 1/self.dmax**2
        else:
            raise ValueError("Invalid barrier type?")
    
    def barrier_derivative2(self,d):
        if self.barriertype == 'quadratic':
            if self.dmax <= d:
                return 0
            return 2
        elif self.barriertype == 'log':
            if d < 0:
                return 0
            return 1/d**2
        elif self.barriertype == 'inv':
            if d < 0:
                return 0
            if d >= self.dmax:
                return 0
            return 2/d**3
        else:
            raise ValueError("Invalid barrier type?")
            
    def barrier_gradient(self,d,gd):
        if hasattr(d,'__iter__'):
            return sum(self.barrier_gradient(di,gdi) for (di,gdi) in zip(d,gd))
        return self.barrier_derivative(d)*gd
    
    def barrier_hessian(self,d,gd,Hd):
        return self.barrier_derivative2(d)*Hd + np.outer(gd,gd)*self.barrier_derivative(d)
    