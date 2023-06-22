from __future__ import print_function,division
import math
import numpy as np
from ..dynamics import Dynamics

class DoubleIntegrator(Dynamics):
    """A double integrator x=(q,q') and u=q''. The state space is
    2d-dimensional and u is d-dimensional.
    """
    def __init__(self,d):
        if d <= 0:
            raise ValueError("Invalid dimension d, must be >= 1")
        self.d = d
    def stateDimension(self):
        return self.d*2
    def controlDimension(self):
        return self.d
    def derivative(self,x,u):
        return np.concatenate((x[self.d:],u))
    def derivative_jacobian(self,x,u):
        dx = np.block([[np.zeros((self.d,self.d)),np.eye(self.d)],[np.zeros((self.d,self.d)),np.zeros((self.d,self.d))]])
        du = np.block([[np.zeros((self.d,self.d))],[np.eye(self.d)]])
        return dx,du
