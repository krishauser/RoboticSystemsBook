from __future__ import print_function,division
from ..utilities import differences
import numpy  as np

MAX_INTEGRATION_STEPS = 10000

class ControlSpace:
    """A state space with states x and control variable u.  The primary function of
    this class is to define the transformation
    
        xnext = nextState(x,u).
    """
    def __init__(self):
        pass
    def stateDimension(self):
        """Returns the dimension of the state space associated with this."""
        raise NotImplementedError()
    def controlDimension(self):
        """Returns the set from which controls should be sampled"""
        raise NotImplementedError()
    def nextState(self,x,u):
        """Produce the next state after applying control u to state x"""
        raise NotImplementedError()
    def interpolator(self,x,u):
        """Returns the interpolator that goes from x to self.nextState(x,u)"""
        raise NotImplementedError()
    def connection(self,x,y):
        """Returns a sequence of controls that connects x to y, if
        applicable"""
        return None
        
    def nextState_jacobian(self,x,u):
        """Returns a pair of Jacobian matrices dx_n/dx, dx_n/du of
        the nextState(x,u) function.
        
        Subclasses can use nextState_jacobian_diff to approximate
        the Jacobian."""
        return self.nextState_jacobian_diff(x,u)
    
    def nextState_jacobian_diff(self,x,u,h=1e-4):
        Jx = differences.jacobian_forward_difference((lambda y:self.nextState(y,u)),x,h)
        Ju = differences.jacobian_forward_difference((lambda v:self.nextState(x,v)),u,h)
        return (Jx,Ju)

    def checkDerivatives(self,x,u,baseTol=1e-3):
        Jx,Ju = self.nextState_jacobian(x,u)
        dJx,dJu = self.nextState_jacobian_diff(x,u)
        xtol = max(1,np.linalg.norm(dJx))*baseTol
        utol = max(1,np.linalg.norm(dJu))*baseTol
        res = True
        if np.linalg.norm(Jx-dJx) > xtol:
            print("Derivatives of ControlSpace",self.__class__.__name__,"incorrect in Jacobian x by",np.linalg.norm(Jx-dJx),"diff norm",np.linalg.norm(dJx))
            #print("  Computed",Jx)
            #print("  Differenced",dJx)
            res = False
        if np.linalg.norm(Ju-dJu) > utol:
            print("Derivatives of ControlSpace",self.__class__.__name__,"incorrect in Jacobian u by",np.linalg.norm(Ju-dJu),"diff norm",np.linalg.norm(dJu))
            #print("  Computed",Ju)
            #print("  Differenced",dJu)
            res = False
        return res


class LTIControlSpace(ControlSpace):
    """Implements a discrete-time, linear time invariant control
    space f(x,u) = Ax+Bu.
    """
    def __init__(self,A,B):
        self.A = A
        self.B = B
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A matrix must be square")
        if self.A.shape[0] != self.B.shape[0]:
            raise ValueError("Matrices are of incompatible shape")
    def stateDimension(self):
        return self.A.shape[0]
    def controlDimension(self):
        return self.B.shape[1]
    def nextState(self,x,u):
        return self.A.dot(x) + self.B.dot(u)
    def connection(self,x,y):
        #TODO: solve for multi-step control (if controllable)
        if self.B.shape[1] < self.B.shape[0]: return None
        xn = self.A.dot(x)
        dx = np.asarray(y)-np.asarray(xn)
        Binv = np.linalg.pinv(self.B)
        return [Binv.dot(dx)]
    def nextState_jacobian(self,x,u):
        return self.A,self.B


class Dynamics:
    """A differential equation relating state x to control variable u.
    The derivative() method gives diffential constraints x'=f(x,u)"""
    def stateDimension(self):
        raise NotImplementedError()
    def controlDimension(self):
        raise NotImplementedError()
    def derivative(self,x,u):
        raise NotImplementedError()
    def derivative_jacobian(self,x,u):
        """Returns a pair of Jacobian matrices dx'/dx, dx'/du of
        the derivative(x,u) function.
        """
        return self.derivative_jacobian_diff(x,u)
        
    def derivative_jacobian_diff(self,x,u,h=1e-4):
        Jx = differences.jacobian_forward_difference((lambda y:self.derivative(y,u)),x,h)
        Ju = differences.jacobian_forward_difference((lambda v:self.derivative(x,v)),u,h)
        return (Jx,Ju)

    def checkDerivatives(self,x,u,baseTol=1e-3):
        Jx,Ju = self.derivative_jacobian(x,u)
        dJx,dJu = self.derivative_jacobian_diff(x,u)
        xtol = max(1,np.linalg.norm(dJx))*baseTol
        utol = max(1,np.linalg.norm(dJu))*baseTol
        res = True
        if np.linalg.norm(Jx-dJx) > xtol:
            print("Derivatives of Dynamics",self.__class__.__name__,"incorrect in Jacobian x by",np.linalg.norm(Jx-dJx),"diff norm",np.linalg.norm(dJx))
            #print("  Computed",Jx)
            #print("  Differenced",dJx)
            res = False
        if np.linalg.norm(Ju-dJu) > utol:
            print("Derivatives of Dynamics",self.__class__.__name__,"incorrect in Jacobian u by",np.linalg.norm(Ju-dJu),"diff norm",np.linalg.norm(dJu))
            #print("  Computed",Ju)
            #print("  Differenced",dJu)
            res = False
        return res

    def integrate(self,x,dx):
        """For non-Euclidean (e.g., geodesic) spaces, implement this."""
        return x+dx


class IntegratorControlSpace (ControlSpace):
    """A control space that performs integration of a differential equation
    to determine the next state.

    The derivative function x'=f(x,u) is translated into a nextState
    function xnext = g(x,u) via integration over the duration T at the
    resolution dt.
    
    Euler integration is used.  TODO: use higher order methods.
    """
    def __init__(self,dynamics,T=1.0,dt=0.01):
        self.dynamics = dynamics
        self.T = T
        self.dt = dt
    def stateDimension(self):
        return self.dynamics.stateDimension()
    def controlDimension(self):
        return self.dynamics.controlDimension()
    def trajectory(self,x,u):
        duration = self.T
        path = [x]
        t = 0.0
        assert self.dt > 0
        dt0 = self.dt
        if duration / self.dt > MAX_INTEGRATION_STEPS:
            print("Warning, more than",MAX_INTEGRATION_STEPS,"steps requested for IntegratorControlSpace",self.__class__.__name__)
            dt0 = duration/MAX_INTEGRATION_STEPS
        while t < duration:
            dx = self.dynamics.derivative(path[-1],u)
            assert len(dx)==len(x),"Derivative %s dimension not equal to state dimension: %d != %d"%(self.dynamics.__class__.__name__,len(dx),len(x))
            dt = min(dt0,duration-t)
            xnew = self.dynamics.integrate(path[-1],dx*dt)
            path.append(xnew)
            t = min(t+dt0,duration)
        return path
    def nextState(self,x,u):
        return self.trajectory(x,u)[-1]
    def interpolator(self,x,u):
        return self.trajectory(x,u)


def simulate(dynamics,x0,ufunc,T=1,dt=1e-3):
    """Returns a simulation trace of dynamics using Euler integration over
    duration T and time step dt. 
    
    Args:
        dynamics (Dynamics): the system.
        x0 (np.ndarray): the initial state.
        ufunc (callable): a policy u(t,x) returning a control vector.
        T (float): integration duration
        dt (float): time step
    
    Returns:
        dict: maps 't', 'x', 'u', 'dx' to traces of these time, state, control,
        and derivative, respectively.
    """
    assert len(x0)==dynamics.stateDimension()
    res = dict((idx,[]) for idx in ['t','x','u','dx'])
    t = 0
    while t < T:
        u = ufunc(t,x0)
        assert len(u) == dynamics.controlDimension()
        dx = dynamics.derivative(x0,u)
        res['t'].append(t)
        res['x'].append(x0)
        res['dx'].append(dx)
        res['u'].append(u)
        x0 = dynamics.integrate(x0,dt*dx)
        t += dt
    return res
