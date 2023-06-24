from __future__ import print_function,division
from builtins import range
from six import iteritems

from ..utilities import differences
import numpy as np


class ObjectiveFunction:
    """Objective function base class.  Measures the cost of a trajectory
    [x0,...,xn],[u1,...,un] or a path [x0,...,xn].
    
    To use numerical optimizers, subclasses should implement the
    incremental_gradient, incremental_hessian, terminal_gradient,
    and terminal_hessian functions.  Finite difference functions are
    provided for a subclass implementer's convenience.  If you are writing
    your own functions, you may use self.checkDerivatives(...) to self-test
    your derivative functions.
    
    Objective functions can be scaled by multiplication (* operator), and added
    together (+ operator).
    """
    def __str__(self):
        return self.__class__.__name__
    def incremental(self,x,u=None):
        return 0.0
    def terminal(self,x):
        return 0.0
    def cost(self,xpath,upath):
        """Helper: returns the total cost of a given state,control trajectory"""
        if upath is None:
            c = 0.0
            for i in range(len(xpath)-1):
                c += self.incremental(xpath[i],None)
            return c + self.terminal(xpath[-1])
        assert len(xpath)==len(upath)+1
        c = 0.0
        for i in range(len(upath)):
            c += self.incremental(xpath[i],upath[i])
        return c + self.terminal(xpath[-1])
    
    def incremental_gradient(self,x,u):
        """Return a pair (df/dx,df/du). """
        return np.zeros(len(x)),np.zeros(len(u))
    def incremental_hessian(self,x,u):
        """Return a triple (d^2f/dx^2,d^2f/dxdu,d^2f/du^2)"""
        return np.zeros((len(x),len(x))),np.zeros((len(x),len(u))),np.zeros((len(u),len(u)))
    def terminal_gradient(self,x):
        """Return df/dx"""
        return np.zeros(len(x))
    def terminal_hessian(self,x):
        """Return d^2f/dx^2.  Return terminal_hessian_diff """
        return np.zeros((len(x),len(x)))
    
    #subclasses can copy and paste the below code to use finite-differences to approximate
    #gradients.
    #def incremental_gradient(self,x,u):
    #    return self.incremental_gradient_diff(x,u)
    #def incremental_hessian(self,x,u):
    #    return self.incremental_hessian_diff(x,u)
    #def terminal_gradient(self,x):
    #    return self.terminal_gradient_diff(x)
    #def terminal_hessian(self,x):
    #    return self.terminal_hessian_diff(x)
    
    def incremental_gradient_diff(self,x,u,h=1e-5):
        return (differences.gradient_forward_difference((lambda y:self.incremental(y,u)),x,h),
            differences.gradient_forward_difference((lambda v:self.incremental(x,v)),u,h))
    def incremental_hessian_diff(self,x,u,h=1e-4):
        return (differences.hessian_forward_difference((lambda y:self.incremental(y,u)),x,h),
            differences.hessian2_forward_difference(self.incremental,x,u,h),
            differences.hessian_forward_difference((lambda v:self.incremental(x,v)),u,h))
    def terminal_gradient_diff(self,x,h=1e-5):
        return differences.gradient_forward_difference(self.terminal,x,h)
    def terminal_hessian_diff(self,x,h=1e-4):
        return differences.hessian_forward_difference(self.terminal,x,h)
    
    def checkDerivatives(self,x,u,baseTol=1e-3):
        gx,gu = self.incremental_gradient(x,u)
        Hxx,Hxu,Huu = self.incremental_hessian(x,u)
        tx = self.terminal_gradient(x)
        txx = self.terminal_hessian(x)
        dgx,dgu = self.incremental_gradient_diff(x,u)
        dHxx,dHxu,dHuu = self.incremental_hessian_diff(x,u)
        dtx = self.terminal_gradient_diff(x)
        dtxx = self.terminal_hessian_diff(x)
        xtol = max(1,np.linalg.norm(dgx),np.linalg.norm(dtx))*baseTol
        utol = max(1,np.linalg.norm(dgu))*baseTol
        xxtol = max(1,np.linalg.norm(dHxx),np.linalg.norm(dtxx))*baseTol
        xutol = max(1,np.linalg.norm(dHxu))*baseTol
        uutol = max(1,np.linalg.norm(dHuu))*baseTol
        res = True
        if np.linalg.norm(gx-dgx) > xtol:
            print("Derivatives of ObjectiveFunction",str(self),"incorrect in incremental grad x by",np.linalg.norm(gx-dgx),"diff norm",np.linalg.norm(dgx))
            res = False
        if np.linalg.norm(gu-dgu) > utol:
            print("Derivatives of ObjectiveFunction",str(self),"incorrect in incremental grad u by",np.linalg.norm(gu-dgu),"diff norm",np.linalg.norm(dgu))
            print("  Function returned",gu)
            print("  Finite differences got",dgu)
            res = False
        if np.linalg.norm(tx-dtx) > xtol:
            print("Derivatives of ObjectiveFunction",str(self),"incorrect in terminal grad by",np.linalg.norm(tx-dtx),"diff norm",np.linalg.norm(dtx))
            res = False
        if np.linalg.norm(Hxx-dHxx) > xxtol:
            print("Derivatives of ObjectiveFunction",str(self),"incorrect in incremental hess xx by",np.linalg.norm(Hxx-dHxx),"diff norm",np.linalg.norm(dHxx))
            res = False
        if np.linalg.norm(Hxu-dHxu) > xutol:
            print("Derivatives of ObjectiveFunction",str(self),"incorrect in incremental hess xu by",np.linalg.norm(Hxu-dHxu),"diff norm",np.linalg.norm(dHxu))
            res = False
        if np.linalg.norm(Huu-dHuu) > uutol:
            print("Derivatives of ObjectiveFunction",str(self),"incorrect in incremental hess uu by",np.linalg.norm(Huu-dHuu),"diff norm",np.linalg.norm(dHuu))
            print("  Function returned")
            print(Huu)
            print("  Finite differences got")
            print(dHuu)
            res = False
        if np.linalg.norm(txx-dtxx) > xxtol:
            print("Derivatives of ObjectiveFunction",str(self),"incorrect in terminal hess by",np.linalg.norm(txx-dtxx),"diff norm",np.linalg.norm(dtxx))
            print("  Function returned")
            print(txx)
            print("  Finite differences got")
            print(dtxx)
            res = False
        return res

    def __add__(self,rhs):
        return _AddObjectiveFunction(self,rhs)
    def __mul__(self,rhs):
        if isinstance(rhs,(int,float)):
            return _MulObjectiveFunction(self,rhs)
        return _ProductObjectiveFunction(self,rhs)
    def __rmul__(self,rhs):
        return _MulObjectiveFunction(self,rhs)


class QuadraticObjectiveFunction(ObjectiveFunction):
    """Penalizes state and control effort L(x,u,t) = 1/2 (x-goal)^T P (x-goal) + 1/2 u^T Q u,
    Phi(x) = 1/2 (x-goal)^T R (x-goal).
    """
    def __init__(self,P,Q,R,goal=None):
        self.P = P
        self.Q = Q
        self.R = R
        self.goal = goal
        if goal is None:
            self.goal = np.zeros(P.shape[0])
    def incremental(self,x,u):
        x = np.asarray(x)-self.goal
        u = np.asarray(u)
        return np.dot(x,np.dot(self.P,x))*0.5 + np.dot(u,np.dot(self.Q,u))*0.5    
    def incremental_gradient(self,x,u):
        x = np.asarray(x)-self.goal
        u = np.asarray(u)
        return  (np.dot(self.P,x),np.dot(self.Q,u))
    def incremental_hessian(self,x,u):
        return (self.P,np.zeros((len(x),len(u))),self.Q)
    def terminal(self,x):
        x = np.asarray(x)-self.goal
        return 0.5*np.dot(x,np.dot(self.R,x))
    def terminal_gradient(self,x):
        x = np.asarray(x)-self.goal
        return np.dot(self.R,x)
    def terminal_hessian(self,x):
        return self.R


class LambdaObjectiveFunction(ObjectiveFunction):
    """Adapts a function incremental(x,u) and/or terminal(x) to an
    ObjectiveFunction object.  Also uses automatically uses finite
    differences for gradients/hessians.
    """
    def __init__(self,incremental=None,terminal=None):
        self.fincremental = incremental
        self.fterminal = terminal
    def incremental(self,x,u=None):
        if self.fincremental: return self.fincremental(x,u)
        else: return 0.0
    def terminal(self,x):
        if self.fterminal: return self.fterminal(x)
        else: return 0.0
        
    def incremental_gradient(self,x,u):
        if self.fincremental:
            return self.incremental_gradient_diff(x,u)
        else:
            return ObjectiveFunction.incremental_gradient(self,x,u)
    def incremental_hessian(self,x,u):
        if self.fincremental:
            return self.incremental_hessian_diff(x,u)
        else:
            return ObjectiveFunction.incremental_hessian(self,x,u)
    def terminal_gradient(self,x):
        if self.fincremental:
            return self.terminal_gradient_diff(x)
        else:
            return ObjectiveFunction.terminal_gradient(self,x)
    def terminal_hessian(self,x):
        if self.fincremental:
            return self.terminal_hessian_diff(x)
        else:
            return ObjectiveFunction.terminal_hessian(self,x)


class PointwiseObjectiveFunction(ObjectiveFunction):
    """Converts a pointwise incremental function to work over longer edge
    integrators.  Given a function pointwise(x,u), produces the incremental
    cost by integrating over the interpolator's length with a fine timestep.
    """
    def __init__(self,space,pointwise,timestep=0.01):
        self.space = space
        self.fpointwise = pointwise
        self.timestep = timestep
    def incremental(self,x,u):
        e = self.space.interpolator(x,u)
        l = e.length()
        if l == 0: return 0
        t = 0
        c = 0
        while t < l:
            c += self.fpointwise(e.eval(t / l),u)
            t += self.timestep
        return c*self.timestep
    def incremental_gradient(self,x,u):
        return self.incremental_gradient_diff(x,u)
    def incremental_hessian(self,x,u):
        return self.incremental_hessian_diff(x,u)


        
class _AddObjectiveFunction(ObjectiveFunction):
    def __init__(self,*fs):
        self.fs = []
        for f in fs:
            if isinstance(f,_AddObjectiveFunction):
                self.fs += f.fs
            else:
                self.fs.append(f)
    def __str__(self):
        return " + ".join(str(f) for f in self.fs)
        
    def incremental(self,x,u=None):
        return sum(f.incremental(x,u) for f in self.fs)
    def terminal(self,x):
        return sum(f.terminal(x) for f in self.fs)
    
    def incremental_gradient(self,x,u):
        gs = [f.incremental_gradient(x,u) for f in self.fs]
        return sum(g[0] for g in gs),sum(g[1] for g in gs)
    def incremental_hessian(self,x,u):
        Hs = [f.incremental_hessian(x,u) for f in self.fs]
        return sum(H[0] for H in Hs),sum(H[1] for H in Hs),sum(H[2] for H in Hs)
    def terminal_gradient(self,x):
        return sum(f.terminal_gradient(x) for f in self.fs)
    def terminal_hessian(self,x):
        return sum(f.terminal_hessian(x) for f in self.fs)

class _MulObjectiveFunction(ObjectiveFunction):
    def __init__(self,f,scale):
        self.f = f
        self.scale = scale
    def __str__(self):
        return str(self.scale)+"*"+str(self.f)
    def incremental(self,x,u=None):
        return self.scale*self.f.incremental(x,u)
    def terminal(self,x):
        return self.scale*self.f.terminal(x)
    
    def incremental_gradient(self,x,u):
        fx,fu = self.f.incremental_gradient(x,u)
        return self.scale*fx,self.scale*fu
    def incremental_hessian(self,x,u):
        Hx,Hxu,Hu = self.f.incremental_hessian(x,u)
        return self.scale*Hx,self.scale*Hxu,self.scale*Hu
    def terminal_gradient(self,x):
        return self.scale*self.f.terminal_gradient(x)
    def terminal_hessian(self,x):
        return self.scale*self.f.terminal_hessian(x)

class _ProductObjectiveFunction(ObjectiveFunction):
    def __init__(self,f,g):
        self.f = f
        self.g = g
    def __str__(self):
        return str(self.f)+"*"+str(self.g)
    def incremental(self,x,u=None):
        return self.f.incremental(x,u)*self.g.incremental(x,u)
    def terminal(self,x):
        return self.f.terminal(x)*self.g.terminal(x)
    
    def incremental_gradient(self,x,u):
        f = self.f.incremental(x,u)
        fx,fu = self.f.incremental_gradient(x,u)
        g = self.g.incremental(x,u)
        gx,gu = self.g.incremental_gradient(x,u)
        return (f*gx + g*fx,f*gu + g*fu)
    def incremental_hessian(self,x,u):
        f = self.f.incremental(x,u)
        fx,fu = self.f.incremental_gradient(x,u)
        fxx,fxu,fuu = self.f.incremental_hessian(x,u)
        g = self.g.incremental(x,u)
        gx,gu = self.g.incremental_gradient(x,u)
        gxx,gxu,guu = self.g.incremental_hessian(x,u)
        fgx = np.outer(fx,gx)
        fgu = np.outer(fu,gu)
        return (f*gxx + fgx + fgx.T + g*fxx,f*gxu + np.outer(fx,gu) + np.outer(gx,fu), + g*fxu,f*guu + fgu + fgu.T + g*fuu)
    def terminal_gradient(self,x):
        return self.f.terminal(x)*self.g.terminal_gradient(x) + self.g.terminal(x)*self.f.terminal_gradient(x)
    def terminal_hessian(self,x):
        f = self.f.terminal(x)
        fx = self.f.terminal_gradient(x)
        fxx = self.f.terminal_hessian(x)
        g = self.g.terminal(x)
        gx = self.g.terminal_gradient(x)
        gxx = self.g.terminal_hessian(x)
        fgx = np.outer(fx,gx)
        return f*gxx + fgx + fgx.T + g*fxx
