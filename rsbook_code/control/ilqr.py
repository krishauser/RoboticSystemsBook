from __future__ import print_function,division
from builtins import range
from six import iteritems
import math
import numpy as np
from .objective import ObjectiveFunction
from .dynamics import ControlSpace, Dynamics, IntegratorControlSpace


class iLQR:
    """An implementation of the iLQR trajectory optimization algorithm.
        
    Attributes:
        xref (array of size (T+1,n)): the optimized reference state trajectory
        uref (array of size (T,m)): the optimized reference control trajectory
        gains (pair of arrays): a pair (K,k) of arrays so that for each time step, 
            the optimized control is given by
                
                u(x,t) ~= K[t]*(x - xref[t]) + k[t] + uref[t]
            
            K has dimension (T,m,n) and k has dimension (T,m)
            
        value (triple of T+1-lists): a triple (V,Vx,Vxx) of arrays so that for each
            time step, the quadratic expansion of the value function is given by:
            
                V(x,t) ~= 1/2 dx^T Vxx[t] dx + dx^T Vx[t] + V[t]
            
            with  dx = x-xref[t]. 
        
        costGradients (array of size T,m): the gradients of total cost w.r.t.
            controls.
    """
    def __init__(self,dynamics,objective,dt=None,verbose=0):
        assert isinstance(objective,ObjectiveFunction)
        self.dynamics= dynamics
        if isinstance(self.dynamics,Dynamics):
            if dt is None:
                raise TypeError("If `dynamics` is a Dynamics object, then `dt` must be provided")
            assert dt > 0
            self.dynamics = IntegratorControlSpace(dynamics,dt)
        else:
            if not isinstance(self.dynamics,ControlSpace):
                raise TypeError("`dynamics` must be a Dynamics or ControlSpace object")
        self.objective = objective
        self.xref = None
        self.uref = None
        self.gains = None
        self.value = None
        self.costGradients = None
        self.verbose = verbose

    def run(self,x,u,maxIters,xtol=1e-7,gtol=1e-7,ftol=1e-7,damping=1e-5):
        if len(u)==0:
            raise ValueError("Cannot optimize with no controls")
        if not hasattr(x[0],'__iter__'):
            #assume its a single state
            x0 = x
            x = [x0]
            for ut in u:
                x.append(self.dynamics.nextState(x[-1],ut))
        assert len(x) == len(u)+1
        self.xref = np.array([xt for xt in x])
        self.uref = np.array([ut for ut in u])
        T = len(u)
        m = len(u[0])
        n = len(x[0])
        self.gains = (np.zeros((T,m,n)),np.zeros((T,m)))
        self.value = (np.zeros((T+1)),np.zeros((T+1,n)),np.zeros((T+1,n,n)))
        self.costGradients = np.zeros((T,m))
        if not self.dynamics.checkDerivatives(x[0],u[0]) or not self.dynamics.checkDerivatives(x[-2],u[-1]):
            input("Press enter to continue >")
        if not self.objective.checkDerivatives(x[0],u[0]) or not self.objective.checkDerivatives(x[-2],u[-1]):
            input("Press enter to continue >")
        
        #first cost backup
        costTraj = self.value[0]
        costTraj[:] = self.evalCosts(self.xref,self.uref)
        J0 = costTraj[0]
        if not math.isfinite(J0):
            raise ValueError("Need to provide a feasible path as input?")
        if self.verbose:
            print("iLQR: beginning from cost",J0)
        if self.verbose >= 2:
            print("  INITIAL TRAJECTORY")
            for (a,b) in zip(x,u):
                print("    ",a,b)
            print("    ",x[-1])
            print("  COST TRAJECTORY",costTraj)
            print("  OBJECTIVE TYPE",self.objective)
        
        for iter in range(maxIters):
            alpha = 1.0
            self.backward()
            g = self.costGradients
            gnorm = np.linalg.norm(g)
            if gnorm < gtol:
                return True,'Convergence to stationary point'
            knorm = np.linalg.norm(self.gains[1])
            if self.verbose:
                print("iLQR: Norm of nominal step size: %.3f, gradient norm %.3f"%(knorm,gnorm))
            if np.dot(g.flatten(),self.gains[1].flatten()) > 0:
                if self.verbose:
                    print("WARNING: LQR step has direction reverse from gradient")
                self.gains[1][:] = -g
                knorm = gnorm
            #test gradient descent
            #self.gains[1][:] = -g
            lineSearchIters = 0
            alpha0 = alpha
            while alpha*knorm > xtol and lineSearchIters < maxIters:
                lineSearchIters += 1
                xu = self.forward(alpha)
                if xu is None:
                    #failure, shrink step size
                    alpha *= 0.5
                    continue
                x,u = xu
                Ja = self.evalCosts(x,u,cbranch=J0) 
                if Ja[0] < J0 and abs(Ja[0]-self.objective.cost(x,u)) > 1e-4:
                    print("Uh... difference in costs?",Ja[0],"vs",self.objective.cost(x,u))
                    input("Press enter to continue >")
                if Ja[0] < J0:
                    #accept step
                    self.xref = x
                    self.uref = u
                    self.value[0][:] = Ja
                    if self.verbose >= 2:
                        print("iLQR: Step length %.3g reduced cost to %.3f < %.3f"%(alpha,Ja[0],J0))
                        #print("   Endpoints",x[0],x[1])
                        #print("   Controls",u)
                    if alpha == alpha0:
                        #succeeded on first step, increase default step size
                        alpha *= 2.5
                        if alpha > 1.0:
                            alpha = 1.0
                    break
                else:
                    #failure, shrink step size
                    #print("Rejected step to cost",Ja[0])
                    alpha *= 0.5
                    
            self.value[0][:] = Ja
            J0 = Ja[0]
            
            if alpha*knorm <= xtol or lineSearchIters == maxIters:
                if self.verbose:
                    print("iLQR: Inner iterations stalled at",lineSearchIters,"LS iters, step size",alpha,", gradient norm",knorm,"< tolerance",xtol)
                    print("   COST",self.objective.cost(self.xref,self.uref))
                return True,'Convergence on x'
            self.value[0][:] = Ja
            J0 = Ja[0]
            if self.verbose:
                print("   COST",J0)

        if self.verbose:
            print("iLQR: Terminated with max iterations reached")
        return False,'Max iters reached'
    
    def evalCosts(self,x,u,cbranch=float('inf')):
        """Returns vector of value function evaluated along trajectory."""
        T = len(u)
        assert T+1 == len(x)
        costs = np.empty(len(x))
        costs[-1] = self.objective.terminal(x[T])
        if costs[-1] > cbranch:
            costs[0] = costs[-1]
            return costs
        for i in range(T)[::-1]:
            xt = x[i]
            ut = u[i]
            c = self.objective.incremental(xt,ut)
            costs[i] = costs[i+1] + c
            if costs[i] > cbranch:
                costs[0] = costs[i]
                return costs
        return costs

    def backward(self,damping=1e-3):
        """Computes the LQR backup centered around self.xref,self.uref.
        
        Will fill out self.gains, self.costGradients, and the 2nd and 3rd
        elements of self.value
        """
        T = len(self.gains[0])
        Vx = self.objective.terminal_gradient(self.xref[T])
        Vxx = self.objective.terminal_hessian(self.xref[T])
        if np.linalg.norm(Vxx-Vxx.T) > 1e-3:
            print("ERROR IN TERMINAL HESSIAN",self.xref[T])
            print(Vxx)
            raise ValueError()
        self.value[1][-1] = Vx
        self.value[2][-1] = Vxx
        print("iLQR BACKWARDS PASS")
        #print("   Terminal cost",self.objective.terminal(self.xref[T]))
        #print("   Terminal grad",Vx)
        #print("   Terminal Hessian",Vxx)
        for i in range(T)[::-1]:
            #print("timestep",i)
            xt,ut = self.xref[i],self.uref[i]
            fx,fu = self.dynamics.nextState_jacobian(xt,ut)
            cx,cu = self.objective.incremental_gradient(xt,ut)
            cxx,cxu,cuu = self.objective.incremental_hessian(xt,ut)
            #print("  Next state jacobian x",fx)
            #print("  Next state jacobian u",fu)
            Qxx = fx.T.dot(Vxx.dot(fx))+cxx
            Quu = fu.T.dot(Vxx.dot(fu))+cuu
            Qxu = fx.T.dot(Vxx.dot(fu))+cxu
            Vxc = Vx
            Qx = cx + fx.T.dot(Vxc)
            Qu = cu + fu.T.dot(Vxc)
            if damping > 0:
                Quu = (Quu + Quu.T)*0.5
                Quu_evals, Quu_evecs = np.linalg.eig(Quu)
                Quu_evals[Quu_evals < 0] = 0.0
                Quu_evals += damping
                QuuInv = np.dot(Quu_evecs,np.dot(np.diag(1.0/Quu_evals),Quu_evecs.T))
            else:
                QuuInv = np.linalg.pinv(Quu)
            K = -QuuInv.dot(Qxu.T)
            k = -QuuInv.dot(Qu)
            temp = Qxu.dot(K)
            Vxx = Qxx + temp + temp.T + K.T.dot(Quu.dot(K))
            Vx = Qx + Qxu.dot(k) + K.T.dot(Qu+Quu.dot(k))
            #print("   Vf grad",Vx)
            #print("   Vf Hessian",Vxx)
            self.gains[0][i] = K
            self.gains[1][i] = k
            self.value[1][i] = Vx
            self.value[2][i] = Vxx
            self.costGradients[i] = Qu

    def forward(self,alpha=1.0):
        """Computes the iLQR forward pass, assuming the gain matrices have been computed"""
        x = np.empty(self.xref.shape)
        u = np.empty(self.uref.shape)
        x[0] = self.xref[0]
        u[0] = self.uref[0]
        K,k = self.gains
        for i in range(self.uref.shape[0]):
            if i == 0:
                du = k[0]
            else:
                du = k[i] + K[i].dot(x[i]-self.xref[i])
            u[i] = self.uref[i] + alpha*du
            x[i+1] = self.dynamics.nextState(x[i],u[i])
        return (x,u)

