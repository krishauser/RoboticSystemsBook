from .dynamics import Dynamics,IntegratorControlSpace
from .objective import ObjectiveFunction
#from scipy.interpolate import RegularGridInterpolator



class OptimalControlProblem:
    """A standardized class for an optimal control problem.

    Attributes:
        x0 (array): the initial state.
        dynamics (IntegratorControlSpace): a controlSpace with a fixed timestep
            dt.
        objective (ObjectiveFunction): the optimality criterion
        goal (None, array, or callable): either no terminal condition (None);
            a terminal state (array), or a function term(x) > 0 iff x is a
            terminal state.
        stateChecker (callable, optional): a function f(x) > 0 iff x is a
            valid state
        controlChecker (callable): a function f(x,u) > 0  stating whether u is
            a valid control at state x.
    """
    def __init__(self,x0,dynamics,objective,goal=None,stateChecker=None,controlChecker=None,controlSampler=None,dt=None):
        self.x0 = x0
        self.dynamics = dynamics
        if isinstance(dynamics,Dynamics):
            if dt is None:
                raise ValueError("If a Dynamics is passed as the `dynamics` object, timestep `dt` must also be provided")
            self.dynamics = IntegratorControlSpace(dynamics,dt)
        self.objective = objective
        self.goal = goal
        self.stateChecker = stateChecker
        self.controlChecker = controlChecker
        self.controlSampler = controlSampler
        self.dt = dt

    def isPointToPoint(self):
        return hasattr(self.goal,'__iter__')

    def stateValid(self,x):
        return self.stateChecker is None or self.stateChecker(x)

    def controlValid(self,x,u):
        return self.controlChecker is None or self.controlChecker(x,u)



class ControlSampler:
    def sample(self,state):
        """Returns a list of controls that should be used at the given state"""
        raise NotImplementedError()
        

class LookaheadPolicy:
    """Converts a value function into a 1-step lookahead policy."""
    def __init__(self,problem,valueFunction,goal=None):
        self.dynamics = problem.dynamics
        self.controlSampler = problem.controlSampler
        self.objective = problem.objective
        if goal is None:
            self.goal = problem.goal
        else:
            self.goal = goal
        self.valueFunction = valueFunction

    def __call__(self,x):
        bestcontrol = None
        bestcost = float('inf')
        us = self.controlSampler.sample(x)
        for u in us:
            xnext = self.dynamics.nextState(x,u)
            if self.dynamics.validState(xnext):
                cost = self.objective.incremental(x,u)
                v = self.valueFunction(xnext)
                #print "Value of going from",x,"control",u,"to",xnext,"is",cost,"+",v,"=",cost+v
                if v + cost < bestcost:
                    bestcost = v + cost
                    bestcontrol = u
        if self.goal is None or (callable(self.goal) and self.goal(x)):
            #check whether to terminate
            tcost = self.objective.terminalCost(x)
            if tcost <= bestcost or self.goalabsorbing:
                #print("Better to terminate, cost,",tcost)
                return None
        return bestcontrol



def rollout_policy(dynamics,x,control,dt,numSteps):
    """Returns a state trajectory and control trajectory of length
    numSteps+1 and numSteps, respectively.

    control can either be a callable policy (closed loop) or a list
    of length at least numSteps (open loop).

    A control of None means to terminate.
    """
    if isinstance(dynamics,Dynamics):
        if dt is None:
            raise ValueError("If a Dynamics is passed as the `dynamics` object, timestep `dt` must also be provided")
        dynamics = IntegratorControlSpace(dynamics,dt)
    xs = [x]
    us = []
    if callable(control):
        while len(xs) <= numSteps:
            u = control(xs[-1])
            if u is None: break
            us.append(u)
            xnext = dynamics.nextState(xs[-1],u)
            #print("After",xs[-1],"control",u,"for time",dt,"result is",xnext)
            xs.append(xnext)
    else:
        assert len(control) >= numSteps
        while len(xs) <= numSteps:
            u = control[len(xs)-1]
            if u is None: break
            us.append(u)
            xnext = dynamics.nextState(xs[-1],u)
            xs.append(xnext)
    return xs,us

