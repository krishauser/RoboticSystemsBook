from __future__ import print_function,division
import itertools
import numpy as np
from .optimalcontrol import OptimalControlProblem,LookaheadPolicy,rollout_policy
from ..utilities.astar_fibheap import AStar

class RobustRegularGridInterpolator:
    """Like RegularGridInterpolator but handles interpolation with Inf and NaN
    values better, and clamps to the bounds rather than raising errors.

    Linear interpolation is handled as follows:
    The bottommost vertex of a cell is retrieved by getIndex.
    The 2^d vertices of the cell are examined, and a modified d-linear interpolation 
    is performed.  If all values in the cell's neighbors are finite, the result
    is the same as standard d-linear interpolation.  Any non-finite values
    are ignored.

    bounds_clamp=True indicates that out-of-grid points are clamped to the
    boundary.
    """
    def __init__(self,divs,values,method='linear',bounds_clamp=True):
        self.divs = divs
        self.values = values
        self.method = method
        self.bounds_clamp = bounds_clamp
        if method == 'linear':
            self.evalMethod = self.evalLinear
        else:
            self.evalMethod = self.evalNearest
    def __call__(self,points):
        if hasattr(points[0],'__iter__'):
            return np.array([self.__call__(p) for p in points])
        return self.evalMethod(points)
    def getCell(self,point):
        ind = []
        if self.bounds_clamp:
            for (d,v) in zip(self.divs,point):
                ind.append(max(0,min(np.searchsorted(d,v)-1,len(d)-1)))
        else:
            for (d,v) in zip(self.divs,point):
                ind.append(np.searchsorted(d,v)-1)
        return ind
    def evalNearest(self,point):
        print("EVALUATING NEAREST???")
        if len(point) != len(self.divs):
            raise ValueError("Invalid size of point")
        index = self.getCell(point)
        for dim,i,d,v in zip(range(len(index)),index,self.divs,point):
            if i >= 0 and i +1 < len(d):
                if v-d[i] > (d[i+1]-d[i]):
                    index[dim] += 1
        return self.values[tuple(index)]
    def evalLinear(self,point):
        if len(point) != len(self.divs):
            raise ValueError("Invalid size of point")
        index = tuple(self.getCell(point))
        vcenter = self.values[index]
        #if not np.isfinite(vcenter):
        #    return vcenter
        verts = []
        uinterp = []
        for i,d,v in zip(index,self.divs,point):
            if i < 0:
                verts.append((0,0))
                uinterp.append(0)
            elif i+1 == len(d):
                verts.append((len(d)-1,len(d)-1))
                uinterp.append(0)
            else:
                verts.append((i,i+1))
                uinterp.append((v-d[i])/(d[i+1]-d[i]))
        #print("Cell range",verts,"params",uinterp)
        cellslice = tuple([slice(v[0],v[1]+1) for v in verts])
        vcell = self.values[cellslice]
        #print(vcell)
        try:
            while len(vcell.shape) > 0:
                #print("Cell",vcell)
                if vcell.shape[-1] == 1:
                    vcell = vcell[...,0]
                else:
                    u = uinterp[len(vcell.shape)-1]
                    vcelldim0 = vcell[...,0]
                    vcelldim1 = vcell[...,1]
                    invalid0 = ~np.isfinite(vcelldim0)
                    invalid1 = ~np.isfinite(vcelldim1)
                    np.copyto(vcelldim0,vcelldim1,where=invalid0)
                    np.copyto(vcelldim1,vcelldim0,where=invalid1)
                    vcell = (1-u)*vcelldim0 + u*vcelldim1
        except RuntimeWarning:
            pass
        return vcell

class HJBSolver:
    """Solves an optimal control problem with the Hamilton-Jacobi-Bellman equation
    and a value iteration method.  Note that this is a backward iteration.

    The value function is the cost-to-go to the goal.

    If there is a single goal state, or a very small goal set, the solution may be
    inexact.

    The goalabsorbing attribute tells this whether to assume the goal set is absorbing.

    Parameters:
        maxSubSteps (int): number of steps to take to exit a cell.  Default 20
    """
    def __init__(self,problem,
                stateBounds,stateDivs,
                goalabsorbing='auto'):
        """
        Args:
            problem (OptimalControlProblem): contains the dynamics, objective, and goal.
                Also, problem.controlSampler must be defined
            stateBounds (list of pairs): the range [(x1min,x1max),...,(xnmin,xnmax)]
                over which the grid is defined.
            stateDivs (list of ints): the number of divisions of the state space.
            goalabsorbing (bool, optional).  Sets whether the goal set is absorbing or
                not.  If 'auto', this is set to False if problem.goal = None (otherwise 
                everything would be absorbing).  If problem.goal is a point or a function,
                it is assumed absorbing.
        """
        self.problem = problem
        self.dynamics = problem.dynamics
        self.controlSampler = problem.controlSampler
        assert problem.controlSampler is not None
        self.objective = problem.objective
        self.maxSubSteps = 20
        assert len(stateBounds) == len(stateDivs)
        numCells = np.prod(stateDivs)
        if numCells > 10000000:
            raise RuntimeError("More than 100 million cells in desired discretization "+str(stateDivs))
        assert all(d >= 1 for d in stateDivs),"Need at least two grid points in each dimension"
        self.stateBounds = stateBounds
        self.stateMin,self.stateMax = zip(*stateBounds)
        self.stateMin = np.array(self.stateMin)
        self.stateMax = np.array(self.stateMax)
        self.stateDivs = stateDivs
        self.gridResolution = np.divide(self.stateMax-self.stateMin,self.stateDivs)
        self.gridResolutionInv = np.divide(1.0,self.gridResolution)
        self.value = np.full(stateDivs,float('inf'))
        self.value.fill(float('inf'))
        self.policy = np.full(stateDivs,None,dtype=object)
        self.terminal = np.full(stateDivs,False,dtype=bool)
        if problem.goal is not None and hasattr(problem.goal,'__iter__'):
            if hasattr(problem.goal[0],'__iter__'):
                goals = problem.goal
            else:
                goals = [problem.goal]
                
            #it's one or more states, mark all other states as having infinite cost
            for g in goals:
                index = self.stateToCell(g)
                assert self.validCell(index),"Goal "+str(g)+" is not in the grid"
                self.value[index] = self.objective.terminal(problem.goal)
                self.terminal[index] = True
            self.goalTest = None
            if goalabsorbing == 'auto':
                goalabsorbing = True
        else:
            if not callable(problem.goal):
                raise TypeError("Goal must be a state, list of states, or function")
            #it's a goal test or None, evaluate all terminal state costs
            self.goalTest = problem.goal
            for index in np.ndindex(*self.value.shape):
                x = self.cellToCenterState(index)
                if problem.goal is None or problem.goal(x):
                    self.terminal[index] = True
                    self.value[index] = self.objective.terminal(x)
            if goalabsorbing == 'auto':
                goalabsorbing = (problem.goal is not None)
        self.goalabsorbing = goalabsorbing

        dims = [np.linspace(a+c*0.5, b-c*0.5, d) for (a,b,c,d) in zip(self.stateMin,self.stateMax,self.gridResolution,self.stateDivs)]
        self.valueInterpolator = RobustRegularGridInterpolator(dims,self.value)

        #stores a list of (nextstate,control,cost) tuples
        self.transitionMatrix = None

    def cellToCenterState(self,index):
        return self.stateMin + np.multiply(np.asarray(index)+0.5,self.gridResolution)

    def stateToCell(self,state):
        return tuple(np.floor(np.multiply(np.asarray(state)-self.stateMin,self.gridResolutionInv)).astype(int))

    def validCell(self,cell):
        return all(c >= 0 and c < d for (c,d) in zip(cell,self.stateDivs))

    def interpolateValue(self,state):
        return self.valueInterpolator(state)

    def valueIteration(self,iters=1,maxDiffThreshold=0):
        """Runs iters iterations of value iteration.  Stops if the value function
        differs by less than maxDiffThreshold.
        """
        if self.transitionMatrix is None:
            #set up the transition matrix
            self.transitionMatrix = np.full(self.stateDivs,None,dtype=object)
            numtransitions = 0
            numsubsteps = 0
            for index in np.ndindex(*self.value.shape):
                x = self.cellToCenterState(index)
                options = []
                self.transitionMatrix[index] = options
                if self.terminal[index]:
                    options.append((None,None,0))
                    if self.goalabsorbing:
                        #don't consider subsequent actions
                        continue
                us = self.controlSampler.sample(x)
                for u in us:
                    #advance up to maxSubSteps steps to get out of this cell
                    xtemp = x
                    cost = 0
                    numtransitions += 1
                    for i in range(1,self.maxSubSteps+1):
                        numsubsteps += 1
                        if not self.problem.controlValid(xtemp,u):
                            xtemp = None
                            break 
                        xnext = self.dynamics.nextState(xtemp,u)
                        indexnext = self.stateToCell(xnext)
                        if not self.problem.stateValid(xnext) or not self.validCell(indexnext):
                            xtemp = None
                            break
                        cost += self.objective.incremental(xtemp,u)
                        xtemp = xnext
                        if indexnext != index:
                            break
                    if xtemp is not None:
                        options.append((xtemp,u,cost))
            if numsubsteps > numtransitions*2:
                print("Building transition matrix: took an average of",float(numsubsteps)/float(numtransitions),"substeps, you may consider changing the step size")
            #transition matrix is built
        for it in range(iters):
            maxdelta = 0
            for index in np.ndindex(*self.value.shape):
                bestcontrol = None
                bestcost = float('inf')
                options = self.transitionMatrix[index]
                for (xnext,u,cost) in options:
                    if xnext is None:
                        assert bestcontrol is None,"goal termination must be first option"
                        bestcost = self.value[index]
                        bestcontrol = None
                    else:
                        if not np.isfinite(self.value[self.stateToCell(xnext)]):
                            continue
                        vnext = self.valueInterpolator(xnext)
                        if vnext + cost < bestcost:
                            bestcost = vnext + cost
                            bestcontrol = u

                if bestcontrol is not None:
                    maxdelta = max(maxdelta,abs(self.value[index]-bestcost))

                self.policy[index] = bestcontrol
                #update in place, or copy to a different array?
                self.value[index] = bestcost
            if maxdelta < maxDiffThreshold:
                break

    def getPolicy(self,x,lookahead=False):
        """Returns the computed action at the state x.  Returns None to terminate.

        If lookahead = True, this actually looks ahead by time dt to find the best
        action.
        """
        if not lookahead:
            return self.policy[self.stateToCell(x)]
        return LookaheadPolicy(self.problem,self.valueInterpolator)(x)


class OptimalControlTreeSolver(AStar):
    """Searches by keeping around the tree of optimal states from the start.
    Operates by iterating on the cost-to-come from the start, not the cost-to-go.  Also,
    the policy is determined from the goal backward.

    This is a bit more accurate for developing a model predictive control,
    for example.

    It can also be used to do a backwards search.  The dynamics function must
    accept the reverse dynamics, and terminalAsStartCost=True uses the objective
    function's terminal() method to determine the start cost.

    Reverse dynamics means that the dynamics.nextState(x,u,dt) function should
    be able to accept a negative dt, which should return the state xprev such
    that integrating the forward dynamics over duration -dt with control u
    should end up in x.

    Parameters:
        maxVisitedPerCell (int): # of times a cell can be visited (default 1)
        maxSubSteps (int): # of steps to take to attempt to exit a cell (default 20)
    
    """
    def __init__(self,problem,
                stateBounds,stateDivs,
                start=None,goalabsorbing='auto',
                terminalAsStartCost=False):
        """
        Args:
            problem (OptimalControlProblem)
            stateBounds (list of pairs): the range [(x1min,x1max),...,(xnmin,xnmax)]
                over which the grid is defined.
            stateDivs (list of ints): the number of divisions of the state space.
            start (state, list of states, function, or None): the start state(s)
                or a start test. If None, uses x0 in problem.
            goalabsorbing (bool, optional).  Sets whether the goal set is absorbing or
                not.  If 'auto', this is set to False if goal = None (otherwise everything
                would be absorbing).  If goal is a point or a function, it is assumed
                absorbing.
        """
        AStar.__init__(self)
        self.maxVisitedPerCell = 1
        self.maxSubSteps = 20
        self.problem = problem
        self.dynamics = problem.dynamics
        self.controlSampler = problem.controlSampler
        assert problem.controlSampler is not None
        self.objective = problem.objective
        self.terminalAsStartCost = terminalAsStartCost
        assert len(stateBounds) == len(stateDivs)
        numCells = np.prod(stateDivs)
        if numCells > 10000000:
            raise RuntimeError("More than 100 million cells in desired discretization "+str(stateDivs))
        assert all(d >= 1 for d in stateDivs),"Need at least two grid points in each dimension"
        self.stateBounds = stateBounds
        self.stateMin,self.stateMax = zip(*stateBounds)
        self.stateMin = np.array(self.stateMin)
        self.stateMax = np.array(self.stateMax)
        self.stateDivs = stateDivs
        self.gridResolution = np.divide(self.stateMax-self.stateMin,self.stateDivs)
        self.gridResolutionInv = np.divide(1.0,self.gridResolution)
        self.visited = np.full(stateDivs,None,dtype=object)
        
        self.start = start
        if start is None:
            self.start = problem.x0
        if callable(start):
            #multiple start states, make a virtual start state
            self.start = []
            self.startStates = []
            for index in np.ndindex(self.visited.shape):
                x = self.cellToCenterState(index)
                if start(x):
                    self.startStates.append(x)
            if len(self.startStates) == 0:
                raise ValueError("No start cells in grid")
            if terminalAsStartCost:
                self.startCosts = [problem.objective.terminal(x) for x in self.startStates]
            else:
                self.startCosts = [0.0]*len(self.startStates)
        elif hasattr(start,'__iter__') and hasattr(start[0],'__iter__'):
            #multiple start states
            self.start = []
            self.startStates = start
            if terminalAsStartCost:
                self.startCosts = [problem.objective.terminal(x) for x in self.startStates]
            else:
                self.startCosts = [0.0]*len(self.startStates)

        if problem.goal is not None and hasattr(problem.goal,'__iter__'):
            #it's a single state, mark all other states as having infinite cost
            if hasattr(problem.goal[0],'__iter__'):
                goals = problem.goal
            else:
                goals = [problem.goal]
            self.goalCells = set([self.stateToCell(g) for g in goals])
            self.goalTest = lambda x: self.stateToCell(x) in self.goalCells
            if goalabsorbing == 'auto':
                goalabsorbing = True
        else:
            #it's a goal test or None, evaluate all terminal state costs
            self.goalTest = problem.goal
            if goalabsorbing == 'auto':
                goalabsorbing = (problem.goal is not None)
        self.goalabsorbing = goalabsorbing

        self.set_start(self.start)

    def cellToCenterState(self,index):
        return self.stateMin + np.multiply(np.asarray(index)+0.5,self.gridResolution)

    def stateToCell(self,state):
        return tuple(np.floor(np.multiply(np.asarray(state)-self.stateMin,self.gridResolutionInv)).astype(int))

    def validCell(self,cell):
        return all(c >= 0 and c < d for (c,d) in zip(cell,self.stateDivs))

    def clear_visited(self):
        self.visited.fill(None)

    def visit(self, state, node):
        try:
            cell = self.stateToCell(state)
        except Exception:
            if state is None or len(state)==0: #virtual start or goal state
                return
            else:
                raise
        nodes = self.visited[cell]
        if nodes is None:
            self.visited[cell] = [node]
        elif len(nodes) < self.maxVisitedPerCell:
            nodes.append(node)
        else:
            #full cell?
            imax = max((n.g,i) for i,n in enumerate(nodes))[1]
            nodes[imax] = node


    def visited_state_node(self, state):
        try:
            cell = self.stateToCell(state)
        except Exception:
            if state is None or len(state)==0: #virtual start or goal state
                return
            else:
                raise
        nodes = self.visited[cell]
        if nodes is None:
            return None
        elif len(nodes) < self.maxVisitedPerCell:
            for n in nodes:
                if all(n.state==state):
                    return n
            return None
        else:
            return min(nodes,key=lambda n:n.g)

    def costToCome(self):
        """Returns the grid of costs taken to optimally reach each node.
        """
        ctc = np.full(self.visited.shape,float('inf'))
        for index in np.ndindex(self.visited.shape):
            nodes = self.visited[index]
            if nodes is not None:
                ctc[index] = min(n.g for n in nodes)
        return ctc

    def costToComeInterpolator(self):
        """Returns an bilinear interpolator of costs taken to optimally reach each node.
        """
        dims = [np.linspace(a+c*0.5, b-c*0.5, d) for (a,b,c,d) in zip(self.stateMin,self.stateMax,self.gridResolution,self.stateDivs)]
        return RobustRegularGridInterpolator(dims,self.costToCome())

    def getCostToCome(self,x):
        """Returns the costs taken to optimally reach this state.
        """
        index = self.stateToCell(x)
        nodes = self.visited[index]
        if nodes is not None:
            return min(n.g for n in nodes)
        return float('inf')

    def reversePolicy(self):
        """Returns the grid of actions that are taken to optimally reach
        each node.
        """
        rp = np.full(self.visited.shape,None,dtype=object)
        for index in np.ndindex(self.visited.shape):
            nodes = self.visited[index]
            if nodes is not None:
                nbest = min((n.g,i) for i,n in enumerate(nodes))[1]
                rp[index] = nodes[nbest].parentedge
                if rp[index] is not None:
                    rp[index] = rp[index][1]
        return rp

    def getReversePolicy(self,x):
        """Returns the computed action at the state x.  Returns None to terminate.
        """
        index = self.stateToCell(x)
        nodes = self.visited[index]
        if nodes is not None:
            nbest = min((n.g,i) for i,n in enumerate(nodes))[1]
            rp = nodes[nbest].parentedge
            if rp is not None:
                return rp[1]
        return None

    #AStar override
    def is_goal(self, state):
        return state is None
        
    def successors(self, state, maxCost=float('inf')):
        children = []
        costs = []
        actions = []

        if len(state) == 0: # virtual start state
            children = self.startStates
            costs = self.startCosts
            actions = [None]*len(self.startStates)
            return children,costs,actions

        if self.goalTest is None or self.goalTest(state):
            children.append(None)
            if self.terminalAsStartCost:
                costs.append(0)
            else:
                costs.append(self.objective.terminal(state))
            actions.append(None)
            if self.goalabsorbing:
                #no possible extra actions
                return (children,costs,actions)

        us = self.controlSampler.sample(state)
        stateCell = self.stateToCell(state)
        #uniqueCells = dict()
        #uniqueCells[stateCell] = state
        for u in us:
            x = state
            cost = 0
            for steps in range(1,self.maxSubSteps+1):
                if not self.problem.controlValid(x,u):
                    xnext = None
                    break            
                xnext = self.dynamics.nextState(x,u)
                cost += self.objective.incremental(x,u)
                cell = tuple(self.stateToCell(xnext))
                if not self.problem.stateValid(xnext) or not self.validCell(cell):
                    xnext = None
                    break
                if cell != stateCell: #left the initial cell, add it
                    #if cell not in uniqueCells:
                    #    uniqueCells[cell] = xnext
                    break
                x = xnext
            if xnext is not None:
                assert cost > 0
                children.append(xnext)
                costs.append(cost)
                actions.append((steps,u))
        #if len(uniqueCells)==1 and len(children) > 0:
        #    print("Problem setting grid resolution or step size: all children of node",state,"are in the same cell?")
        return children,costs,actions



class GridCostFunctionDisplay:
    """Helper for displaying HJBSolver or OptimalControlTreeSolver results."""
    def __init__(self,gridSolver,cost,policy=None,policyDims=None,figsize=(8,6),**kwargs):
        import matplotlib.pyplot as plt
        self.gridSolver = gridSolver
        self.slices = [np.linspace(a, b, d) for (a,b,d) in zip(gridSolver.stateMin,gridSolver.stateMax,gridSolver.stateDivs)]
        self.referenceCell = [0]*len(gridSolver.stateDivs)
        self.xindex = 0
        self.yindex = 1
        self.cbarcost = None
        self.cbarpolicy = None
        if policy is not None:
            self.fig,(self.axcost,self.axpolicy) = plt.subplots(1,2,figsize=figsize,**kwargs)
        else:
            self.fig = plt.figure(figsize=figsize,**kwargs)
            self.axcost = self.fig.gca()
            self.axpolicy = None
        self.cost = cost
        self.policy = policy
        self.policyDims = policyDims
        self.policyIndex = None
        if policyDims is not None:
            self.policyIndex = 0
        self.refresh(cost,policy)

    def show(self):
        """Call this to show on matplotlib / IPython"""
        import matplotlib.pyplot as plt
        plt.show()
        try:
            import ipywidgets as widgets
            from ipywidgets import interact
            from IPython.display import display
            is_interactive = True
        except ImportError as e:
            print("GridCostFunctionDisplay: can't show interactive dimension selectors")
            is_interactive = False
        if is_interactive:
            if len(self.cost.shape) > 2:
                def setx(xindex):
                    self.xindex = xindex
                    self.refresh(self.cost,self.policy)
                def sety(yindex):
                    self.yindex = yindex
                    self.refresh(self.cost,self.policy)
                def setReferenceCell(index,dim):
                    self.referenceCell[dim] = index
                    self.refresh(self.cost,self.policy)
                interact(setx, xindex=widgets.IntSlider(min=0,max=len(self.slices)-1,value=self.xindex))
                interact(sety, yindex=widgets.IntSlider(min=0,max=len(self.slices)-1,value=self.yindex))
                for i in range(len(self.referenceCell)):
                    w = widgets.IntSlider(min=0,max=self.cost.shape[i]-1,value=self.referenceCell[i],description="Ref cell "+str(i))
                    w.observe(lambda change,dim=i:setReferenceCell(change['new'],dim), names='value')
                    display(w)
            if self.policyDims is not None and self.policyDims > 1:
                def setPolicyIndex(index):
                    self.policyIndex = index
                    self.refresh(self.cost,self.policy)
                interact(setPolicyIndex, index=widgets.IntSlider(min=0,max=self.policyDims-1,value=self.policyIndex,description="Policy index"))

    def refresh(self,cost,policy=None):
        import matplotlib.pyplot as plt
        self.cost = cost
        self.policy = policy
        if self.cbarcost is not None: self.cbarcost.remove()
        if self.cbarpolicy is not None: self.cbarpolicy.remove()
        self.axcost.clear()
        self.axpolicy.clear()
        x0 = self.slices[self.xindex][0] - 0.5*(self.slices[self.xindex][1]-self.slices[self.xindex][0])
        xn = self.slices[self.xindex][-1] - 0.5*(self.slices[self.xindex][-2]-self.slices[self.xindex][-1])
        xmesh = [x0] + ((self.slices[self.xindex][:-1]+self.slices[self.xindex][1:])*0.5).tolist() + [xn]
        y0 = self.slices[self.yindex][0] - 0.5*(self.slices[self.yindex][1]-self.slices[self.yindex][0])
        yn = self.slices[self.yindex][-1] - 0.5*(self.slices[self.yindex][-2]-self.slices[self.yindex][-1])
        ymesh = [y0] + ((self.slices[self.yindex][:-1]+self.slices[self.yindex][1:])*0.5).tolist() + [yn]
        #X, Y = np.meshgrid(self.slices[self.xindex],self.slices[self.yindex])
        X, Y = np.meshgrid(xmesh,ymesh)
        sliceindex = None    
        if len(cost.shape) == 2:
            costplot = self.axcost.pcolormesh(X, Y, cost.T)
        else:
            #extract out the right slice
            sliceindex = [v for v in self.referenceCell]
            sliceindex[self.xindex] = slice(0,cost.shape[self.xindex])
            sliceindex[self.yindex] = slice(0,cost.shape[self.yindex])
            sliceindex = tuple(sliceindex)
            #print("Reading slice",sliceindex)
            costplot = self.axcost.pcolormesh(X, Y, cost[sliceindex].T)
        self.cbarcost = plt.colorbar(costplot,ax=self.axcost)
        if policy is not None:
            assert self.axpolicy is not None,"Can't add a policy after initialization without a policy"
            if sliceindex is not None:
                policy = policy[sliceindex]
            if self.policyIndex is None:
                policyplot = self.axpolicy.pcolormesh(X, Y, policy.astype(float).T)
            else:
                policySlice = np.full(policy.shape,float('inf'))
                for index in np.ndindex(*policy.shape):
                    if policy[index] is not None:
                        policySlice[index] = policy[index][self.policyIndex]
                policyplot = self.axpolicy.pcolormesh(X, Y, policySlice.T)
            self.cbarpolicy = plt.colorbar(policyplot,ax=self.axpolicy)
        #ax1.imshow(bwtree.costToCome().T)
        #ax2.imshow(bwtree.reversePolicy().astype(float).T)
    
    def plotTrajectory(self,xs,onplot='cost',**kwargs):
        """Plots the projection of a trajectory on the current axes"""
        #for x,u in zip(xs[1:],us):
        #    print(x,u)
        xs = np.asarray(xs)
        ax = self.axcost if onplot == 'cost' else self.axpolicy
        ax.plot(xs[:,self.xindex],xs[:,self.yindex],**kwargs)

    def plotRollout(self,x0,policy,numSteps,dt=None,onplot='cost',**kwargs):
        """Plots the rollout of a trajectory"""
        if dt is None:
            dt = self.grid.dt
        xs,us = rollout_policy(self.gridSolver.dynamics,x0,policy,dt,numSteps)
        self.plotTrajectory(xs,onplot,**kwargs)

    def plotFlow(self,policy,onplot='cost',**kwargs):
        """Plots a stream plot for the given policy"""
        X, Y = np.meshgrid(self.slices[self.xindex],self.slices[self.yindex])
        U, V = np.zeros(X.shape),np.zeros(Y.shape)
        for (i,j),index in self.shownIndices():
            x = self.gridSolver.cellToCenterState(index)
            u = policy(x)
            if u is not None:
                dx = self.gridSolver.dynamics.dynamics.derivative(x,u)
                U[j,i] = dx[self.xindex]
                V[j,i] = dx[self.yindex]
        ax = self.axcost if onplot == 'cost' else self.axpolicy
        ax.streamplot(X, Y, U, V, **kwargs)

    def shownIndices(self):
        """returns an iterator over displayed grid cells.  Iterated value is pairs ((i,j),index)"""
        index = [v for v in self.referenceCell]
        for i in range(self.cost.shape[self.xindex]):
            for j in range(self.cost.shape[self.yindex]):
                index[self.xindex] = i
                index[self.yindex] = j
                yield ((i,j),tuple(index))
        return

    def plotGraph(self,onplot='cost',**kwargs):
        """Plots the grid of states and their transitions.  This only works with 
        HJBSolver and OptimalControlTreeSolver, and displays the transitionMatrix or
        search tree, respectively.
        """
        ax = self.axcost if onplot == 'cost' else self.axpolicy
        if isinstance(self.gridSolver,HJBSolver):
            if self.gridSolver.transitionMatrix is not None:
                goalStates = []
                trans = self.gridSolver.transitionMatrix
                numshown = 0
                for (i,j),index in self.shownIndices():
                    numshown += 1
                    x = self.gridSolver.cellToCenterState(index)
                    options = trans[index]
                    for (xnext,cost,u) in options:
                        if xnext is None:
                            goalStates.append(x)
                        else:
                            ax.plot([x[self.xindex],xnext[self.xindex]],[x[self.yindex],xnext[self.yindex]],**kwargs)
                if len(goalStates) > 0:
                    goalStates = np.array(goalStates)
                    ax.scatter(goalStates[:,self.xindex],goalStates[:,self.yindex],**kwargs)
        elif isinstance(self.gridSolver,OptimalControlTreeSolver):
            def plottree(n):
                if n.state is not None and len(n.state) > 0:
                    doplot = True
                    index = self.gridSolver.stateToCell(n.state)
                    for i in range(len(self.cost.shape)):
                        if i == self.xindex or i == self.yindex:
                            continue
                        if index[i] != self.referenceCell[i]:
                            doplot = False
                            break
                    doplot = True
                    for c in n.children:
                        if c.parentedge is not None and doplot:
                            x1 = n.state
                            x2 = c.state
                            ax.plot([x1[self.xindex],x2[self.xindex]],[x1[self.yindex],x2[self.yindex]],**kwargs)
                    for c in n.children:
                        plottree(c)
            def fringestates(n):
                if hasattr(n,'heapNode') and n.heapNode is not None:
                    return [n.state]
                else:
                    return sum([fringestates(c) for c in n.children],[])
            plottree(self.gridSolver.root)
            fringe = np.array(fringestates(self.gridSolver.root))
            if len(fringe) > 0:
                ax.scatter(fringe[:,self.xindex],fringe[:,self.yindex],**kwargs)
