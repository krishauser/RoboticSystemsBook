from __future__ import print_function,division

import math
import numpy as np
import itertools
from ..utilities import search,graph,example_graphs
import time

DEFAULT_RESOLUTION = 100


def grid_search(xstart,xgoal,
    xmin,xmax,
    obstacle_test,obstacle_edge_test=None,
    resolution=None,diagonals=True,
    cost=None,terminal_cost=None,
    implicit=True,
    verbose=0):
    """Performs a grid search motion planning in a Cartesian space.
    
    Can be used in point-to-point mode, point-to-set mode, and also diffusion.
    
    Args:
        xstart (array-like): start config
        xgoal (array-like or function g(x)): goal config, or goal test.
        xmin (array-like): grid minimum
        xmax (array-like): grid maximum
        obstacle_test (function f(x)): returns True if x is in an obstacle.
        obstacle_edge_test (function f(x,y), optional): returns True if the
            edge x->y hits an obstacle.  Default ignores edge collisions.
        resolution (float, optional): the resolution of the grid. Default is 1%
            of the max difference between xmin and xmax.
        diagonals (bool, optional): set to False to stay precisely on the grid.
        cost (function c(x,y), optional): returns the movement cost from x
            to y.
        terminal_cost (function cT(x), optional): returns the cost of
            terminating at x.
        implicit (bool, optional): if True, will use implicit search. 
            Otherwise, will construct the entire graph (False is used mostly
            for testing).
        verbose (int,optional): if > 0, will print information
    
    Returns:
        tuple: a triple (path,distances,predecessors) giving
            - path (list of states or None): the configuration-space path.
            - distances (dict of tuples -> floats): the distances for each
              grid cell.  Note that you will need to 
    """
    xstart = np.asarray(xstart)
    if not callable(xgoal):
        xgoal = np.asarray(xgoal)
    xmin = np.asarray(xmin)
    xmax = np.asarray(xmax)
    assert(len(xstart) == len(xmin))
    assert(len(xstart) == len(xmax))
    if cost is None:
        cost = lambda x,y: np.linalg.norm(x-y)
    if obstacle_edge_test is None:
        obstacle_edge_test = lambda x,y:False
    assert callable(cost)
    if resolution is None:
        rmax = np.max(xmax-xmin)
        resolution = rmax/DEFAULT_RESOLUTION
    invresolution  = np.divide(1.0,resolution)
    dims = [int(math.ceil(v)) for v in np.multiply(xmax-xmin,invresolution)]
    grid_start = 'start'
    grid_goal = 'goal'
    def to_state(v):
        if isinstance(v,str):
            if v is grid_start: return xstart
            if v is grid_goal: return xgoal
            else: raise ValueError("Invalid grid node "+str(v))
        return np.multiply(np.asarray(v),resolution) + xmin
    def from_state(x):
        #returns the closest grid cell to x
        return tuple(np.multiply(x-xmin,invresolution).astype(int))
    def grid_near(x):
        #returns all of the grid vertices of the cell of x
        corner = np.floor(np.multiply(x-xmin,invresolution)).astype(int)
        res = []
        for ofs in itertools.product(*[[0,1]]*len(xmin)):
            res.append(tuple(corner + ofs))
        return res

    #create main graph
    goal_states = []
    if not implicit:
        t0 = time.time()
        G = example_graphs.grid_graph_nd(dims,diagonals)
        t1 = time.time()
        if verbose:
            print("grid_search: Grid has",len(G.nodes()),"nodes, time to create:",t1-t0)
        #add start and goal nodes
        G.vertices.append(grid_start)
        G.edges[grid_start] = grid_near(xstart)
        if not callable(xgoal) or terminal_cost is not None:
            G.vertices.append(grid_goal)
            G.edges[grid_goal] = []

        #cast goal tests to grid
        goal_states = []
        if callable(xgoal):
            if terminal_cost is None:
                #goal set, and no terminal cost
                grid_goal = lambda v: xgoal(to_state(v))
            else:
                #goal set + terminal cost.  Only way to do this with a graph is to add edges from each goal node to the virtual goal
                for v in G.vertices:
                    if xgoal(v):
                        G.edges[v].append(grid_goal)
                        goal_states.append(to_state(v))
        else:
            #goal point
            if terminal_cost is not None:
                print("grid_search: warning, can't use a terminal_cost with a single goal state""")
            for vg in grid_near(xgoal):
                G.edges[vg].append(grid_goal)
            terminal_cost = lambda x:cost(x,xgoal)
        #G must still be a valid graph
        G.assert_valid()
    else:
        goal_nodes = set()
        if callable(xgoal):
            pass
        else:
            if terminal_cost is not None:
                print("grid_search: warning, can't use a terminal_cost with a single goal state""")
            goal_nodes = set(grid_near(xgoal))
    
    #cast costs to the grid
    def grid_cost(v,w):
        if v is grid_start:
            x = xstart
        else:
            x = to_state(v)
        if w is grid_goal:
            return 0 if terminal_cost is None else terminal_cost(x)
        else:
            y = to_state(w)
        return cost(x,y) 
    
    heuristic = None
    if callable(xgoal):
        if len(goal_states) > 0:
            heuristic = lambda v:min(cost(to_state(v),g) for g in goal_states)
        else:
            heuristic = lambda v:0
    else:
        heuristic = lambda v:cost(to_state(v),xgoal)
    t0 = time.time()
    if implicit:
        imin = [0]*len(xstart)
        def successors(v):
            if v is grid_start:
                xv = xstart
                options = grid_near(xstart)
            else:
                xv = to_state(v)
                options = list(example_graphs.grid_node_neighbors_nd(v,diagonals=diagonals,imin=imin,imax=dims))
                if v in goal_nodes:
                    options.append(grid_goal)
            if callable(xgoal) and xgoal(xv):
                #extra edge to account for terminal cost
                options.append(grid_goal)
            for w in options:
                if w is grid_goal: 
                    yield w
                    continue
                xw = to_state(w)
                if not obstacle_test(xw):
                    if not obstacle_edge_test(xv,xw):
                        yield w
    else:
        def successors(v):
            xv = to_state(v)
            for w in G.neighbors(v):
                if w is grid_goal:
                    yield w
                else:
                    xw = to_state(w)
                    if not obstacle_test(xw):
                        if not obstacle_edge_test(xv,xw):
                            yield w
            return
    path,d,p = search.astar_implicit(successors,grid_start,grid_goal,grid_cost,heuristic,verbose=max(verbose-1,0))
    t1 = time.time()
    if verbose:
        if path is not None:
            print("grid_search: search computed path with",len(path),"nodes, cost",d[grid_goal],"taking time:",t1-t0)
        else:
            print("grid_search: search returned no path, taking time:",t1-t0)
    if path is None:
        return None,d,p
    #convert grid-space path to state space
    res = []
    for v in path:
        if v is grid_start:
            res.append(xstart)
        elif v is grid_goal:
            if callable(xgoal):
                pass
            else:
                res.append(xgoal)
        else:
            res.append(to_state(v))
    return res,d,p


def grid_state_to_tuple(x,xmin,xmax,resolution=None,invresolution=None):
    """Returns the closest grid cell to state x.  For best performance,
    assumes x's are numpy ndarrays.
    
    For a slight speedup, set invresolution = 1.0/resolution
    """
    if invresolution is None:
        if resolution is None:
            rmax = np.max(xmax-xmin)
            invresolution = DEFAULT_RESOLUTION/rmax
            return tuple(np.multiply(x-xmin,invresolution).astype(int))
        return tuple(np.divide(x-xmin,resolution).astype(int))
    else:
        return tuple(np.multiply(x-xmin,invresolution).astype(int))


def grid_tuple_to_state(index,xmin,xmax,resolution=None):
    """Returns the state x corresponding to grid cell index.  For best
    performance, assumes x's are numpy ndarrays.
    """
    if resolution is None:
        rmax = np.max(xmax-xmin)
        resolution = rmax/DEFAULT_RESOLUTION
    return np.multiply(np.asarray(index),resolution) + xmin


def optimal_path(x,distances,predecessors,xstart,xmin,xmax,resolution=None,cost=None):
    """Given the outputs distances, predecessors from grid search, and its 
    parameters xstart,xmin,xmax,resolution,cost, returns the optimal path
    to x.  This is a fast operation suitable for real-time use.
    
    Assumes there are no within-cell obstacles at x.
    
    Returns the path from xstart to x.
    """
    xmin = np.asarray(xmin)
    xmax = np.asarray(xmax)
    if resolution is None:
        rmax = np.max(xmax-xmin)
        resolution = rmax/DEFAULT_RESOLUTION
    invresolution = np.divide(1.0,resolution)
    corner = np.floor(np.multiply(x-xmin,invresolution)).astype(int)
    def to_state(index):
        return np.multiply(np.asarray(index),resolution) + xmin
    cellvertices = []
    for ofs in itertools.product(*[[0,1]]*len(xmin)):
        cellvertices.append(tuple(corner + ofs))
    if cost is None:
        cost = lambda x,y:np.linalg.norm(x-y)
    dmin = float('inf')
    vmin = None
    for v in cellvertices:
        if v in distances:
            xv = to_state(v)
            d = distances[v] + cost(xv,x)
            if d < dmin:
                dmin = d
                vmin = v
    if vmin is None:
        return None
    path = search.predecessor_traverse(predecessors,'start',vmin)
    xpath = []
    for v in path:
        if v == 'start':
            xpath.append(xstart)
        else:
            xpath.append(to_state(v))
    if np.any(xpath[-1] != x):
        xpath.append(x)
    return xpath


def optimal_path_cost(x,distances,predecessors,xstart,xmin,xmax,resolution=None,cost=None):
    """Given the outputs distances, predecessors from grid search, and its 
    parameters xstart,xmin,xmax,resolution,cost, returns the cost of the
    optimal path to x.  This is an O(1) operation!
    
    Assumes there are no within-cell obstacles at x.
    
    Returns the cost of the path from xstart to x.
    """
    xmin = np.asarray(xmin)
    xmax = np.asarray(xmax)
    if resolution is None:
        rmax = np.max(xmax-xmin)
        resolution = rmax/DEFAULT_RESOLUTION
    invresolution = np.divide(1.0,resolution)
    corner = np.floor(np.multiply(x-xmin,invresolution)).astype(int)
    cellvertices = []
    cellvertexstates = []
    for ofs in itertools.product(*[[0,1]]*len(xmin)):
        cellvertices.append(tuple(corner + ofs))
        cellvertexstates.append(np.multiply(cellvertices[-1],resolution)+xmin)
    if cost is None:
        cost = lambda x,y:np.linalg.norm(x-y)
    dmin = float('inf')
    for v,xv in zip(cellvertices,cellvertexstates):
        if v in distances:
            dmin = min(dmin,distances[v] + cost(xv,x))
    return dmin
