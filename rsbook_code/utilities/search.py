"""
Includes Dijkstra's algorithm and two A* implementations.
"""
from __future__ import print_function,division

import heapq  #for a fast priority queue implementation


def predecessor_traverse(p,s,g):
    """Used by dijkstra's algorithm to traverse a predecessor dictionary"""
    L = []
    v = g
    while v is not None:
        L.append(v)
        v = p.get(v,None)
    #rather than prepending, we appended and now we'll reverse.  This is a more efficient than prepending
    return L[::-1]


def dijkstras(G,s,g,cost=(lambda v,w:1),verbose=1):
    """Completes a shortest-path search on graph G.
    
    Args:
        G (AdjListGraph or networkx Graph/DiGraph): the graph to search
        s: the start node
        g: the goal node or a goal test
        cost (optional): a callback function c(v,w) that returns the edge cost
        verbose (optional): if nonzero, will print information about search
            progress.
    
    Returns:
        tuple: a triple (path,distances,predecessors) giving
            - path: a list or None: either the path of nodes from s to g with
              minimum cost, or None if no path exists.
            - distances: a dictionary mapping nodes to distances from start
            - predecessors: a dictionary mapping nodes to parent nodes 
              that can be walked by ``predecessor_traverse`` to get the optimal
              path to any reached node.
    """
    if not callable(g):
        gtest = lambda x,goal=g: x==g
    else:
        gtest = g
    d = dict((v,float('inf')) for v in G.nodes())
    p = dict((v,None) for v in G.nodes())
    d[s] = 0
    Q = [(0,s)]   #each element is a tuple (c,v) with c=cost from start, v=vertex
    nnodes = 0
    while len(Q) > 0:
        c,v = heapq.heappop(Q)  #get the element in the queue with the least value of c
        nnodes += 1
        if gtest(v):
            #found a path
            if verbose: print("Dijkstra's succeeded in",nnodes,"iterations")
            return predecessor_traverse(p,s,v),d,p
        for w in G.neighbors(v):
            dcand = d[v] + cost(v,w)   #this is the cost of going through v to w
            if dcand < d[w]:
                #going through v is optimal
                #if the predecessor of w is not None, then we'll have to adjust the heap
                if p[w] is not None:
                    Q = [(c,x) for (c,x) in Q if x is not w]
                    heapq.heapify(Q)
                d[w] = dcand
                p[w] = v
                #put w on the queue
                heapq.heappush(Q,(dcand,w))
    #no path found
    if verbose: print("Dijkstra's failed in",nnodes,"iterations")
    return None,d,p


def astar(G,s,g,cost=(lambda v,w:1),heuristic=(lambda v:0),verbose=1):
    """Completes an A* search on graph G.
    
    Args:
        G (AdjListGraph, networkx Graph / DiGraph): the graph to search.
        s: the start node
        g: the goal node or goal test
        cost (optional): a callback function c(v,w) that returns the edge cost
        heuristic (optional): a callback function h(v) that returns the 
            heuristic cost-to-go between v and g
        verbose (optional): if nonzero, will print information about search
            progress.
    
    Returns:
        tuple: a triple (path,distances,predecessors) giving
            - path: a list or None: either the path of nodes from s to g with
              minimum cost, or None if no path exists.
            - distances: a dictionary mapping nodes to distances from start
            - predecessors: a dictionary mapping nodes to parent nodes 
              that can be walked by ``predecessor_traverse`` to get the optimal
              path to any reached node.
    """
    if not callable(g):
        gtest = lambda x,goal=g: x==g
    else:
        gtest = g
    d = dict((v,float('inf')) for v in G.nodes())
    p = dict((v,None) for v in G.nodes())
    d[s] = 0
    Q = [(0,0,s)]   #each element is a tuple (f,-c,v) with f=c + heuristic(v), c=cost from start, v=vertex
    nnodes = 0
    while len(Q) > 0:
        f,minus_c,v = heapq.heappop(Q)  #get the element in the queue with the least value of c
        nnodes += 1
        if gtest(v):
            #found a path
            if verbose: print("A* succeeded in",nnodes,"iterations")
            return predecessor_traverse(p,s,v),d,p
        for w in G.neighbors(v):
            dcand = d[v] + cost(v,w)   #this is the cost of going through v to w
            if dcand < d[w]:
                #going through v is optimal
                #if the predecessor of w is not None, then we'll have to adjust the heap
                if p[w] is not None:
                    Q = [(f,c,x) for (f,c,x) in Q if x is not w]
                    heapq.heapify(Q)
                d[w] = dcand
                p[w] = v
                #put w back on the queue, with the heuristic value as its priority
                heapq.heappush(Q,(dcand+heuristic(w),-dcand,w))
    #no path found
    if verbose: print("A* failed in",nnodes,"iterations")
    return None,d,p


def astar_implicit(successors,s,g,cost=(lambda v,w:1),heuristic=(lambda v:0),verbose=1):
    """Completes an A* search on a large/infinite implicit graph.
    
    Args:
        successors: a callback function s(v) that returns a list of neighbors 
            of a node v.
        s: the start node
        g: the goal node or goal test
        cost (optional): a callback function c(v,w) that returns the edge cost
        heuristic (optional): a callback function h(v) that returns the 
            heuristic cost-to-go between v and g
        verbose (optional): if nonzero, will print information about search
            progress.
    
    Returns:
        tuple: a triple (path,distances,predecessors) giving
            - path: a list or None: either the path of nodes from s to g with
              minimum cost, or None if no path exists.
            - distances: a dictionary mapping reached nodes to distances from start
            - predecessors: a dictionary mapping reached nodes to parent nodes 
              that can be walked by ``predecessor_traverse`` to get the optimal
              path to any reached node.
    """
    if not callable(g):
        gtest = lambda x,goal=g: x==g
    else:
        gtest = g
    inf = float('inf')
    d = dict()
    p = dict()
    d[s] = 0
    Q = [(0,0,s)]   #each element is a tuple (f,-c,v) with f=c + heuristic(v), c=cost from start, v=vertex
    nnodes = 0
    while len(Q) > 0:
        f,minus_c,v = heapq.heappop(Q)  #get the element in the queue with the least value of c
        nnodes += 1
        if gtest(v):
            #found a path
            if verbose: print("A* succeeded in",nnodes,"iterations")
            return predecessor_traverse(p,s,v),d,p
        for w in successors(v):
            dcand = d[v] + cost(v,w)   #this is the cost of going through v to w
            if dcand < d.get(w,float('inf')):
                #going through v is optimal
                #if the predecessor of w is not None, then we'll have to adjust the heap
                if w in p:
                    Q = [(f,c,x) for (f,c,x) in Q if x is not w]
                    heapq.heapify(Q)
                d[w] = dcand
                p[w] = v
                #put w back on the queue, with the heuristic value as its priority
                heapq.heappush(Q,(dcand+heuristic(w),-dcand,w))
    #no path found
    if verbose: print("A* failed in",nnodes,"iterations")
    return None,d,p
