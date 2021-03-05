from __future__ import print_function,division

from .graph import AdjListGraph
import itertools
import sys

def grid_graph(M,N,diagonals=False):
    """Makes a grid graph of size (M,N).  Vertices are indices (i,j).
    
    If diagonals=True, then diagonal edges are added.
    """
    G = AdjListGraph([],[])
    for i in range(M):
        for j in range(N):
            n = (i,j)
            G.add_node(n)
    for i in range(M):
        for j in range(N):
            n = (i,j)
            if i > 0:
                G.add_edge(n,(i-1,j))
            if j > 0:
                G.add_edge(n,(i,j-1))
            if i+1 < M:
                G.add_edge(n,(i+1,j))
            if j+1 < N:
                G.add_edge(n,(i,j+1))
            if diagonals:
                if i > 0 and j > 0:
                    G.add_edge(n,(i-1,j-1))
                if i > 0 and j+1 < N:
                    G.add_edge(n,(i-1,j+1))
                if i+1 < M and j > 0:
                    G.add_edge(n,(i+1,j-1))
                if i+1 < M and j+1 < N:
                    G.add_edge(n,(i+1,j+1))
    return G

def grid_node_neighbors_nd(index,diagonals=False,imin=None,imax=None,wrap=False):
    """Iterates over neighbors of a node in grid_graph_nd without actually
    constructing the grid.
    
    Args:
        index (array-like): the (integer) node
        diagonals (bool, optional): whether to iterate over diagonal edges. Note:
            if diagonals = True, there are 3^d-1 edges per node.
        imin (array-like, optional): if given, there is a lower bound on the
            grid.
        imax (array-like, optional): if given, there is an upper bound on the
            grid (index range is (imin[i]...,imax[i]-1)).
        wrap (bool or list of bools, optional): if True, the grid is allowed to
            wrap in all directions. If a list of bools, the grid is allowed to
            wrap in the given directions.
    """
    cap = True
    if imin is None and imax is None:
        cap = False
        wrap = False
    if imin is None and wrap is not False:
        imin = [-sys.maxint - 1]*len(index)
    if imax is None and wrap is not False:
        imax = [sys.maxint]*len(index)
    in_bounds = None
    enforce_bounds = None
    if not cap:
        pass
    elif wrap is False:
        in_bounds = lambda x,i: imin[i] <= x <= imax[i]-1
    elif wrap is True:
        enforce_bounds = lambda x,i: imax[i]-1 if x < imin[i] else (imin[i] if x >= imax[i] else x)
    else:
        assert hasattr(wrap,'__iter__')
        assert len(wrap) == len(index),"Wrap array must have the same size as shape"
        in_bounds = lambda x,i: True if wrap[i] else imin[i] <= x <= imax[i]-1
        enforce_bounds = lambda x,i: x if not wrap[i] else (imax[i]-1 if x < imin[i] else (imin[i] if x >= imax[i] else x))
    
    if diagonals:
        for ofs in itertools.product(*[[-1,0,1]]*len(index)):
            vn = [x+d for (x,d) in zip(index,ofs)]
            if in_bounds is not None:
                if not all(in_bounds(x,i) for (i,x) in enumerate(vn)):
                    continue
            if enforce_bounds is not None:
                vn = [enforce_bounds(x,i) for (i,x) in enumerate(vn)]
            vn = tuple(vn)
            if vn != index:
                yield vn
    else:
        #only add axes
        vn = list(index)
        for i,d in enumerate(index):
            vn[i] -= 1
            if in_bounds is None or in_bounds(vn[i],i):
                if enforce_bounds is not None:
                    vn[i] = enforce_bounds(vn[i],i)
                yield tuple(vn)
            vn[i] = d
            vn[i] += 1
            if in_bounds is None or in_bounds(vn[i],i):
                if enforce_bounds is not None:
                    vn[i] = enforce_bounds(vn[i],i)
                yield tuple(vn)
            vn[i] = d
    return

def grid_graph_nd(shape,diagonals=False,wrap=False):
    """Makes a grid graph of a given shape (d1,d2,..,dN). Vertices are
    indices (i1,...,iN).
    
    If diagonals=True, then diagonal edges are added.  Note: if diagonals=True,
    there are 3^d-1 edges per node.
    
    If wrap=True, or wrap is a d-length array with some True values, the grid
    is allowed to wrap in those dimensions.
    
    Requires Numpy.
    """
    import numpy as np
    
    if hasattr(wrap,'__iter__'):
        assert len(wrap) == len(shape),"Wrap array must have the same size as shape"
    else:
        wrap = [wrap]*len(shape)
    
    G = AdjListGraph([],[])
    for v in np.ndindex(*shape):
        v = tuple(v)
        G.add_node(v)
    for v in np.ndindex(*shape):
        if diagonals:
            for ofs in itertools.product(*[[-1,0,1]]*len(shape)):
                vn = [x+d for (x,d) in zip(v,ofs)]
                add = True
                for i,x in enumerate(vn):
                    if x < 0:
                        if wrap[i]:
                            x=shape[i]-1
                        else:
                            add = False
                            break
                    if x >= shape[i]:
                        if wrap[i]:
                            x=0
                        else:
                            add = False
                            break
                if add:
                    vn = tuple(vn)
                    if vn != v:
                        G.add_edge(v,vn)
        else:
            #only add axes
            for i,d in enumerate(v):
                vn = list(v)
                if d > 0 or wrap[i]:
                    vn[i] -= 1
                    if wrap[i] and vn[i] < 0: vn[i] = shape[i]-1
                    G.add_edge(v,tuple(vn))
                    vn[i] += 1
                if d+1 < shape[i] or wrap[i]:
                    vn[i] += 1
                    if wrap[i] and vn[i] >= shape[i]: vn[i] = 0
                    G.add_edge(v,tuple(vn))
                    vn[i] -= 1
    return G
