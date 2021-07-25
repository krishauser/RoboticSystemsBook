import numpy as np
import heapq

def fmm(f,x0,bmin,bmax,h):
    """Uses the Fast Marching Method to solve for the shortest paths from point
    x0 to all points in a domain [bmin,bmax] under the incremental cost function
    f(x).

    The resulting cost field V(x) is the value function for the cost
    $\min_y \int_0^1 f(y(t))\|y^\prime(t)\| dt$ subject to $y(0)=x0$ and $y(1)=x$.
    
    Cost is O(N log N) with N the number of grid cells
    $\prod_i \lceil(bmax_i-bmin_i)/h\rceil$.

    Args:
        f (callable): the cost function
        x0 (ndarray): the source point
        bmin (ndarray): the grid minimum
        bmax (ndarray): the grid maximum
        h (float): grid resolution
    
    Returns:
        ndarray: A grid representing the value function V over the domain [bmin,bmax].
    
    """
    dims = np.ceil((bmax-bmin)/h).astype(int)
    d = np.full(dims,float('inf'))
    i0 = np.floor((x0-bmin)/h).astype(int)
    f0 = f(x0)
    x0_closest = (i0*h+bmin)
    i0 = tuple(i0)
    d[i0] = np.linalg.norm(x0 - x0_closest)/f0
    Q = []
    heapq.heappush(Q,(0,i0))
    CLOSED = set([i0])
    FRINGE = set()
    while len(Q)>0:
        (dn,n) = heapq.heappop(Q)
        x = np.array(n)*h+bmin
        #print("Pop",n,"=",x,"at dist",dn)
        fx = f(x)
        if fx == 0:
            if n in FRINGE:
                d[n] = float('inf')
                FRINGE.remove(n)
                CLOSED.add(n)
            continue
        if n in FRINGE:
            #consider propagating from closed items
            close = []
            for i in range(len(x0)):
                temp = list(n)
                temp[i] += 1
                temp = tuple(temp)
                temp2 = list(n)
                temp2[i] -= 1
                temp2 = tuple(temp2)
                if temp in CLOSED and temp2 in CLOSED:
                    if d[temp] > d[temp2]:
                        temp = temp2
                    close.append(temp)
                elif temp in CLOSED:
                    close.append(temp)
                elif temp2 in CLOSED:
                    close.append(temp2)
                else:
                    close.append(None)
            close = [c for c in close if c is not None]
            assert len(close) >= 1
            if len(close) == 1:
                dnew = d[close[0]]+h/fx
            else:
                dclose = [d[c] for c in close]
                N = len(dclose)
                #magic formula
                dnew = np.average(dclose)+np.sqrt(np.sum(dclose)**2-N*(np.dot(dclose,dclose)-(h/fx)**2))/N
            if dnew < d[n]:
                d[n] = dnew
            FRINGE.remove(n)
            CLOSED.add(n)
        else:
            pass
            
        for i in range(len(x0)):
            #loop over neighbors
            temp = list(n)
            temp[i] += 1
            if temp[i] < d.shape[i]:
                temp = tuple(temp)
                if temp not in FRINGE and temp not in CLOSED:
                    d[temp] = d[n] + h/fx
                    FRINGE.add(temp)
                    heapq.heappush(Q,(d[temp],temp))
            temp = list(n)
            temp[i] -= 1
            if temp[i] >= 0:
                temp = tuple(temp)
                if temp not in FRINGE and temp not in CLOSED:
                    d[temp] = d[n] + h/fx
                    FRINGE.add(temp)
                    heapq.heappush(Q,(d[temp],temp))
    return d