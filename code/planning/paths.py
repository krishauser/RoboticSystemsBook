import numpy as np
import math

def linear(m0,m1,u,domain=None):
    """For completeness: linear interpolation between m0 and m1 at parameter u in [0,1].
    
    Alternatively, if `domain` != None, then this will use domain=(a,b)
    """
    if domain is not None:
        return linear(m0,m1,(u-domain[0])/(domain[1]-domain[0]))
    return (1.0-u)*m0 + u*m1


def piecewise_linear(ms,u,times=None):
    """Evaluate a piecewise linear spline at interpolation parameter u in [0,1]
    
    Milestones are given by the array `ms`.  `ms` is assumed to be a list of
    n Numpy arrays, or an n x d Numpy array.
    
    If `times` != None, this will be a list of n non-decreasing time indices.
    """
    if times is not None:
        raise NotImplementedError("Not done with timed paths")
    n = len(ms)
    s = u*n
    i = int(math.floor(s))
    u = s - i
    if i < 0: return ms[0]
    elif i+1 >= n: return ms[-1]
    return linear(ms[i],ms[i+1],u)


def hermite(m0,m1,t0,t1,u,domain=None):
    """Evaluate a cubic hermite curve at interpolation parameter u in [0,1].
    
    Endpoints are m0 and m1, with derivatives t0 and t1.  These are assumed to be Numpy arrays.
    
    Alternatively, if `domain` != None, then this will use domain=(a,b)
    as the interpolation domain
    """
    if domain is not None:
        assert isinstance(domain,(list,tuple)) and len(domain) == 2,"Need to provide a pair as a domain"
        scale = (domain[1]-domain[0])
        t = (u - domain[0])/scale
        return hermite(m0,m1,t0*scale,t1*scale,t)
    u2 = u**2
    u3 = u**3
    cm0 = 2*u3 - 3*u2 + 1
    cm1 = -2*u3 + 3*u2
    ct0 = u3 - 2*u2 + u
    ct1 = u3 - u2
    return cm0*m0 + cm1*m1 + ct0*t0 + ct1*t1


def hermite_deriv(m0,m1,t0,t1,u,domain=None):
    """Evaluate the derivative of a cubic hermite curve at interpolation parameter u in [0,1].
    
    Endpoints are m0 and m1, with derivatives t0 and t1.  These are assumed to be numpy arrays.
    
    Alternatively, if `domain` != None, then this will use domain=(a,b)
    as the interpolation domain
    """
    if domain is not None:
        assert isinstance(domain,(list,tuple)) and len(domain) == 2,"Need to provide a pair as a domain"
        scale = (domain[1]-domain[0])
        t = (u - domain[0])/scale
        return hermite_deriv(m0,m1,t0*scale,t1*scale,t)
    u2 = u**2
    cm0 = 6*u2 - 6*u
    cm1 = -6*u2 + 6*u
    ct0 = 3*u2 - 4*u + 1
    ct1 = 3*u2 - 2*u
    return cm0*m0 + cm1*m1 + ct0*t0 + ct1*t1


def hermite_spline(ms,ts,u,times=None):
    """Evaluate a cubic hermite spline at interpolation parameter u in [0,1].
    
    Milestones are given in `ms`, with derivatives in `ts`.  These are assumed to be 
    lists of n Numpy arrays, or n x d Numpy arrays.
    
    If `times` != None, this will be a list of n non-decreasing time indices.
    """
    if times is not None:
        raise NotImplementedError("Not done with timed paths")
    
    n = len(ms)
    s = u*n
    i = int(math.floor(s))
    u = s - i
    if i < 0: return ms[0]
    elif i+1 >= n: return ms[-1]
    return hermite(ms[i],ms[i+1],ts[i],ts[i+1],u)
    
