from __future__ import print_function,division
from builtins import range
import numpy as np

def gradient_forward_difference(f,x,h):
    """Approximation of the gradient of f(x) using forward differences with step size h"""
    g = np.zeros(len(x))
    f0 = f(x)
    for i in range(len(x)):
        v = x[i]
        x[i] += h
        g[i] = (f(x)-f0)
        x[i] = v
    g *= 1.0/h
    return g

def jacobian_forward_difference(f,x,h):
    """Approximation of the Jacobian of vector function f(x) using forward differences with step size h"""
    f0 = np.asarray(f(x))
    J = np.zeros((len(f0),len(x)))
    for i in range(len(x)):
        v = x[i]
        x[i] += h
        J[:,i] = (np.asarray(f(x))-f0)
        x[i] = v
    J *= 1.0/h
    return J

def hessian_forward_difference(f,x,h):
    """Approximation of the hessian of f(x) using forward differences with
    step size h.
    """
    H = np.zeros((len(x),len(x)))
    f0 = f(x)
    fs = []
    for i in range(len(x)):
        v = x[i]
        x[i] += h
        fs.append(f(x))
        x[i] = v
    for i in range(len(x)):
        v = x[i]
        x[i] += h
        for j in range(i):
            w = x[j]
            x[j] += h
            fij = f(x)
            H[i,j] = (fij-fs[j]) - (fs[i]-f0)
            H[j,i] = H[i,j]
            x[j] = w
        x[i] = v
        x[i] -= h
        fij = f(x)
        H[i,i] = (fs[i]-f0) - (f0-fij)
        x[i] = v
    H *= 1.0/h**2
    return H

def hessian2_forward_difference(f,x,y,h):
    """Approximation of the hessian of a 2-parameter function f(x,y) w.r.t. x and y
    using forward differences with step size h.
    """
    H = np.zeros((len(x),len(y)))
    f0 = f(x,y)
    fxs = []
    fys = []
    for i in range(len(x)):
        v = x[i]
        x[i] += h
        fxs.append(f(x,y))
        x[i] = v
    for i in range(len(y)):
        v = y[i]
        y[i] += h
        fys.append(f(x,y))
        y[i] = v
    for i in range(len(x)):
        v = x[i]
        x[i] += h
        for j in range(len(y)):
            w = y[j]
            y[j] += h
            fij = f(x,y)
            H[i,j] = ((fij-fys[j]) - (fxs[i]-f0))
            y[j] = w
        x[i] = v
    H *= 1.0/h**2
    return H
