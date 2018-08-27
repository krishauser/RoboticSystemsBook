import matplotlib.pyplot as plt
import numpy as np
from plotting import *

def PID(x,I,dx,KP,KI,KD):
    u = -np.dot(KP,x)-np.dot(KI,I)-np.dot(KD,dx)
    return u

def PID_trajectory(A,B,c,D,x0,dx0,KP,KI,KD,dt=1e-3,T=10,xdes=None,dxdes=None):
    """
    For 2nd order system Ax + Bdx + c + Du and PID controller
    Returns dictionary of trajectories
    [t(t), x(t), I(t), dx(t), ddx(t), u(t), uP(t), uI(t), uD(t)]"""
    x = x0.copy()
    dx = dx0.copy()
    I = np.zeros(len(x0))
    res = dict((idx,[]) for idx in ['t','xdes','dxdes','x','I','dx','ddx','u','uP','uI','uD'])
    t = 0
    while t < T:
        xd = xdes(t) if xdes is not None else np.zeros(len(x0))
        dxd = dxdes(t) if dxdes is not None else np.zeros(len(x0))
        u = PID(x-xd,I,dx-dxd,KP,KI,KD)
        ddx = np.dot(A,x-xd) + np.dot(B,dx-dxd) + c + np.dot(D,u)
        res['t'].append(t)
        res['x'].append(x.copy())
        if xdes is not None:
            res['xdes'].append(xd)
        if dxdes is not None:
            res['dxdes'].append(dxd)
        res['I'].append(I)
        res['dx'].append(dx.copy())
        res['u'].append(u)
        res['ddx'].append(ddx.copy())
        res['uP'].append(-np.dot(KP,x-xd))
        res['uD'].append(-np.dot(KD,dx-dxd))
        res['uI'].append(-np.dot(KI,I))
        I += dt*(x-xd)
        t += dt
        x += dx*dt
        dx += ddx*dt
    return res

        
A=np.zeros((2,2))
A[1,0] = 0.4
A[0,1] = -0.4
B=np.zeros((2,2))
B[1,0] = 0
B[0,1] = -0
c=np.zeros(2)
#c[1] = 0.5
D=np.eye(2)*1
x0=np.array([1.,0.])
dx0=np.array([0.,0.])
res1 = PID_trajectory(A,B,c,D,x0=x0,dx0=dx0,KP=np.eye(2)*1,KI=np.eye(2)*0.25,KD=np.eye(2)*2,dt=0.01,T=20)
res2 = PID_trajectory(A,B,c,D,x0=x0,dx0=dx0,KP=np.eye(2)*1,KI=np.eye(2)*0.25,KD=np.eye(2)*1,dt=0.01,T=20)
res3 = PID_trajectory(A,B,c,D,x0=x0,dx0=dx0,KP=np.eye(2)*1,KI=np.eye(2)*0.25,KD=np.eye(2)*0.5,dt=0.01,T=20)
#plotmulti([res1,res1],['x','x'],['x','x'],[0,1],['black','green','red','blue'])
plt.figure(figsize=(6,6))
plotxy([res1,res2,res3],['kD=2','kD=1','kD=0.5'],['x','x','x'],[0,1],['black','green','red','blue'])
#plotxy([res1,res1],['x','u'],['x','u'],[0,1],['black','green','red','blue'])
plt.show()
