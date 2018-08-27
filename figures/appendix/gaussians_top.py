from scipy.stats import norm,multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib import cm
import numpy as np
import math

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

xbound = [-2,6]
ybound = [-4,4]
N=100
X,Y = np.mgrid[xbound[0]:xbound[1]:(N*1j),ybound[0]:ybound[1]:(N*1j)]
x= np.mgrid[xbound[0]:xbound[1]:(N*1j)]

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

figure = 'marginalization'
#figure = 'basic'

pos = np.dstack((X,Y))
A = np.array([[0.9,0.2],[0.0,1.3]])
b = np.array([2,0.0])
rv = multivariate_normal(np.zeros(2),np.eye(2))
rv2 = multivariate_normal(np.dot(A,rv.mean)+b,np.dot(A,np.dot(rv.cov,A.T)))
rv3 = multivariate_normal(np.dot(A,rv2.mean)+b,np.dot(A,np.dot(rv2.cov,A.T)))
    
fig1 = plt.figure(figsize=(8,3))
ax2 = fig1.add_subplot(1, 2, 1)
ax2.set_xlim(*xbound)
ax2.set_ylim(*ybound)
Z1 = rv.pdf(pos)
Z2 = rv2.pdf(pos)
Z3 = rv3.pdf(pos)
vals = ax2.contour(X, Y, Z1, 3, linewidth=1, cmap=cm.autumn)
vals = ax2.contour(X, Y, Z2, 3, linewidth=1, cmap=cm.autumn)
vals = ax2.contour(X, Y, Z3, 3, linewidth=1, cmap=cm.autumn)
ax2.scatter(*zip(rv.mean,rv2.mean,rv3.mean),color='k')

plt.show()
