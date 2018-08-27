from scipy.stats import norm,multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from matplotlib import cm
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import math

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

xbound = [-3,3]
ybound = [-3,3]
N=100
X,Y = np.mgrid[ybound[0]:ybound[1]:(N*1j),xbound[0]:xbound[1]:(N*1j)]
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
nv = norm(0,1)
if figure == 'marginalization':
    rv = multivariate_normal(np.zeros(2),np.array([[0.25,0.3],[0.3,1]]))
else:
    rv = multivariate_normal(np.zeros(2),np.eye(2))
    
fig1 = plt.figure(figsize=(8,3))
ax1 = fig1.add_subplot(1, 2, 1, projection='3d')
ax1.set_xlim(*xbound)
ax1.set_ylim(*ybound)
Z = rv.pdf(pos)
if figure == 'marginalization':
    #plot of marginalization
    ax1.plot(xs=np.ones(x.shape)*3, ys=x, zs=nv.pdf(x), color='k')
    for y in x[::20]:
        a = Arrow3D([xbound[0], xbound[1]], [y,y], 
                    [0.01, 0.01], mutation_scale=20, 
                    lw=1, arrowstyle="-|>", color="r")
        ax1.add_artist(a)
    ax1.set_xlabel(r'$x_1$')
    ax1.set_ylabel(r'$x_2$')
elif figure == 'basic':
    ax1.set_xlabel(r"$x_1$")
    ax1.set_ylabel(r"$x_2$")
    ax1.set_title(r"$P(X_1=x_1,X_2=x_2)$")
#surface
vals = ax1.plot_surface(X, Y, Z,  rstride=3, cstride=3, cmap=cm.coolwarm, linewidth=0)

ax2 = fig1.add_subplot(1, 2, 2, projection='3d')
if figure == 'marginalization':
    #plot of conditioning
    xval=x[66]
    ax2.set_xlabel(r'$x_1$')
    ax2.set_ylabel(r'$x_2$')
    print xval,"shift =",rv.cov[1,0]/rv.cov[0,0]*xval
    cv = norm(rv.cov[1,0]/rv.cov[0,0]*xval,math.sqrt(rv.cov[1,1]-rv.cov[1,0]**2/rv.cov[0,0]))
    ax2.plot(xs=np.ones(x.shape)*xval, ys=x, zs=cv.pdf(x)*norm.pdf(xval,0,math.sqrt(rv.cov[0,0])), color='k',lw=0.5)
    ax2.plot(xs=np.ones(x.shape)*xval, ys=x, zs=cv.pdf(x), color='k')
    a = Arrow3D([xval, xval], [cv.mean(),cv.mean()], 
                [0.05, 0.4], mutation_scale=20, 
                lw=1, arrowstyle="-|>", color="r")
    ax2.add_artist(a)
    vals = ax2.plot_surface(X, Y, Z,  rstride=3, cstride=3, cmap=cm.coolwarm, linewidth=0)
elif figure == 'basic':
    ax2.set_zlim(0.0, 1.0)
    ax2.set_xlim(*xbound)
    ax2.set_ylim(*ybound)
    ax2.set_xlabel(r"$x_1$")
    ax2.set_ylabel(r"$x_2$")
    ax2.set_title(r"$P(X_1\leq x_1,X_2\leq x_2)$")
    Z = nv.cdf(X)*nv.cdf(Y)
    vals = ax2.plot_surface(X, Y, Z, color='w', rstride=3, cstride=3, cmap=cm.coolwarm, linewidth=1)
plt.show()
