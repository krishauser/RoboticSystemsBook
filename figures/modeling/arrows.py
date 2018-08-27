from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from klampt.math import so3,se3,vectorops

class Arrow3D(FancyArrowPatch):
    def __init__(self, start, end, shrinkA=0.0, shrinkB=0.0, mutation_scale=20, arrowstyle="-|>", color='k', lw=1, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, shrinkA=shrinkA, shrinkB=shrinkB, mutation_scale=mutation_scale, arrowstyle=arrowstyle, color=color, lw=lw, **kwargs)
        self._verts3d = zip(start,end)

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def add_line(ax,a,b,linestyle=None, color='k',lw=1,*args,**kwargs):
    """Draws a default line between a and b on the plot ax"
    ax.plot([a[0],b[0]], [a[1],b[1]], [a[2],b[2]], linestyle, *args, color=color, lw=lw, **kwargs)

def add_arrow(ax,a,b,linestyle=None,*args,**kwargs):
    """Draws a default arrow from a to b on the plot ax"
    a = Arrow3D(a,b, linestyle=linestyle, *args, **kwargs)
    ax.add_artist(a)

def add_coordinate_transform(ax,R,t,size=1.0,*args,**kwargs):
    """Draws a coordinate transform on the plot ax"""
    axes = so3.matrix(so3.transpose(R))
    colors = ['r','g','b']
    for (v,c) in zip(axes,colors):
        a = Arrow3D(t, vectorops.madd(t,v,size), lw=1, color=c, *args, **kwargs)
        ax.add_artist(a)


if __name__ == '__main__':
    ####################################################
    # This part is just for reference if
    # you are interested where the data is
    # coming from
    # The plot is at the bottom
    #####################################################

    import numpy as np
    from numpy import *

    # Generate some example data
    mu_vec1 = np.array([0,0,0])
    cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20)

    mu_vec2 = np.array([1,1,1])
    cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20)

    # concatenate data for PCA
    samples = np.concatenate((class1_sample, class2_sample), axis=0)

    # mean values
    mean_x = mean(samples[:,0])
    mean_y = mean(samples[:,1])
    mean_z = mean(samples[:,2])

    #eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(cov_mat1)

    ################################
    #plotting eigenvectors
    ################################    

    fig = plt.figure(figsize=(4,4),dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(samples[:,0], samples[:,1], samples[:,2], 'o', markersize=10, color='g', alpha=0.2)
    ax.plot([mean_x], [mean_y], [mean_z], 'o', markersize=10, color='red', alpha=0.5)
    colors = ['r','g','b']
    for c,v in zip(colors,eig_vec):
        a = Arrow3D([mean_x, mean_y, mean_z], v, color=c)
        ax.add_artist(a)
    add_line(ax,[0,0,0],[1,1,1],'--',lw=1)
    add_arrow(ax,[1,0,0],[2,1,1],'dashed',lw=1)
    ax.set_xlabel(r'$\alpha x_values$')
    ax.set_ylabel('y_values')
    ax.set_zlabel('z_values')

    plt.title('Eigenvectors')

    plt.draw()
    plt.show()
