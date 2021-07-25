try:
    from OpenGL import GL
except ImportError:
    pass
from klampt.plan.cspace import CSpace,MotionPlan
from klampt.math import vectorops
from klampt.vis import gldraw
import math
import random

class Circle:
    def __init__(self,x=0,y=0,radius=1):
        self.center = (x,y)
        self.radius = radius
        
    def contains(self,point):
        return (vectorops.distance(point,self.center) <= self.radius)

    def signedDistance(self,point):
        return (vectorops.distance(point,self.center) - self.radius)
    
    def signedDistance_gradient(self,point):
        d = vectorops.sub(point,self.center)
        return vectorops.div(d,vectorops.norm(d))

    def drawGL(self,res=0.01):
        numdivs = int(math.ceil(self.radius*math.pi*2/res))
        GL.glBegin(GL.GL_TRIANGLE_FAN)
        GL.glVertex2f(*self.center)
        for i in range(numdivs+1):
            u = float(i)/float(numdivs)*math.pi*2
            GL.glVertex2f(self.center[0]+self.radius*math.cos(u),self.center[1]+self.radius*math.sin(u))
        GL.glEnd()
    
    def drawMPL(self,ax,**options):
        from matplotlib.patches import Circle
        ax.add_patch(Circle(self.center,self.radius,**options))


class Box:
    def __init__(self,x1=0,y1=0,x2=0,y2=0):
        self.bmin = [min(x1,x2),min(y1,y2)]
        self.bmax = [max(x1,x2),max(y1,y2)]

    def bounds(self):
        return (self.bmin,self.bmax)

    def sample(self):
        return [random.uniform(a,b) for (a,b) in zip(self.bmin,self.bmax)]

    def contains(self,x):
        assert len(x)==len(self.bmin)
        for (xi,a,b) in zip(x,self.bmin,self.bmax):
            if xi < a or xi > b:
                return False
        return True

    def project(self,x):
        assert len(x)==len(self.bmin)
        assert len(x)==len(self.bmax)
        xnew = x[:]
        for i,(xi,a,b) in enumerate(zip(x,self.bmin,self.bmax)):
            if xi < a:
                xnew[i] = a
            elif xi > b:
                xnew[i] = b
        return xnew

    def signedDistance(self,x):
        import numpy as np
        xclamp = np.zeros(len(x))
        assert len(x)==len(self.bmin)
        mindist = float('inf')
        for i,(xi,a,b) in enumerate(zip(x,self.bmin,self.bmax)):
            xclamp[i] = min(b,max(xi,a))
            mindist = min(mindist,xi-a,b-xi)
        if mindist < 0:
            #outside
            x = np.asarray(x)
            return np.dot(x-xclamp,x-xclamp)
        else:
            #inside
            return -mindist

    def signedDistance_gradient(self,x):
        import numpy as np
        xclamp = np.empty(len(x))
        assert len(x)==len(self.bmin)
        mindist = float('inf')
        imindist = None
        iminsign = 1.0
        for i,(xi,a,b) in enumerate(zip(x,self.bmin,self.bmax)):
            xclamp[i] = min(b,max(xi,a))
            if xi-a < mindist:
                imindist = i
                iminsign = -1.0
                mindist = xi-a
            if b-xi < mindist:
                imindist = i
                iminsign = 1.0
                mindist = b-xi
        if mindist < 0:
            #outside
            x = np.asarray(x)
            return 2*(x-xclamp)
        else:
            #inside
            res = np.zeros(len(x))
            res[imindist] = iminsign
            return res
        
    def drawGL(self):
        GL.glBegin(GL.GL_QUADS)
        GL.glVertex2f(*self.bmin)
        GL.glVertex2f(self.bmax[0],self.bmin[1])
        GL.glVertex2f(*self.bmax)
        GL.glVertex2f(self.bmin[0],self.bmax[1])
        GL.glEnd()
    
    def drawMPL(self,ax,**options):
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle(self.bmin,self.bmax[0]-self.bmin[0],self.bmax[1]-self.bmin[1],**options))


class Geometric2DCSpace (CSpace):
    def __init__(self):
        CSpace.__init__(self)
        self.bound = [[0,1],[0,1]]
        self.bmin = [0,0]
        self.bmax = [1,1]
        self.obstacles = []

    def addObstacle(self,obs):
        self.obstacles.append(obs)
        self.addFeasibilityTest(lambda x: not obs.contains(x))

    def toScreen(self,q):
        return (q[0]-self.bmin[0])/(self.bmax[0]-self.bmin[0]),(q[1]-self.bmin[1])/(self.bmax[1]-self.bmin[1])

    def toState(self,x,y):
        return (self.bmin[0]+x*(self.bmax[0]-self.bmin[0]),
                self.bmin[1]+y*(self.bmax[1]-self.bmin[1]))

    def beginDraw(self):
        if self.bmin != [0,0] or self.bmin != [1,1]:
            GL.glPushMatrix()
            GL.glScalef(1.0/(self.bmax[0]-self.bmin[0]),1.0/(self.bmax[1]-self.bmin[1]),1)
            GL.glTranslatef(-self.bmin[0],-self.bmin[1],0)

    def endDraw(self):
        if self.bmin != [0,0] or self.bmin != [1,1]:
            GL.glPopMatrix()

    def drawObstaclesGL(self):
        self.beginDraw()
        GL.glColor3f(0.2,0.2,0.2)
        for o in self.obstacles:
            o.drawGL()
        self.endDraw()

    def drawVerticesGL(self,qs):
        self.beginDraw()
        GL.glBegin(GL.GL_POINTS)
        for q in qs:
            GL.glVertex2f(q[0],q[1])
        GL.glEnd()
        self.endDraw()

    def drawRobotGL(self,q):
        GL.glColor3f(0,0,1)
        GL.glPointSize(7.0)
        self.drawVerticesGL([q])
    
    def drawObstaclesMPL(self,ax):
        for o in self.obstacles:
            o.drawMPL(ax,color=(0.2,0.2,0.2),alpha=0.5,linewidth=0)

    def drawVerticesMPL(self,ax,qs,options):
        Xs = [q[0] for q in qs]
        Ys = [q[1] for q in qs]
        ax.scatter(Xs,Ys,size=10,**options)
    
    def drawRobotMPL(self,ax,q,options):
        self.drawVerticesMPL(ax,[q],c=(0,0,1))


def circleTest():
    space = Geometric2DCSpace()
    space.addObstacle(Circle(0.5,0.4,0.39))
    start=[0.06,0.25]
    goal=[0.94,0.25]
    return space,start,goal

def rrtChallengeTest(w=0.03,eps=0.01):
    space = Geometric2DCSpace()
    space.bound[1][1] = w*2+eps
    space.bmax[1] = w*2+eps
    space.addObstacle(Box(0,w*2+eps,1,1))
    space.addObstacle(Box(w,w,1,w+eps))
    start=[1-w*0.5,w+eps+w*0.5]
    goal=[1-w*0.5,w*0.5]
    return space,start,goal

def kinkTest(w=0.02,bypass=True):
    if bypass:
        top = 0.7
    else:
        top = 1.0
    space = Geometric2DCSpace()
    space.addObstacle(Box(0.3,0,0.5+w*0.5,0.2-w*0.5))
    space.addObstacle(Box(0.5+w*0.5,0,0.7,0.3-w*0.5))
    space.addObstacle(Box(0.3,0.2+w*0.5,0.5-w*0.5,top))
    space.addObstacle(Box(0.5-w*0.5,0.3+w*0.5,0.7,top))
    start=[0.06,0.25]
    goal=[0.94,0.25]
    return space,start,goal

def bugtrapTest(w=0.1):
    space = Geometric2DCSpace()
    space.addObstacle(Box(0.55,0.25,0.6,0.75))
    space.addObstacle(Box(0.15,0.25,0.55,0.3))
    space.addObstacle(Box(0.15,0.7,0.55,0.75))
    space.addObstacle(Box(0.15,0.25,0.2,0.5-w*0.5))
    space.addObstacle(Box(0.15,0.5+w*0.5,0.2,0.75))
    start=[0.5,0.5]
    goal=[0.65,0.5]
    return space,start,goal


def testPlannerSuccessRate(N=100,duration=10,spawnFunc=lambda: kinkTest(0.0025,False)):
    import time
    import matplotlib.pyplot as plt
    finished = []
    for run in range(N):
        space,s,g = spawnFunc()
        space.eps = 1e-3
        if run == 0 and False:  #show space
            space.drawObstaclesMPL(plt.axes())
            plt.scatter([s[0],g[0]],[s[1],g[1]])
            plt.show()
        plan = MotionPlan(space,type='prm',knn=5)
        plan.setEndpoints(s,g)
        
        t0 = time.time()
        finished.append(None)
        while time.time()-t0 < duration:
            plan.planMore(5)
            if plan.getPath():
                finished[-1] = time.time()-t0
                print("Found path with",len(plan.getPath()),"milestones in",time.time()-t0,"s")
                break
        if finished[-1] is None:
            print("Failed to find path in",duration,"s")
    import numpy as np
    finished = [v for v in finished if v != None]
    hist,edges = np.histogram(finished,20,(0,duration))
    print(hist,edges)
    hist = hist*100/N
    chist = np.cumsum(hist)
    plt.bar(edges[:-1],100-chist,duration/20)
    plt.xlabel('Time (s)')
    plt.ylabel('% failed')
    plt.xlim(0,duration)
    plt.ylim(0,100)
    plt.savefig('histogram.png')
    plt.show()
    

def testOptimizingPlannerAnimations(PLANNER_TYPE,spawnFunc=circleTest):
    from matplotlib.animation import FuncAnimation, PillowWriter
    import matplotlib.pyplot as plt
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.split(__file__)[0]+'/..'))
    print(sys.path[-1])
    import plotting
    fig, ax = plt.subplots(figsize=(4,4))

    space,plan,s,g = None,None,None,None

    def init():
        space,s,g = spawnFunc()

        if PLANNER_TYPE == 'prm_neighborhood':
            plan = MotionPlan(space,type='prm',connectionThreshold=0.1,ignoreConnectedComponents=True)
        elif PLANNER_TYPE == 'prm_knn':
            plan = MotionPlan(space,type='prm',knn=5,ignoreConnectedComponents=True)
        elif PLANNER_TYPE == 'prmstar':
            plan = MotionPlan(space,type='prm*')
        elif PLANNER_TYPE == 'rrtstar':
            plan = MotionPlan(space,type='rrt*',bidirectional=False)
        else:
            raise ValueError("Invalid PLANNER_TYPE")
        plan.setEndpoints(s,g)

    def update(frame):
        ax.clear()
        ax.axis('equal')
        ax.set_xlim(space.bound[0][0],space.bound[0][1])
        ax.set_ylim(space.bound[1][0],space.bound[1][1])
    
        plan.planMore(100)
        space.drawObstaclesMPL(ax)
        G = plan.getRoadmap()
        plotting.mpl_plot_graph(ax,G,vertex_options={'s':4,'color':(1,1,0)},
            edge_options={'color':(0,0,0,0.2),'linewidth':0.5})
        path = plan.getPath()
        if path is not None:
            xs = [q[0] for q in path]
            ys = [q[1] for q in path]
            ax.plot(xs,ys,c='b',linewidth=2)
        ax.scatter([s[0]],[s[1]],s=20,color='g',zorder=10)
        ax.scatter([g[0]],[g[1]],s=20,color='r',zorder=10)
        ax.set_title('{} iterations'.format(plan.getStats()['numIters']))
    
    anim = FuncAnimation(fig,update,10,init_func=init)
    writergif = PillowWriter(fps=2) 
    f = PLANNER_TYPE+'.gif'
    anim.save(f, writer=writergif)
    plt.show()


if __name__ == '__main__':
    import sys
    testPlannerSuccessRate()
    # #PLANNER_TYPE = 'prm_neighborhood'
    # #PLANNER_TYPE = 'prm_knn'
    # #PLANNER_TYPE = 'prmstar'
    # PLANNER_TYPE = 'rrtstar'
    # if len(sys.argv) > 1:
    #     PLANNER_TYPE = sys.argv[1]
    # testOptimizingPlannerAnimations(PLANNER_TYPE)

