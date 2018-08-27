import matplotlib.pyplot as plt

def PID(x,I,dx,kP,kI,kD):
    u = -kP*x-kI*I-kD*dx
    return u

def PID_trajectory(a,b,c,d,x0,dx0,kP,kI,kD,dt=1e-3,T=10,xdes=None,dxdes=None):
    """Returns dictionary of trajectories
    [t(t), x(t), I(t), dx(t), ddx(t), u(t), uP(t), uI(t), uD(t)]"""
    x = x0
    dx = dx0
    I = 0
    res = dict((idx,[]) for idx in ['t','xdes','dxdes','x','I','dx','ddx','u','uP','uI','uD'])
    t = 0
    while t < T:
        xd = xdes(t) if xdes is not None else 0
        dxd = dxdes(t) if dxdes is not None else 0
        u = PID(x-xd,I,dx-dxd,kP,kI,kD)
        ddx = a*(x-xd) + b*(dx-dxd) + c + d*u
        res['t'].append(t)
        res['x'].append(x)
        if xdes is not None:
            res['xdes'].append(xd)
        if dxdes is not None:
            res['dxdes'].append(dxd)
        res['I'].append(I)
        res['dx'].append(dx)
        res['u'].append(u)
        res['ddx'].append(ddx)
        res['uP'].append(-kP*(x-xd))
        res['uD'].append(-kD*(dx-dxd))
        res['uI'].append(-kI*I)
        I += dt*(x-xd)
        t += dt
        x += dx*dt
        dx += ddx*dt
    return res


def plotmulti(trajs,names,which=['x','xdes'],colors=None,linestyles=None):
    if linestyles is None:
        linestyles = ['-', '--', ':', '-.']
    plots = []
    labels = []
    for i,(res,name,item) in enumerate(zip(trajs,names,which)):
        color = None if colors is None else colors[i]
        linestyle = None if linestyles is None else linestyles[i]
        x,=plt.plot(res['t'],res[item],label=item,color=color,linestyle=linestyle)
        plots.append(x)
        labels.append(name)
    plt.legend(labels=labels,handles=plots)
    plt.show()

def plotarrows(res):
    x,=plt.plot(res['t'],res['x'],label='x')
    #xdes,=plt.plot(res['t'],res['xdes'],label='xdes',color='black')
    #dx,=plt.plot(res['t'],res['dx'],label='dx')
    #I,=plt.plot(res['t'],res['I'],label='I')
    #uP,=plt.plot(res['t'],res['uP'],label='uP')
    #uI,=plt.plot(res['t'],res['uI'],label='uI')
    #uD,=plt.plot(res['t'],res['uD'],label='uD')
    #plt.legend(handles=[x,dx,I,uP])
    ax = plt.axes()
    ax.yaxis.grid(which='major',color='gray',linestyle=':')
    skip = 500
    for i in range(0,len(res['t']),skip):
        t = res['t'][i]
        x = res['x'][i]
        scale = 0.2
        fp = res['uP'][i]*scale
        fd = res['uD'][i]*scale
        fi = res['uI'][i]*scale
        xshift = 0.1
        shift = 0
        if abs(fp) > 0.01:
            sgn = fp/abs(fp)
            ax.arrow(t, x, 0, sgn*max(0.001,(abs(fp)-0.05)), head_width=0.2, head_length=0.05, fc='r', ec='r')
            if (fp > 0) != (fd > 0):
                shift += xshift
        if abs(fd) > 0.01:
            sgn = fd/abs(fd)
            ax.arrow(t+shift, x+fp, 0, sgn*max(0.001,(abs(fd)-0.05)), head_width=0.2, head_length=0.05, fc='g', ec='g')
            if (fd > 0) != (fi > 0):
                shift += xshift
        if abs(fi) > 0.01:
            sgn = fi/abs(fi)
            ax.arrow(t+shift, x+fp+fd, 0, sgn*max(0.001,(abs(fi)-0.05)), head_width=0.2, head_length=0.05, fc='b', ec='b')
    #plt.legend()
    plt.show()
        
plot = 'kd'
if plot == 'kp':
    res1 = PID_trajectory(0,0,0,20,x0=0,dx0=0,kP=1,kI=1,kD=1,xdes=lambda t:(0 if t < 1 else 1))
    res2 = PID_trajectory(0,0,0,20,x0=0,dx0=0,kP=2,kI=1,kD=1,xdes=lambda t:(0 if t < 1 else 1))
    res3 = PID_trajectory(0,0,0,20,x0=0,dx0=0,kP=0.5,kI=1,kD=1,xdes=lambda t:(0 if t < 1 else 1))
    plotmulti([res1,res1,res2,res3],['xd','kP=1, kI=1, kD=1','kP=2','kP=0.5'],['xdes','x','x','x'],['black','green','red','blue'],['-','-','--',':'])
elif plot == 'ki':
    res1 = PID_trajectory(0,0,0,20,x0=0,dx0=0,kP=1,kI=1,kD=1,xdes=lambda t:(0 if t < 1 else 1))
    res2 = PID_trajectory(0,0,0,20,x0=0,dx0=0,kP=1,kI=2,kD=1,xdes=lambda t:(0 if t < 1 else 1))
    res3 = PID_trajectory(0,0,0,20,x0=0,dx0=0,kP=1,kI=0.5,kD=1,xdes=lambda t:(0 if t < 1 else 1))
    plotmulti([res1,res1,res2,res3],['xd','kP=1, kI=1, kD=1','kI=2','kI=0.5'],['xdes','x','x','x'],['black','green','red','blue'],['-','-','--',':'])
elif plot == 'kd':
    res1 = PID_trajectory(0,0,0,20,x0=0,dx0=0,kP=1,kI=1,kD=1,xdes=lambda t:(0 if t < 1 else 1))
    res2 = PID_trajectory(0,0,0,20,x0=0,dx0=0,kP=1,kI=1,kD=2,xdes=lambda t:(0 if t < 1 else 1))
    res3 = PID_trajectory(0,0,0,20,x0=0,dx0=0,kP=1,kI=1,kD=0.5,xdes=lambda t:(0 if t < 1 else 1))
    plotmulti([res1,res1,res2,res3],['xd','kP=1, kI=1, kD=1','kD=2','kD=0.5'],['xdes','x','x','x'],['black','green','red','blue'],['-','-','--',':'])
elif plot == 'pd':
    #testing P vs PD control on unstable oscillator
    res = PID_trajectory(0,0.1,0,1,x0=1,dx0=0,kP=3,kI=0,kD=0)
    #res = PID_trajectory(0,0.1,0,1,x0=1,dx0=0,kP=3,kI=0,kD=1)
    plotarrows(res)
elif plot == 'pi':
    #damped P / PI testing with and without bias
    #res = PID_trajectory(0,-3,0,1,x0=1,dx0=0,kP=3,kI=0,kD=0)
    #res = PID_trajectory(0,-3,0,1,x0=1,dx0=0,kP=1,kI=0,kD=0)
    #res = PID_trajectory(0,-3,1,1,x0=1,dx0=0,kP=3,kI=0,kD=0)
    res = PID_trajectory(0,-3,1,1,x0=1,dx0=0,kP=3,kI=0.5,kD=0)
    #plt.plot(res['t'],res['I'],label='I')
    #plt.show()
    #plt.ylim(-0.2,1.2)
    plotarrows(res)

