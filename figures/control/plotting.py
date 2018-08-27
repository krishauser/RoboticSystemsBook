import matplotlib.pyplot as plt
import numpy as np

def plotxy(trajs,names,which=['x'],indices=[0,1],colors=None,linestyles=None):
    """Plots one or more trajectories on the x,y plane"""
    if linestyles is None:
        linestyles = ['-', '--', ':', '-.']
    plots = []
    labels = []
    for i,(res,name,item) in enumerate(zip(trajs,names,which)):
        color = None if colors is None else colors[i]
        linestyle = None if linestyles is None else linestyles[i]
        x=np.array([v[indices[0]] for v in res[item]])
        y=np.array([v[indices[1]] for v in res[item]])
        pos = np.where(np.abs(np.diff(y)) >= 5)[0]+1
        if len(pos) > 0:
            x[pos]=np.nan
            y[pos]=np.nan
        plot,=plt.plot(x,y,label=item,color=color,linestyle=linestyle)
        plots.append(plot)
        labels.append(name)
    plt.legend(labels=labels,handles=plots)

def plotmulti(trajs,names,which=['x'],indices=[0],colors=None,linestyles=None):
    """Plots one or more trajectories as a function of time"""
    if linestyles is None:
        linestyles = ['-', '--', ':', '-.']
    plots = []
    labels = []
    for i,(res,name,item) in enumerate(zip(trajs,names,which)):
        color = None if colors is None else colors[i]
        linestyle = None if linestyles is None else linestyles[i]
        x = np.array(res['t'][:])
        if indices[i] is None:
            y = np.array(res[item])
        else:
            y = np.array([v[indices[i]] for v in res[item]])
        pos = np.where(np.abs(np.diff(y)) >= 5)[0]+1
        if len(pos) > 0:
            x[pos]=np.nan
            y[pos]=np.nan
        plot,=plt.plot(x,y,label=item,color=color,linestyle=linestyle)
        plots.append(plot)
        labels.append(name)
    plt.legend(labels=labels,handles=plots)

def plotflow(xbound,ybound,N,funcs,colorfunc=None,**kwargs):
    if not hasattr(funcs,'__iter__'):
        return plotflow(xbound,ybound,N,[funcs],colorfunc)
    if not hasattr(colorfunc,'__iter__'):
        colorfunc = [colorfunc]*len(funcs)
    Y, X = np.mgrid[ybound[0]:ybound[1]:(N*1j),xbound[0]:xbound[1]:(N*1j)]
    fig0, ax0 = plt.subplots()

    for i,func in enumerate(funcs):
        U,V = zip(*[func(v) for v in zip(X.reshape(-1),Y.reshape(-1))])
        U = np.array(U).reshape(X.shape)
        V = np.array(V).reshape(X.shape)
        speed = np.sqrt(U**2 + V**2)
        if colorfunc is None:
            color = speed
        else:
            color = np.array([colorfunc[i]([v[0],v[1]],[v[2],v[3]]) for v in zip(X.reshape(-1),Y.reshape(-1),U.reshape(-1),V.reshape(-1))])
            color = color.reshape(X.shape)

        strm = ax0.streamplot(X, Y, U, V, color=color, **kwargs)
    fig0.colorbar(strm.lines)
