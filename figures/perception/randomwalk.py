import matplotlib.pyplot as plt
import numpy as np
from kalman import *

def kf_trace(F,g,P,H,j,Q,Xmean,Xvar,Z):
    if not isinstance(F,np.ndarray): F = np.array([[F]])
    if not isinstance(g,np.ndarray): g = np.array([g])
    if not isinstance(P,np.ndarray): P = np.array([[P]])
    if H is not None:
        if not isinstance(H,np.ndarray): H = np.array([[H]])
        if not isinstance(j,np.ndarray): j = np.array([j])
        if not isinstance(Q,np.ndarray): Q = np.array([[Q]])
    if not isinstance(Xmean,np.ndarray): Xmean = np.array([Xmean])
    if not isinstance(Xvar,np.ndarray): Xvar = np.array([[Xvar]])
    cur_mean,cur_cov = Xmean,Xvar
    res_mean = [cur_mean]
    res_cov = [cur_cov]
    for z in Z:
        if not isinstance(z,np.ndarray): z = np.array([z])
        cur_mean,cur_cov = kalman_filter_predict(cur_mean,cur_cov,F,g,P)
        if H is not None:
            cur_mean,cur_cov = kalman_filter_update(cur_mean,cur_cov,F,g,P,H,j,Q,z)
        res_mean.append(cur_mean)
        res_cov.append(cur_cov)
    return res_mean,res_cov

T = 100
N = 20
dt = 0.1
motion_noise_magnitude = 1.0
noise_magnitude = 0.3

fig1 = plt.figure(figsize=(10,4))
ax1 = fig1.add_subplot(1, 2, 1)
ax1.set_xlabel("Time")
ax1.set_ylabel("State")
ax1.set_ylim(-3,3)
ax1.set_xlim(0,10)
x = np.array(range(T))*dt
for i in xrange(N):
    eps = np.random.normal(size=T)*motion_noise_magnitude
    y = np.cumsum(eps*dt)
    ax1.plot(x,y)
y,yvar = kf_trace(F=1,g=0,P=motion_noise_magnitude*dt**2,H=None,j=None,Q=noise_magnitude**2,Xmean=0,Xvar=0,Z=eps)
y = np.array([yi[0] for yi in y])
yvar = np.array([yi[0,0] for yi in yvar])
kf_pred, = ax1.plot(x,y[:-1],label="KF prediction")
ax1.plot(x,y[:-1]+2.0*np.sqrt(yvar)[:-1],label="KF prediction + 2*std",lw=0.5,color='k',linestyle='--')
ax1.plot(x,y[:-1]-2.0*np.sqrt(yvar)[:-1],label="KF prediction + 2*std",lw=0.5,color='k',linestyle='--')
ax1.legend(handles=[kf_pred])

ax2 = fig1.add_subplot(1, 2, 2)
ax2.set_xlabel("Time")
ax2.set_ylabel("State")
ax2.set_ylim(-3,3)
ax2.set_xlim(0,10)
#eps_truth = np.random.normal(size=T)
#y_truth = np.cumsum(eps*dt)
y_truth = np.sin(np.array(range(T))*dt*0.5)*1.0
x = np.array(range(T))*dt
z = y_truth + np.random.normal(size=T)*noise_magnitude
y,yvar = kf_trace(F=1,g=0,P=motion_noise_magnitude*dt**2,H=1,j=0,Q=noise_magnitude**2,Xmean=0,Xvar=0,Z=z)
y = np.array([yi[0] for yi in y])
yvar = np.array([yi[0,0] for yi in yvar])
Zmse = np.sqrt(np.sum((z-y_truth)**2))
KFmse = np.sqrt(np.sum((y[:-1]-y_truth)**2))
print "Z MSE",Zmse
print "KF MSE",KFmse
print "Reduction (%)",(Zmse-KFmse)/Zmse*100
ground_truth, = ax2.plot(x,y_truth,label="Ground truth",color='k')
obs = ax2.scatter(x,z,label="Observations",color='gray',s=9)
kf_estimate, = ax2.plot(x,y[:-1],label="KF estimate")
ax2.plot(x,y[:-1]+2.0*np.sqrt(yvar)[:-1],label="KF estimate + 2*std",lw=0.5,color='k',linestyle='--')
ax2.plot(x,y[:-1]-2.0*np.sqrt(yvar)[:-1],label="KF estimate + 2*std",lw=0.5,color='k',linestyle='--')
ax2.legend(handles=[ground_truth,obs,kf_estimate])
plt.show()
