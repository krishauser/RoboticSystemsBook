import numpy as np

#Kalman filtering code

def gaussian_linear_transform(mean,cov,A,b):
    """Given a prior x~N(mean,cov), returns the
    mean and covariance of the variate y=A*x+b.
    """
    ymean = np.dot(A,mean)+b
    ycov = np.dot(A,np.dot(cov,A.T))
    return (ymean,ycov)

def kalman_filter_predict(prior_mean,prior_cov,F,g,SigmaX):
    """For the Kalman filter model with transition model:
      x[t] = F*x[t-1] + g + eps_x
    with eps_x ~ N(0,SigmaX) 
    and given prior estimate x[t-1]~N(prior_mean,prior_cov),
    computes the predicted mean and covariance matrix at x[t]

    Output:
    - A pair (mu,cov) with mu the updated mean and cov the updated covariance
      matrix.

    Note: all elements are numpy arrays.
    """
    if isinstance(SigmaX,(int,float)):
        SigmaX = np.eye(len(prior_mean))*SigmaX
    muprime = np.dot(F,prior_mean)+g
    covprime = np.dot(F,np.dot(prior_cov,F.T))+SigmaX
    return (muprime,covprime)

def kalman_filter_update(prior_mean,prior_cov,F,g,SigmaX,H,j,SigmaZ,z):
    """For the Kalman filter model with transition model:
      x[t] = F*x[t-1] + g + eps_x
    and observation model
      z[t] = H*x[t] + j + eps_z
    with eps_x ~ N(0,SigmaX) and eps_z ~ N(0,SigmaZ),
    and given prior estimate x[t-1]~N(prior_mean,prior_cov),
    computes the updated mean and covariance matrix after observing z[t]=z.

    Output:
    - A pair (mu,cov) with mu the updated mean and cov the updated covariance
      matrix.

    Note: all elements are numpy arrays.

    Note: can be applied as an approximate extended Kalman filter by setting
    F*x+g and H*x+j to be the linearized models about the current estimate
    prior_mean.  (The true EKF would propagate the mean update and linearize
    the observation term around the mean update)
    """
    if isinstance(SigmaX,(int,float)):
        SigmaX = np.eye(len(prior_mean))*SigmaX
    if isinstance(SigmaZ,(int,float)):
        SigmaZ = np.eye(len(z))*SigmaZ
    muprime = np.dot(F,prior_mean)+g
    covprime = np.dot(F,np.dot(prior_cov,F.T))+SigmaX
    C = np.dot(H,np.dot(covprime,H.T))+SigmaZ
    zpred = np.dot(H,muprime)+j
    K = np.dot(covprime,np.dot(H.T,np.linalg.pinv(C)))
    mu = muprime + np.dot(K,z-zpred)
    cov = np.dot(covprime,np.eye(covprime.shape[0])-np.dot(K,H))
    return (mu,cov)
