import numpy as np
from tqdm import tqdm
from itertools import combinations

def HyperbolicLCS_2D(direction,xP,yP,tv,eps,lam=1e-10):
    """
    Computes hyperbolic LCSs in 3D trajectory datasets

    Args:
        direction: string
            Type of hyperbolic LCS, 'forward' or 'backward'
        xP, yP: 2D arrays
            Coordinates of particle trajectories, with lines representing
            different particles and columns representing different times
        tv: 1D array
            Times values corresponding to the columns of xP, yP
        eps: scalar
            Neighborhood radius for particles to feed in the least squares
            algorithm (refer to paper for more details)
        lam: scalar, optional
            Weight of regularization term in the least squares objective
            function (refer to paper for more details)

    Returns:
        FTLE: 1D array
            FTLE values at each particle location; the forward
            and backward FTLE are displayed at the beginning and end of the 
            time window, respectively
    """

    # Number of particles
    NP = xP.shape[0]
    tf, t0 = tv[-1], tv[0]
    T = tf-t0

    # Extract initial and final time snapshots
    if direction == 'forward':
        xP0, yP0 = xP[:,0], yP[:,0]
        xPf, yPf = xP[:,-1], yP[:,-1]
    elif direction == 'backward':
        xP0, yP0 = xP[:,-1], yP[:,-1]
        xPf, yPf = xP[:,0], yP[:,0]

    # Calculate FTLE values at each particle
    FTLE = np.zeros(NP)
    for i in tqdm(range(NP)):
        # Compute initial distances
        dxP = xP0-xP0[i]
        dyP = yP0-yP0[i]
        
        # Find pairwise combinations of neighbor indices
        neighbors = np.flatnonzero((dxP**2+dyP**2)<eps**2)
        combs = list(combinations(range(len(neighbors)),2))
        ind1 = [comb[0] for comb in combs]
        ind2 = [comb[1] for comb in combs]
        
        # Form X and Y data matrices
        X = np.zeros((2,len(combs)))
        Y = np.zeros((2,len(combs)))
        X[0,:] = xP0[neighbors[ind1]]-xP0[neighbors[ind2]]
        X[1,:] = yP0[neighbors[ind1]]-yP0[neighbors[ind2]]
        Y[0,:] = xPf[neighbors[ind1]]-xPf[neighbors[ind2]]
        Y[1,:] = yPf[neighbors[ind1]]-yPf[neighbors[ind2]]
        # Least square fit of flow map gradient
        A = Y@X.T + lam*max(1,len(neighbors))*np.eye(2)
        B = X@X.T + lam*max(1,len(neighbors))*np.eye(2)
        DF = A@np.linalg.inv(B)
        
        # Calculate FTLE as the largest singular value of DF
        FTLE[i] = np.log(np.linalg.norm(DF,2))/T

    return FTLE

def HyperbolicLCS_3D(type,xP,yP,zP,tv,eps,lam=1e-10):
    """
    Computes hyperbolic LCSs in 3D trajectory datasets

    Args:
        direction: string
            Type of hyperbolic LCS, 'forward' or 'backward'
        xP, yP, zP: 2D arrays
            Coordinates of particle trajectories, with lines representing
            different particles and columns representing different times
        tv: 1D array
            Times values corresponding to the columns of xP, yP, zP
        eps: scalar
            Neighborhood radius for particles to feed in the least squares
            algorithm (refer to paper for more details)
        lam: scalar, optional
            Weight of regularization term in the least squares objective
            function (refer to paper for more details)

    Returns:
        FTLE: 1D array
            FTLE values at each particle location; the forward
            and backward FTLE are displayed at the beginning and end of the 
            time window, respectively
    """

    # Number of particles
    NP = xP.shape[0]
    tf, t0 = tv[-1], tv[0]
    T = tf-t0

    # Extract initial and final time snapshots
    if type == 'forward':
        xP0, yP0, zP0 = xP[:,0], yP[:,0], zP[:,0]
        xPf, yPf, zPf = xP[:,-1], yP[:,-1], zP[:,-1]
    elif type == 'backward':
        xP0, yP0, zP0 = xP[:,-1], yP[:,-1], zP[:,-1]
        xPf, yPf, zPf = xP[:,0], yP[:,0], zP[:,0]

    # Calculate FTLE values at each particle
    FTLE = np.zeros(NP)
    for i in tqdm(range(NP)):
        # Compute initial distances
        dxP = xP0-xP0[i]
        dyP = yP0-yP0[i]
        dzP = zP0-zP0[i]
        
        # Find pairwise combinations of neighbor indices
        neighbors = np.flatnonzero((dxP**2+dyP**2+dzP**2)<eps**2)
        combs = list(combinations(range(len(neighbors)),2))
        ind1 = [comb[0] for comb in combs]
        ind2 = [comb[1] for comb in combs]
        
        # Form X and Y data matrices
        X = np.zeros((3,len(combs)))
        Y = np.zeros((3,len(combs)))
        X[0,:] = xP0[neighbors[ind1]]-xP0[neighbors[ind2]]
        X[1,:] = yP0[neighbors[ind1]]-yP0[neighbors[ind2]]
        X[2,:] = zP0[neighbors[ind1]]-zP0[neighbors[ind2]]
        Y[0,:] = xPf[neighbors[ind1]]-xPf[neighbors[ind2]]
        Y[1,:] = yPf[neighbors[ind1]]-yPf[neighbors[ind2]]
        Y[2,:] = zPf[neighbors[ind1]]-zPf[neighbors[ind2]]
        # Least square fit of flow map gradient
        A = Y@X.T + lam*max(1,len(neighbors))*np.eye(3)
        B = X@X.T + lam*max(1,len(neighbors))*np.eye(3)
        DF = A@np.linalg.inv(B)
        
        # Calculate FTLE as the largest singular value of DF
        FTLE[i] = np.log(np.linalg.norm(DF,2))/T

    return FTLE