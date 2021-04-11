import numpy as np
from tqdm import tqdm
from itertools import combinations

def HyperbolicLCS(eps,lam,tv,xP,yP):

    # Number of particles
    NP = xP.shape[0]
    tf, t0 = tv[-1], tv[0]
    T = tf-t0

    # Extract initial and final time snapshots
    xP0, yP0 = xP[:,0], yP[:,0]
    xPf, yPf = xP[:,-1], yP[:,-1]

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