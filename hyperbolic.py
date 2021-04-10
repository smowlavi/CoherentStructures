import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations

SavePath = './data/'
SaveName = 'Bickley_NP6000_T40'

# Parameters
eps = 0.5
lam = 1e-10

# Load data
data = sio.loadmat(SavePath+SaveName)
xP, yP, tv = data['xP'], data['yP'], data['tv'].flatten()
NP, tf, t0 = xP.shape[0], tv[-1], tv[0]
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

# Plot results
ind = np.flatnonzero((xP0>=0) & (xP0<=20) & (yP0>=-3) & (yP0<=3))
fig = plt.figure(figsize=[6,3])
ax = plt.gca()
sc = ax.scatter(xP0[ind],yP0[ind],s=6,c=FTLE[ind],cmap='jet',vmin=0,vmax=0.16)
ax.set_xlim([0,20])
ax.set_ylim([-3,3])
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Forward FTLE from t0 = %g to tf = %g' % (t0,tf))
plt.show()