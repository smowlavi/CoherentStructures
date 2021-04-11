import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from functions import HyperbolicLCS

SavePath = './data/'
SaveName = 'Bickley_NP6000_T40'

# Parameters
eps = 0.5
lam = 1e-10

# Load data
data = sio.loadmat(SavePath+SaveName)
xP, yP, tv = data['xP'], data['yP'], data['tv'].flatten()

# Compute hyperbolic LCS
FTLE = HyperbolicLCS(eps,lam,tv,xP,yP)

# Extract initial and final time snapshots
t0, tf = tv[0], tv[-1]
xP0, xPf = xP[:,0], xP[:,-1]
yP0, yPf = yP[:,0], yP[:,-1]

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