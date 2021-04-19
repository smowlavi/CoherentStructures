"""
An example demonstrating the computation of forward hyperbolic LCSs in a 3D
system, the ABC flow. In order to reduce the computational time, we only 
consider trajectories initialized in slabs of thickness 0.2*pi at the planes
x=0, y=0, and z=2*pi.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from functions import HyperbolicLCS_3D

if __name__ == "__main__":

    SavePath = './data/'
    SaveName = 'ABC_NP100000_T20'

    # Parameters
    direction = 'forward'
    eps = 0.4

    # Load data
    data = sio.loadmat(SavePath+SaveName)
    xP, yP, zP, tv = data['xP'], data['yP'], data['zP'], data['tv'].flatten()

    # Compute forward hyperbolic LCS
    FTLE = HyperbolicLCS_3D(direction,xP,yP,zP,tv,eps)

    # Plot results
    ind = np.flatnonzero((xP[:,0]>=0) & (xP[:,0]<=2*np.pi) & 
                         (yP[:,0]>=0) & (yP[:,0]<=2*np.pi) & 
                         (zP[:,0]>=0) & (zP[:,0]<=2*np.pi))
    fig = plt.figure(figsize=[5,4])
    ax = plt.axes(projection='3d',azim=230,elev=30)
    sc = ax.scatter3D(xP[ind,0],yP[ind,0],zP[ind,0],s=6,c=FTLE[ind],cmap='jet',vmin=0,vmax=0.28)
    fig.colorbar(sc, ax=ax, ticks=np.arange(0,0.28,0.07), extend='both')
    ax.set_xlim([0,2*np.pi])
    ax.set_ylim([0,2*np.pi])
    ax.set_zlim([0,2*np.pi])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_ylabel('$z$')
    ax.set_title('Forward FTLE from t0 = %g to tf = %g' % (tv[0],tv[-1]))
    plt.show()