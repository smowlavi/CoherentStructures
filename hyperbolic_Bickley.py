"""
An example demonstrating the computation of forward hyperbolic LCSs in a 2D
system, the Bickley jet.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from functions.hyperbolic import HyperbolicLCS_2D

if __name__ == "__main__":

    SavePath = './data/'
    SaveName = 'hyperbolic_Bickley'

    # Parameters
    direction = 'forward'
    eps = 0.5

    # Load data
    data = sio.loadmat(SavePath+SaveName)
    xP, yP, tv = data['xP'], data['yP'], data['tv'].flatten()

    # Compute forward hyperbolic LCS
    FTLE = HyperbolicLCS_2D(xP,yP,tv,direction,eps)

    # Plot results
    fig = plt.figure(figsize=[6,3])
    ax = plt.gca()
    sc = ax.scatter(xP[:,0],yP[:,0],s=6,c=FTLE,cmap='jet',vmin=0,vmax=0.16)
    fig.colorbar(sc, ax=ax, ticks=np.arange(0,0.16,0.04), extend='both')
    ax.set_xlim([0,20])
    ax.set_ylim([-3,3])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('%s FTLE from t0 = %g to tf = %g with eps = %g' 
                 % (direction,tv[0],tv[-1],eps))
    plt.show()