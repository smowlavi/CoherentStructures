"""
An example demonstrating the computation of elliptic LCSs in a 2D system, 
the Bickley jet. This script, which identifies the elliptic LCSs by visualizing 
the clusters of particles that they enclose, would typically be run after the 
right values for minPts, eps and num_meaningful_clusters have been found
by running elliptic_Bickley_sweep.py.
"""

import numpy as np
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import scipy.io as sio
import os

from functions.elliptic import PairwiseDist_2D, EllipticLCS, Colors, RemoveSpuriousClusters

if __name__ == "__main__":

    SavePath = './data/'
    SaveName = 'elliptic_Bickley'

    # Parameters
    minPts = 10
    eps = 2.0
    num_meaningful_clusters = 7

    # Load data
    data = sio.loadmat(SavePath+SaveName)
    xP, yP, tv = data['xP'], data['yP'], data['tv'].flatten()

    # Check if pairwise distances have already been computed
    if os.path.isfile(SavePath+SaveName+'_Dij.mat'):
        
        # Load precomputed pairwise distances
        data = sio.loadmat(SavePath+SaveName+'_Dij')
        Dij = squareform(data['Dij'].flatten())
    
    else:

        # Compute pairwise distances
        Dij = PairwiseDist_2D(xP,yP,xPeriodicBC=(0.0,20.0))

        # Save the pairwise distances for future use
        sio.savemat(SavePath+SaveName+'_Dij.mat',{'Dij':squareform(Dij)})
        print('Pairwise distances saved in %s' % SavePath+SaveName+'_Dij.mat')

    # Compute cluster labels
    labels = EllipticLCS(Dij,minPts,eps)

    # Plot results
    fig = plt.figure(figsize=[7,3])
    ax = plt.gca()
    sc = ax.scatter(xP[:,0],yP[:,0],s=6,c=Colors(RemoveSpuriousClusters(labels,num_meaningful_clusters)))
    ax.set_xlim([0,20])
    ax.set_ylim([-3,3])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('DBSCAN clustering from t0 = %g to tf = %g with minPts = %g and eps = %g' 
                 % (tv[0],tv[-1],minPts,eps))
    plt.show()