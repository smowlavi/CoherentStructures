"""
An example demonstrating the computation of elliptic LCSs in a 3D system, 
the ABC flow.
"""

import numpy as np
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import scipy.io as sio
import os

from functions.elliptic import PairwiseDist_3D, EllipticLCS, Colors, RemoveSpuriousClusters

if __name__ == "__main__":

    SavePath = './data/'
    SaveName = 'ABC_NP13824_T20'

    # Parameters
    minPts = 25
    eps = 1.1
    num_meaningful_clusters = 6

    # Load data
    data = sio.loadmat(SavePath+SaveName)
    xP, yP, zP, tv = data['xP'], data['yP'], data['zP'], data['tv'].flatten()

    # Check if pairwise distances have already been computed
    if os.path.isfile(SavePath+SaveName+'_Dij.mat'):
        
        # Load precomputed pairwise distances
        data = sio.loadmat(SavePath+SaveName+'_Dij')
        Dij = squareform(data['Dij'].flatten())
    
    else:

        # Compute pairwise distances
        Dij = PairwiseDist_3D(xP,yP,zP,
                              xPeriodicBC=(0.0,2*np.pi),
                              yPeriodicBC=(0.0,2*np.pi),
                              zPeriodicBC=(0.0,2*np.pi))

        # Save the pairwise distances for future use
        sio.savemat(SavePath+SaveName+'_Dij.mat',{'Dij':squareform(Dij)})
        print('Pairwise distances saved in %s' % SavePath+SaveName+'_Dij.mat')

    # Compute cluster labels
    labels = EllipticLCS(Dij,minPts,eps)

    # Plot results
    fig = plt.figure(figsize=[5,4])
    ax = plt.axes(projection='3d',azim=230,elev=30)
    sc = ax.scatter3D(xP[:,0],yP[:,0],zP[:,0],s=6,
                      c=Colors(RemoveSpuriousClusters(labels,num_meaningful_clusters)))
    ax.use_sticky_edges = True
    ax.set_xlim([0,2*np.pi])
    ax.set_ylim([0,2*np.pi])
    ax.set_zlim([0,2*np.pi])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_ylabel('$z$')
    ax.set_title('DBSCAN clustering from t0 = %g to tf = %g \nwith minPts = %g and eps = %g' 
                 % (tv[0],tv[-1],minPts,eps))
    plt.show()