"""
An example demonstrating the computation of elliptic LCSs in a 2D system, 
the Bickley jet. Here, we evaluate the number of particles in each of the
num_tracked_clusters largest groups identified by the algorithm as a function 
of the parameter eps, for a given value of minPts. This provides a basis for 
selecting the right values of minPts, eps, as well as the number of 
physically-meaningful groups (set by the parameter num_meaningful_clusters in 
elliptic_Bickley.py).
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from tqdm import tqdm

from functions.elliptic import PairwiseDist_2D, EllipticLCS, Colors, RemoveSpuriousClusters

if __name__ == "__main__":

    SavePath = './data/'
    SaveName = 'Bickley_NP1080_T40'

    # Parameters
    minPts = 10
    eps_values = np.arange(0.1,5,0.1) # values of eps to sweep through
    num_tracked_clusters = 10 # number of clusters to track

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


    cluster_sizes = np.zeros((len(eps_values),num_tracked_clusters))
    for i,eps in enumerate(tqdm(eps_values)):
        # Compute cluster labels
        labels = EllipticLCS(Dij,minPts,eps)

        # Compute and rank the size (number of particles) of each cluster
        freqs = pd.Series(labels).value_counts()

        # Retrieve the number of clusters (exclusing noise)
        num_clusters = len(freqs[freqs.index!=-1].values)

        # Store the size of the num_tracked_clusters largest clusters
        num = np.min([num_tracked_clusters,num_clusters])
        cluster_sizes[i,:num] = freqs[freqs.index!=-1].values[:num]

    
    # Plot results
    fig = plt.figure(figsize=[5,4])
    ax = plt.gca()
    ax.plot(eps_values,cluster_sizes)
    ax.set_yscale('log')
    ax.set_xlabel('$eps$')
    ax.set_ylabel('$N_i$')  
    leg = ax.legend(['%g'%(i+1) for i in range(len(eps_values))],labelspacing=.2,loc='lower right')
    ax.set_title('$\mathtt{minPts} = %g$'%minPts)
    ax.set_title('DBSCAN clustering from t0 = %g to tf = %g \nwith minPts = %g' 
                 % (tv[0],tv[-1],minPts))
    plt.show()