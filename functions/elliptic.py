import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from matplotlib import cm
from matplotlib.colors import to_rgb
import seaborn as sns
from numba import njit

@njit
def PairwiseDist_2D(xP,yP,xPeriodicBC=None,yPeriodicBC=None):
    """
    Computes the pairwise distances for the 2D trajectory dataset

    Args:
        xP, yP: 2D arrays
            Coordinates of particle trajectories, with lines representing
            different particles and columns representing different times
        xPeriodicBC, yPeriodicBC, zPeriodicBC: tuples, optional
            Tuples of the form (min,max) indicating the extent of each 
            periodic direction, so that the periodicity is taken into 
            account when computing pairwise distances. If a direction is 
            not periodic, leave the corresponding argument unspecified

    Returns:
        Dij: 2D array
            Symmetric distance matrix storing the pairwise distances

    """

    # Number of particles
    NP = xP.shape[0]

    # Compute symmetrix matrix of pairwise distances
    Dij = np.zeros((NP,NP))
    for i in range(NP):
        if i%10==0:
            print('Progress:',(i+1)/NP*100,'%')

        for j in range(i):
            dxP = xP[i,:]-xP[j,:]
            dyP = yP[i,:]-yP[j,:]

            if xPeriodicBC is not None:
                x_min, x_max = xPeriodicBC
                dxP1 = np.mod((xP[i,:]-x_min),(x_max-x_min)) - np.mod((xP[j,:]-x_min),(x_max-x_min))
                dxP = np.minimum(np.abs(dxP1), (x_max-x_min)-np.abs(dxP1))
            if yPeriodicBC is not None:
                y_min, y_max = yPeriodicBC
                dyP1 = np.mod((yP[i,:]-y_min),(y_max-y_min)) - np.mod((yP[j,:]-y_min),(y_max-y_min))
                dyP = np.minimum(np.abs(dyP1), (y_max-y_min)-np.abs(dyP1))

            dist = np.sqrt(np.sum(np.stack((dxP,dyP))**2,axis=0))
            Dij[i,j] = np.mean(dist)

    for i in range(NP):
        for j in range(i,NP):
            Dij[i,j] = Dij[j,i]

    return Dij

@njit
def PairwiseDist_3D(xP,yP,zP,
                    xPeriodicBC=None,yPeriodicBC=None,zPeriodicBC=None):
    """
    Computes the pairwise distances for the 3D trajectory dataset

    Args:
        xP, yP, zP: 2D arrays
            Coordinates of particle trajectories, with lines representing
            different particles and columns representing different times
        xPeriodicBC, yPeriodicBC, zPeriodicBC: tuples, optional
            Tuples of the form (min,max) indicating the extent of each 
            periodic direction, so that the periodicity is taken into 
            account when computing pairwise distances. If a direction is 
            not periodic, leave the corresponding argument unspecified
    Returns:
        Dij: 2D array
            Symmetric distance matrix storing the pairwise distances

    """

    # Number of particles
    NP = xP.shape[0]

    # Compute symmetrix matrix of pairwise distances
    Dij = np.zeros((NP,NP))
    for i in range(NP):
        if i%10==0:
            print('Progress:',(i+1)/NP*100,'%')
        
        for j in range(i):
            dxP = xP[i,:]-xP[j,:]
            dyP = yP[i,:]-yP[j,:]
            dzP = zP[i,:]-zP[j,:]

            if xPeriodicBC is not None:
                x_min, x_max = xPeriodicBC
                dxP1 = np.mod((xP[i,:]-x_min),(x_max-x_min)) - np.mod((xP[j,:]-x_min),(x_max-x_min))
                dxP = np.minimum(np.abs(dxP1), (x_max-x_min)-np.abs(dxP1))
            if yPeriodicBC is not None:
                y_min, y_max = yPeriodicBC
                dyP1 = np.mod((yP[i,:]-y_min),(y_max-y_min)) - np.mod((yP[j,:]-y_min),(y_max-y_min))
                dyP = np.minimum(np.abs(dyP1), (y_max-y_min)-np.abs(dyP1))
            if zPeriodicBC is not None:
                z_min, z_max = zPeriodicBC
                dzP1 = np.mod((zP[i,:]-z_min),(z_max-z_min)) - np.mod((zP[j,:]-z_min),(z_max-z_min))
                dzP = np.minimum(np.abs(dzP1), (z_max-z_min)-np.abs(dzP1))

            dist = np.sqrt(np.sum(np.stack((dxP,dyP,dzP))**2,axis=0))
            Dij[i,j] = np.mean(dist)

    for i in range(NP):
        for j in range(i,NP):
            Dij[i,j] = Dij[j,i]

    return Dij

def EllipticLCS(Dij,minPts,eps):
    """
    Given pairwise distances, computes elliptic LCSs through DBSCAN clustering

    Args:
        Dij: 2D array
            Symmetric distance matrix storing the pairwise distances
        minPts: scalar
            Parameters minPts, also called min_samples, of the DBSCAN 
            algorithm (refer to paper for more details)
        eps: scalar, optional
            Parameters eps of the DBSCAN algorithm (refer to paper for 
            more details)

    Returns:
        labels: 1D array
            Cluster labels for every particle; those belonging to noise are
            assigned the label -1
    """

    # Compute DBSCAN clustering of the particles
    db = DBSCAN(eps=eps,min_samples=minPts,metric='precomputed').fit(Dij)

    return db.labels_

def Colors(labels,cmap='tab10'):
    """
    This function assigns to each label a color taken from the chosen
    colormap according to the ranked size of the corresponding group
    (that is, the group with the most particles will always be assigned
    the first color in the colormap, etc). Particles belonging to noise
    (i.e. with label -1) are always assigned a light gray color.

    Args:
        labels: 1D array
            Cluster labels for every particle; with label -1 indicating
            that the particle belongs to noise
        cmap: string
            Name of a Matplotlib colormap

    Returns:
        colors: list of tuples
            List of tuples specifying the RGB values of the color assigned 
            to each particle
    """

    # Compute the number of particles belonging to each label
    freqs = pd.Series(labels).value_counts()

    # Rank the labels according to the number of corresponding particles
    sorted_labels = freqs[freqs.index!=-1].index.tolist() + ([-1] if -1 in freqs.index.tolist() else []) 
    
    # Create the corresponding sequence of colors
    sorted_colors = sns.color_palette(palette=cmap,n_colors=len(sorted_labels))
    if -1 in sorted_labels:
        sorted_colors[-1] = to_rgb('lightgray')

    # Creating a mapping between label value and color index
    unique_labels = {val:key for key,val in enumerate(sorted_labels)}

    return [sorted_colors[unique_labels[lbl]] for lbl in labels]

def RemoveSpuriousClusters(labels,num_clusters):
    """
    This function removes spurious small clusters by replacing their
    label with the value -1, so that corresponding particles are
    assigned to noise.

    Args:
        labels: 1D array
            Cluster labels for every particle; with label -1 indicating
            that the particle belongs to noise
        num_clusters: int
            Number of cluster to keep, ranked according to the number of
            consitutent particles. All clusters smaller than the num_clusters
            largest clusters will be assigned to noise by replacing their
            labels with the value -1
    Returns:
        updated_labels: 1D array
            Updated cluster labels for every particle
    """

    # Compute the number of particles belonging to each label
    plabels = pd.Series(labels) 
    freqs = pd.Series(labels).value_counts()

    # Rank the (non-noise) labels according to the number of corresponding particles
    sorted_labels = freqs[freqs.index!=-1].index.tolist()

    # Replace the labels of small spurious clusters with the value -1
    plabels = plabels.mask(np.isin(labels,sorted_labels[num_clusters:]), -1)

    return plabels.values