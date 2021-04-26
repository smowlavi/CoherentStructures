# CoherentStructures

This code accompanies the paper *Coherent structures in sparse and noisy data* by Mowlavi, Serra, Maiorino, and Mahadevan (2021).

We provide algorithms for the identification of Lagrangian Coherent Structures (LCSs) of hyperbolic and elliptic nature -- see figure -- in flows characterized by sparse and noisy trajectory datasets, such as those obtained from experiments. Hyperbolic LCSs are surfaces along which the local separation rate between neighboring particles is maximized or minimized. Elliptic LCSs are surfaces enclosing regions of coherent global dynamics, that is, regions inside of
which particles move together over time.

The algorithms, which take as input trajectory data for an ensemble of particles, are located in the Python modules ./functions/hyperbolic.py and ./functions/elliptic.py. Their use is demonstrated through two of the examples shown in the paper, the Bickley jet and ABC flow.

![sketch](./sketch.png)

## Main files

* **compute_force** computes the force between two elastic bodies (see Figure 1) for a given orientation of the bodies and contact normal direction. This script is intended as a demonstration of how the contact force would be implemented in a DEM code.

## Dependencies

[Numba](https://numba.pydata.org): Used to accelerate the computation of the pairwise distance matrix.