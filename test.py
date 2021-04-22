import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_rgba
import scipy.io as sio
import os

SavePath = './data/'
SaveName = 'Bickley_NP1080_T40'

data = sio.loadmat(SavePath+SaveName)
xP = data['xP']
print(xP[:5,-1])

data = sio.loadmat(SavePath+SaveName+'_Dij')
Dij = data['Dij'].flatten()

SavePath = './data/'
SaveName = 'RegIC_NP1080_T40_NoiseStd0'

data = sio.loadmat(SavePath+SaveName)
Dij_old = data['Dij_cond'].flatten()

print(Dij[0:5])
print(Dij_old[0:5])