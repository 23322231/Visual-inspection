import numpy as np

def wrap(array):
    return np.roll(array, np.floor(np.array(array.shape) / 2).astype(int)*(-1),axis=(1,0))