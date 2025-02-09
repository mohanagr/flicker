import numba as nb
import numpy as np

@nb.njit()
def test():
    stack = np.zeros(10,dtype='int64')
    x=2;y=1