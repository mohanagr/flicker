import numpy as np
from scipy.signal import welch, resample_poly
import matplotlib.pyplot as plt

def get_psd(y, nperseg=1024):
    f, P = welch(y, nperseg=nperseg, noverlap=nperseg // 2)
    return f, P

def reproduce():
    up = 10
    f2 = 0.5
    f1 = f2 / up  # 0.05
    N_low = 100000
    
    # Generate y1
    spec1 = np.zeros(N_low // 2 + 1, dtype='complex128')
    idx1_start = int(f1 * N_low)
    idx1_end = int(f2 * N_low)
    size1 = idx1_end + 1 - idx1_start
    spec1[idx1_start:idx1_end+1] = np.random.randn(size1) + 1j * np.random.randn(size1)
    y1 = np.fft.irfft(spec1)
    
    # Upsample y1
    y1_up = resample_poly(y1, up=up, down=1, window=('kaiser', 1))
    
    # Generate y2
    N_high = len(y1_up)
    spec2 = np.zeros(N_high // 2 + 1, dtype='complex128')
    idx2_start = int(f1 * N_high) # This is 0.05 * N_high
    idx2_end = int(f2 * N_high)
    size2 = idx2_end + 1 - idx2_start
    spec2[idx2_start:idx2_end+1] = np.random.randn(size2) + 1j * np.random.randn(size2)
    y2 = np.fft.irfft(spec2)
    
    # Combine
    # In simple.py it was y1_up + 10*y2. 
    # Let's scale y2 so they have similar power levels at the boundary to see the stitch.
    # y1 was generated with N_low points, y2 with N_high points.
    # irfft scaling is 1/N.
    # Power of y1 is roughly size1 / N_low^2.
    # Power of y1_up (with default resample_poly) is roughly same as y1.
    # Power of y2 is roughly size2 / N_high^2.
    # size2 = 10 * size1. N_high = 10 * N_low.
    # So power of y2 is roughly (10 * size1) / (100 * N_low^2) = 0.1 * power of y1.
    # Thus 10*y2 makes y2's power 100 * 0.1 = 10 times y1's power?
    # Wait, simple.py has 10*y2.
    
    ynew = y1_up + 10 * y2
    
    f, P = get_psd(ynew, nperseg=4096)
    
    # The boundary is at f = 0.05
    boundary_idx = np.argmin(np.abs(f - 0.05))
    print(f"Boundary frequency: {f[boundary_idx]}")
    print(f"PSD around boundary: {P[boundary_idx-5:boundary_idx+6]}")
    
    # Check for a spike: ratio of boundary PSD to neighbors
    avg_neighbor = (P[boundary_idx-1] + P[boundary_idx+1]) / 2
    spike_ratio = P[boundary_idx] / avg_neighbor
    print(f"Spike ratio: {spike_ratio}")

if __name__ == "__main__":
    reproduce()
