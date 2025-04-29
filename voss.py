import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

def voss_mccartney_alpha(n_samples, n_rows, alpha):
    """
    Generate 1/f^alpha noise via a weighted Voss–McCartney algorithm.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_rows : int
        Number of octave rows (≈ log2(nyquist/min_freq)).
    alpha : float
        Desired spectral slope: S(f) ∝ 1/f^alpha.
    
    Returns
    -------
    y : np.ndarray
        The generated time series, normalized to unit variance.
    """
    # precompute per-row scales so var(row[k]) ∝ 2^{-k(1−α)}
    beta  = 0.5 * (1.0 - alpha)
    scale = 2.0 ** (-beta * np.arange(n_rows))
    
    rows  = np.zeros(n_rows)
    total = 0.0
    y     = np.empty(n_samples)
    
    for i in range(n_samples):
        n = i + 1
        k = (n & -n).bit_length() - 1
        if k >= n_rows:
            k = n_rows - 1
        
        new = scale[k] * (2.0 * np.random.rand() - 1.0)
        total += new - rows[k]
        rows[k] = new
        y[i]    = total
    
    return y / np.std(y)


if __name__ == "__main__":
    N      = 2000000
    ROWS   = 16
    ALPHA1  = 3.7        # try 0.5 (pink), 1.0, 1.7, 2.0 (brown), etc.
    ALPHA2  = 1        # try 0.5 (pink), 1.0, 1.7, 2.0 (brown), etc.
    
    y1 = voss_mccartney_alpha(N, ROWS, ALPHA1)
    f1, Pxx1 = welch(y1, fs=1.0, nperseg=2**14)
    # quick PSD check
    y2 = voss_mccartney_alpha(N, ROWS, ALPHA2)
    f2, Pxx2 = welch(y2, fs=1.0, nperseg=2**14)
    
    
    plt.figure()
    plt.loglog(f1[1:], Pxx1[1:], label=f"α={ALPHA1}")
    plt.loglog(f2[1:], Pxx2[1:], label=f"α={ALPHA2}")
    plt.title(f"Voss–McCartney 1/f^{{α}} Noise")
    plt.xlabel("Normalized Frequency")
    plt.ylabel("PSD")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.show()