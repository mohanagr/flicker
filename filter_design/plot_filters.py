import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz

def compare_filters():
    up = 10
    half_size = 32
    num_taps = 2 * half_size * up + 1
    cutoff = 1 / up  # 0.1

    # Filter 1: Sharp Kaiser (beta=1)
    h_kaiser = firwin(num_taps, cutoff, window=("kaiser", 1), scale=True)

    # Filter 2: Smooth Hamming
    h_hamming = firwin(num_taps, cutoff, window="hamming", scale=True)

    # Calculate frequency responses
    w1, h1 = freqz(h_kaiser, worN=8192)
    w2, h2 = freqz(h_hamming, worN=8192)

    plt.figure(figsize=(12, 6))

    # Plot Magnitude in dB
    plt.subplot(1, 2, 1)
    plt.semilogx(w1 / np.pi, 20 * np.log10(np.abs(h1) + 1e-12), label='Kaiser (beta=1)')
    plt.semilogx(w2 / np.pi, 20 * np.log10(np.abs(h2) + 1e-12), label='Hamming')
    plt.axvline(cutoff, color='r', linestyle='--', alpha=0.5, label='Cutoff (0.1)')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.xlabel('Normalized Frequency (x Nyquist)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Log-Freq Magnitude Response')
    plt.legend()
    plt.ylim([-100, 5])
    plt.xlim([0.01, 1])

    # Plot Power (Linear Magnitude Squared)
    plt.subplot(1, 2, 2)
    plt.loglog(w1 / np.pi, np.abs(h1)**2, label='Kaiser (beta=1)')
    plt.loglog(w2 / np.pi, np.abs(h2)**2, label='Hamming')
    plt.axvline(cutoff, color='r', linestyle='--', alpha=0.5, label='Cutoff (0.1)')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.xlabel('Normalized Frequency (x Nyquist)')
    plt.ylabel('Power (abs^2)')
    plt.title('Log-Log Power Response')
    plt.legend()
    plt.ylim([1e-10, 2])
    plt.xlim([0.01, 1])

    plt.tight_layout()
    plt.savefig('filter_comparison.png')
    print("Plot saved to filter_comparison.png")

def visualize_aliasing_and_overlap():
    up = 10
    half_size = 32
    num_taps = 2 * half_size * up + 1
    cutoff = 1 / up  # 0.1 normalized frequency at high rate

    # Decade 1 (Low) Parameters
    f2_low = 0.5  # Nyquist of low rate
    f1_low = 0.05 # Original lower bound (high rate)
    f1_ext = f1_low * 0.9 # 10% further left = 0.045
    
    # Hamming Filter Response
    h_hamming = firwin(num_taps, cutoff, window="hamming", scale=True)
    w, h = freqz(h_hamming, worN=8192)
    freq_high = w / np.pi # Normalized to high-rate Nyquist (0 to 1)
    mag_sq = np.abs(h)**2

    plt.figure(figsize=(12, 8))

    # 1. Plot Filter and Aliasing
    plt.subplot(2, 1, 1)
    plt.loglog(freq_high, mag_sq, label='Hamming Filter (Power)', color='black', lw=2)
    
    # Shade the baseband region
    plt.fill_between(freq_high, 1e-10, mag_sq, where=(freq_high <= 0.1), alpha=0.1, color='blue', label='Baseband Image')
    
    # Show Aliasing regions (images of the baseband signal)
    # The upsampling process places images at k*fs_low. fs_low = 0.2 in high-rate units.
    for i in range(1, 5):
        alias_center = i * 0.2
        plt.axvspan(alias_center - 0.1, alias_center + 0.1, color='red', alpha=0.05)
        # Plot the "leaking" power from the first alias
        # The filter suppresses the alias at 0.2 - f
        plt.text(alias_center, 1e-8, f'Alias {i}', ha='center', color='red', fontsize=8)

    plt.axvline(0.1, color='red', linestyle='--', label='Upsampling Nyquist (0.1)')
    plt.axvline(0.05, color='green', linestyle=':', label='Decade Boundary (0.05)')
    
    plt.ylim([1e-10, 2])
    plt.xlim([0.01, 1])
    plt.ylabel('Filter Power Response')
    plt.title('Filter Response and Aliasing Regions (High Rate)')
    plt.legend(loc='lower left')
    plt.grid(True, which="both", alpha=0.2)

    # 2. Plot Overlap Regions for Two Decades
    plt.subplot(2, 1, 2)
    
    # Decade 1 (Upsampled): 0.0045 to 0.05
    d1_f = np.logspace(np.log10(0.0045), np.log10(0.05), 100)
    plt.fill_between(d1_f, 0, 1, alpha=0.3, color='blue', label='Decade 1 (Upsampled)')
    
    # Decade 2 (Original): 0.045 to 0.5
    d2_f = np.logspace(np.log10(0.045), np.log10(0.5), 100)
    plt.fill_between(d2_f, 0, 1, alpha=0.3, color='orange', label='Decade 2 (Direct)')

    # Highlight Overlap: 0.045 to 0.05
    plt.axvspan(0.045, 0.05, color='purple', alpha=0.2, label='Overlap Region (10%)')
    
    plt.xscale('log')
    plt.xlim([0.01, 1])
    plt.ylim([0, 1.5])
    plt.xlabel('Normalized Frequency (x High-Rate Nyquist)')
    plt.ylabel('Signal Presence')
    plt.title('Decade Overlap Visualization (10% Extension)')
    plt.legend()
    plt.grid(True, which="both", alpha=0.2)

    plt.tight_layout()
    plt.savefig('aliasing_overlap.png')
    print("Plot saved to aliasing_overlap.png")

if __name__ == "__main__":
    compare_filters()
    visualize_aliasing_and_overlap()
    plt.show()
