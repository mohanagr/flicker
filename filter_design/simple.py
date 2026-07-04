import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz
from scipy.special import sici
import time
import numba as nb
import sys
from scipy.signal import welch, firwin, resample_poly,upfirdn

def plot_spectra(ys, size, both=True):
    fig = plt.gcf()
    fig.set_size_inches(10, 4)
    if both:
        plt.subplot(121)
        plt.title("Stacked FFT PS")
        for y in ys:
            spec = y.reshape(-1, size)
            spec = np.mean(np.abs(np.fft.rfft(spec, axis=1)) ** 2, axis=0)
            plt.loglog(spec)
        plt.subplot(122)
        plt.title("Welch PS w/ windowing & overlap")
        for y in ys:
            f, P = welch(y, nperseg=size, noverlap=size // 2)
            plt.loglog(P)
    else:
        plt.title("Stacked FFT PS")
        for y in ys:
            spec = y.reshape(-1, size)
            spec = np.mean(np.abs(np.fft.rfft(spec, axis=1)) ** 2, axis=0)
            plt.loglog(spec)
    plt.tight_layout()
    plt.show()



up = 10
f2 = 0.5
f1 = f2 / up
cut = 0.3

f1  -= cut/up

N = 2 * 1000000

print(f2 * N, f1 * N)

overlap_right = int(cut * N) #0.4 to 0.5
overlap_left = int(cut / up * N) #0.04 to 0.05
print("overlap samples", overlap_left, overlap_right)

ps_val = np.ones(int(f2*N)+1-int(f1*N))
len_ps = len(ps_val)
if cut > 0:
    #right-edge cosine taper
    ps_val[len_ps - overlap_right:] *= np.cos(np.linspace(0, np.pi/2, overlap_right))**2
    #left-edge sine taper
    ps_val[:overlap_left] *= np.sin(np.linspace(0, np.pi/2, overlap_left))**2

ps=np.zeros(N//2+1,dtype='complex128')
ps[int(f1*N):int(f2*N)+1]= ps_val
# plt.plot(np.abs(ps))
# plt.title("Original spectra")
# plt.show()


spec=np.zeros(N//2+1,dtype='complex128')
spec[int(f1*N):int(f2*N)+1]= np.sqrt(ps_val) *(np.random.randn(int(f2*N)+1-int(f1*N)) + 1j * np.random.randn(int(f2*N)+1-int(f1*N)))
y1=np.fft.irfft(spec)
# plot_spectra([y1,], 2000)

half_size = 32
bw = half_size
h = firwin(2 * half_size * up + 1, 1 / up, window=("kaiser", 10),scale=True)
# h = firwin(2 * half_size * up + 1, 1 / up, window=("hamming"),scale=True)

y1_up = resample_poly(y1,up=10,down=1,window=('hamming'))
print("len y1_up", len(y1_up))
# plot_spectra([y1_up,], 2000)

N = len(y1_up)



ps_val = np.ones(int(f2*N)+1-int(f1*N))
len_ps = len(ps_val)
if cut > 0:
    overlap_right = int(cut * N) #0.4 to 0.5
    overlap_left = int(cut / up * N) #0.04 to 0.05
    print("overlap samples", overlap_left, overlap_right)
    #right-edge cosine taper
    ps_val[len_ps - overlap_right:] *= np.cos(np.linspace(0, np.pi/2, overlap_right))**2
    #left-edge sine taper
    ps_val[:overlap_left] *= np.sin(np.linspace(0, np.pi/2, overlap_left))**2
spec=np.zeros(N//2+1,dtype='complex128')
spec[int(f1*N):int(f2*N)+1]= np.sqrt(ps_val) *(np.random.randn(int(f2*N)+1-int(f1*N)) + 1j * np.random.randn(int(f2*N)+1-int(f1*N)))
y2=np.fft.irfft(spec)
# plot_spectra([y2,], 2000)

plot_spectra([y1_up, 10*y2,], 2000, both=False)
# sys.exit()
ynew = y1_up + 10*y2
plot_spectra([ynew,], 2000, both=False)
# plt.title("Filter response function")
# plt.plot(2*up*np.arange(0,len(h)//2+1)/len(h),np.abs(np.fft.rfft(h))**2)
# plt.yscale("log",base=10)
# plt.xscale("log",base=2)
# plt.axvline(1,ls='dashed',c='grey')
# plt.axvline(2*f1/up,ls='dashed',c='blue')
# plt.axvline(12*f2/up,ls='dashed',c='blue')
# plt.show()