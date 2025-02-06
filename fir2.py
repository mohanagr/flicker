import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz
from scipy.special import sici
import time
import numba as nb
import sys


##
#  FFT method works for caclulating contribution of random numbers
#  Still need to get the contribution of past realization working
#
##

def get_acf(tau,f1,f2):
    s1,c1=sici(2*np.pi*f1*tau)
    s2,c2=sici(2*np.pi*f2*tau)
    y=2*(c2-c1)
    y=np.nan_to_num(y,nan=2*np.log(f2/f1)+1e-3)
    return y

def get_next_y(y_past,x,coeffs):
    n=len(y_past)
    print("x passed", x)
    print("y passed", y_past)
    y_big = np.hstack([y_past,np.zeros(len(x))])
    print("y big shape", y_big.shape)
    print("coeffs shape", coeffs.shape)
    for i in range(len(x)):
        y_big[i+n] = coeffs@y_big[i:n+i] + x[i]
    # plt.plot(y_big)
    # plt.show()
    return y_big[n:]

def get_next_y(y_past,x,coeffs):
    n=len(y_past)
    print("x passed", x)
    print("y passed", y_past)
    y_big = np.hstack([y_past,np.zeros(len(x))])
    print(y_big.shape)
    print(coeffs.shape)
    for i in range(len(x)):
        y_big[i+n] = coeffs@y_big[i:n+i] + x[i]
    # plt.plot(y_big)
    # plt.show()
    return y_big[n:]

# np.random.seed(42)
f1=0.05
f2=0.5
N=2*1000
ps=np.zeros(N//2+1,dtype='complex128')
ps[int(f1*N):int(f2*N)+1]=1/np.arange(int(f1*N),int(f2*N)+1) #N/2 is the scaling factor to line the two PS up.
# acf_dft=N*np.fft.irfft(ps)
# acf_anl=get_acf(np.arange(0,N//2+1),f1,f2)

nsamps=2000
acf_anl=get_acf(np.arange(0,nsamps),f1,f2)
C=toeplitz(acf_anl)
Cinv=np.linalg.inv(C)
vec=get_acf(np.arange(0,nsamps)+1,f1,f2)
vec=vec[::-1]
coeffs=vec.T@Cinv
sigma = np.sqrt(C[0,0]-vec@Cinv@vec.T)

# plt.plot(coeffs);plt.show()
noise = N*np.fft.irfft(np.sqrt(ps/2) * (np.random.randn(N//2+1) + 1j * np.random.randn(N//2+1)))  

rand_bank = sigma*np.random.randn(3*nsamps)
yy_krig = np.zeros(2*nsamps,dtype='float64')
yy_krig[:nsamps] = noise.copy()
yy = np.zeros(4*nsamps,dtype='float64')
for i in range(3*nsamps):
    yy[nsamps + i] = coeffs@yy[i:i+nsamps] + rand_bank[i]
for i in range(nsamps):
    yy_krig[nsamps + i] = coeffs@yy_krig[i:i+nsamps] + rand_bank[i]


delta = np.zeros(nsamps)
delta[0]=1
fir = get_next_y(np.zeros(len(coeffs)),delta,coeffs)
plt.plot(fir)
plt.title("impulse response")
plt.show()
hf = np.fft.rfft(np.hstack([fir,0*rand_bank]))

randf = np.fft.rfft(np.hstack([rand_bank,0*fir]))

randf_chunk = np.fft.rfft(np.hstack([rand_bank[-2*nsamps:],0*fir]))
hf_chunk = np.fft.rfft(np.hstack([fir,np.zeros(2*nsamps)]))

yy_next = np.fft.irfft(hf*randf)[:len(rand_bank)]
yy_chunk = np.fft.irfft(hf_chunk*randf_chunk)[:2*nsamps]

print(yy.shape,yy_chunk.shape)
plt.plot(yy[3*nsamps:])
plt.plot(yy_chunk[nsamps:])
plt.title("last  generated chunks")
plt.show()


plt.plot(yy_next)
plt.plot(yy[nsamps:])
plt.title("convolution verification")
plt.show()

plt.title("krig'd vs IIR")
plt.loglog(np.abs(np.fft.rfft(yy_next[-2000:])),label='IIR (after burn-in)')
plt.loglog(np.abs(np.fft.rfft(noise)),label='FFT realization')
plt.loglog(np.abs(np.fft.rfft(yy_krig[nsamps:])),label='krigd')
plt.ylim(1e-1,1e3)
plt.legend()
plt.show()