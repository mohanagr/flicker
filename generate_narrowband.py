import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz
from scipy.special import sici
import time
import numba as nb
import sys
from rocket_fft import c2r
import concurrent.futures
from scipy.signal import firwin
import sys
from upsample_poly import big_interp_c
from scipy.interpolate import CubicSpline, splrep, splev

np.random.seed(42)

@nb.njit(parallel=True)
def spline_eval(xnew, ynew, x, coeffs):
    dx = x[1]-x[0]
    x0 = x[0]
    m = len(x)
    n = len(ynew)
    n_coeffs = coeffs.shape[1]
    for i in nb.prange(n):
        idx = min(max(int((xnew[i]- x0)//dx),0),m-2)
        xx = xnew[i]-x[idx]
        vv=xx
        cc = coeffs[idx,:]
        tot = cc[0]
        # ynew[i] = coeffs[idx,0] #constant term
        for j in range(1,n_coeffs):
            tot += vv * cc[j]
            vv *= xx
        ynew[i]=tot

def cubic_spline(xnew, x, y, out=None):
    order=3
    cs_obj = splrep(x,y)
    if out is not None:
        ynew=out
    else:
        ynew = np.empty(len(xnew),dtype=y.dtype)
    dx = x[1]-x[0]
    nn = int((x[-1]-x[0])*3/dx + 1)
    x1 = np.linspace(x[0],x[-1],nn)
    y1 = splev(x1,cs_obj)
    sections = np.hstack([y1[:-1].reshape(-1,3),y[1:].reshape(-1,1)])
    A=np.zeros([order+1,order+1],dtype="float64")
    for i in range(order+1):
        A[:,i]=(x1[:(order+1)]-x1[0])**i
    coeffs = sections@np.linalg.inv(A.T)
    spline_eval(xnew, ynew, x, coeffs)
    return ynew

def get_acf(tau,df):
    y = np.sinc(df*tau)
    y[0]+=1e-6
    return y

def get_impulse(x,coeffs):
    n=len(coeffs)
    y_big = np.zeros(len(x)+len(coeffs))
    for i in range(len(x)):
        y_big[i+n] = coeffs@y_big[i:n+i] + x[i]
    return y_big[n:]


@nb.njit(cache=True)
def gen_krig(bank, rand_bank, hf, sigma, krig_bank_size, krig_len):
    # FIRST krig_len points in the OUTPUT will be crap. only use after that.
    norm=1/(krig_bank_size + krig_len)
    noise = sigma*np.random.randn(krig_bank_size + krig_len) #generate bigger
    noise[:krig_len] = rand_bank[:] #then replace first krig_len with stored last randn
    rand_bank[:] = noise[-krig_len:] #save the last krig_len noise for next generation
    # np.fft.irfft(hf*np.fft.rfft(noise),out=bank[:])
    c2r(hf*np.fft.rfft(noise),bank[:],np.asarray([0,],dtype='int64'),False,norm,16)

df=0.4
N=2*1000
ps=np.zeros(N//2+1,dtype='complex128')
ps[:int(df/2*N)+1]=1/(df*N + 1)
acf_dft=N*np.fft.irfft(ps)
acf_anl=get_acf(np.arange(0,N//2+1),df) #make sure both match bc we will generate initial realization from ps

plt.plot(acf_dft[:1000])
plt.plot(acf_anl[:1000])
plt.show()

coeff_len = 1024
krig_len = 1024 #number of coeffs in the FIR filter
acf_anl=get_acf(np.arange(0,coeff_len),df)
C=toeplitz(acf_anl)
Cinv=np.linalg.inv(C)

vec=get_acf(np.arange(0,coeff_len)+1,df)
vec=vec[::-1]
coeffs=vec.T@Cinv
sigma = np.sqrt(C[0,0]-vec@Cinv@vec.T)
print("krig stddev", sigma)

L=100 #upsample fastest if L is a multiple of 4 due to AVX2 being enabled.
bw=32

krig_bank_size = 20000 + bw
bank=np.zeros(krig_len+krig_bank_size,dtype='float64') #KRIG BANK
ybig=np.zeros(L*(krig_bank_size+bw-2*bw),dtype='float64') #oversampling of krig'ed timestream
rand_bank = np.zeros(krig_len,dtype='float64') #only gotta store the last krig_len rand for future kring generation
rand_bank[:] = sigma*np.random.randn(krig_len)
osamp_bank = np.zeros(bw,dtype='float64')

delta = np.zeros(krig_len) #how far back you wanna convolve, length of FIR filter essentially
delta[0]=1
fir = get_impulse(delta,coeffs) #size of krig coeffs can be different, don't matter.

hf = np.fft.rfft(np.hstack([fir,np.zeros(krig_bank_size)])) #transfer function

# first len(fir) is garbage. after len(fir), krig timestream continues. 
# eat into a bit of garbage to fill the past bw values for oversampling
# bw < krig_len
gen_krig(bank, rand_bank, hf, sigma, krig_bank_size, krig_len)
bank[krig_len-bw:krig_len] = osamp_bank
osamp_bank[:] = bank[-bw:] #copy last few to ensure continuity in oversampling



h = firwin(L*2*bw+1,1/L,window=('kaiser', 14))
print("krig bank size", krig_bank_size)
print("len bank", len(bank))
print("len y/L", len(ybig)/L)

big_interp_c(bank[krig_len-bw:],h,L,ybig,bw)

plt.loglog(np.abs(np.fft.rfft(bank[krig_len:-bw]))) #get 20000
plt.loglog(np.abs(np.fft.rfft(ybig))) #get 20000
plt.show()

#delay and put on carrier wave.