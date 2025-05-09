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
import os
os.environ['NUMBA_OPT']='3'
os.environ['NUMBA_LOOP_VECTORIZE']='1'
os.environ['NUMBA_ENABLE_AVX']='1'

@nb.njit(cache=True)
def gen_krig(bank, rand_bank, hf, sigma, krig_bank_size, krig_len):
    # FIRST krig_len points in the OUTPUT will be crap. only use after that.
    norm=1/(krig_bank_size + krig_len)
    noise = sigma*np.random.randn(krig_bank_size + krig_len) #generate bigger
    noise[:krig_len] = rand_bank[:] #then replace first krig_len with stored last randn
    rand_bank[:] = noise[-krig_len:] #save the last krig_len noise for next generation
    # np.fft.irfft(hf*np.fft.rfft(noise),out=bank[:])
    c2r(hf*np.fft.rfft(noise),bank[:],np.asarray([0,],dtype='int64'),False,norm,16)

class narrowband():
    def __init__(self, nlevels, nsamp, f1, f2):

        coeff_len = 1024
        krig_len = 1024 #number of coeffs in the FIR filter
        acf_anl=self.get_acf(np.arange(0,coeff_len),f1, f2, nlevels)
        C=toeplitz(acf_anl)
        Cinv=np.linalg.inv(C)
        vec=self.get_acf(np.arange(0,coeff_len)+1, f1, f2, nlevels)
        vec=vec[::-1]
        coeffs=vec.T@Cinv
        sigma = np.sqrt(C[0,0]-vec@Cinv@vec.T)
        print("krig stddev", sigma)
        self.sigma = sigma
        L=10
        bw=32
        krig_bank_size = 63*1024
        self.bw=bw
        self.krig_len = krig_len
        self.L = L

        self.krig_bank_size = krig_bank_size
        self.bank=np.zeros((nlevels,krig_len+krig_bank_size),dtype='float64') #KRIG BANK
        self.krig_ptr = np.zeros(nlevels,dtype='int64')
        self.samp_ptr = np.zeros(nlevels,dtype='int64')
        self.ybig=np.empty(nsamp,dtype='float64') #generation starts from sample no. 1
        self.rand_bank = np.zeros((nlevels,krig_len),dtype='float64') #only gotta store the last krig_len rand for future kring generation
        self.rand_bank[:] = sigma*np.random.randn(nlevels*krig_len).reshape(nlevels,krig_len)

        delta = np.zeros(krig_len) #how far back you wanna convolve, length of FIR filter essentially
        delta[0]=1
        fir = self.get_impulse(delta,coeffs) #size of krig coeffs can be different, don't matter.

        self.hf = np.fft.rfft(np.hstack([fir,np.zeros(self.krig_bank_size)])) #krig transfer function

        # first len(fir) is garbage. after len(fir), krig timestream continues. 
        
        h=L*firwin(2*bw*L+1, 1/L,window=('kaiser',1))
    
    def get_acf(self,tau,df):
        y = np.sinc(df*tau)
        y[0]+=1e-6
        return y

    def get_impulse(self, x,coeffs):
        n=len(coeffs)
        y_big = np.zeros(len(x)+len(coeffs))
        for i in range(len(x)):
            y_big[i+n] = coeffs@y_big[i:n+i] + x[i]
        return y_big[n:]

    def generate(self):
        #generate next krig batch
        gen_krig(self.bank, self.rand_bank, self.hf, self.sigma, self.krig_bank_size, self.krig_len)
        self.bank[self.krig_len-self.bw:self.krig_len] = self.osamp_bank
        self.osamp_bank[:] = self.bank[-self.bw:] #copy last few to ensure continuity in oversampling
        return self
    
    def osamp(self):
        #oversample current krig batch
        big_interp_c(self.bank[self.krig_len-self.bw:],self.h,self.L,self.ybig,self.bw)
        return self