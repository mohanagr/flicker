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

# np.random.seed(42)
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
    def __init__(self, df):
        coeff_len = 1024
        krig_len = 1024 #number of coeffs in the FIR filter
        acf_anl=self.get_acf(np.arange(0,coeff_len),df)
        C=toeplitz(acf_anl)
        Cinv=np.linalg.inv(C)
        vec=self.get_acf(np.arange(0,coeff_len)+1,df)
        vec=vec[::-1]
        coeffs=vec.T@Cinv
        sigma = np.sqrt(C[0,0]-vec@Cinv@vec.T)
        print("krig stddev", sigma)
        self.sigma = sigma
        L=100 #upsample fastest if L is a multiple of 4 due to AVX2 being enabled.
        bw=32
        krig_bank_size = 20000 + bw
        self.bw=bw
        self.krig_len = krig_len
        self.L = L
        self.krig_bank_size = krig_bank_size
        self.bank=np.zeros(krig_len+krig_bank_size,dtype='float64') #KRIG BANK
        self.ybig=np.zeros(L*(krig_bank_size+bw-2*bw),dtype='float64') #oversampling of krig'ed timestream
        self.rand_bank = np.zeros(krig_len,dtype='float64') #only gotta store the last krig_len rand for future kring generation
        self.rand_bank[:] = sigma*np.random.randn(krig_len)
        self.osamp_bank = np.zeros(bw,dtype='float64') #bank for storing last few samples for oversampling

        delta = np.zeros(krig_len) #how far back you wanna convolve, length of FIR filter essentially
        delta[0]=1
        fir = self.get_impulse(delta,coeffs) #size of krig coeffs can be different, don't matter.

        self.hf = np.fft.rfft(np.hstack([fir,np.zeros(self.krig_bank_size)])) #krig transfer function

        # first len(fir) is garbage. after len(fir), krig timestream continues. 
        # eat into a bit of garbage to fill the past bw values for oversampling
        # bw < krig_len
        
        self.h = firwin(L*2*bw+1,1/L,window=('kaiser', 10)) #oversampling filter
    
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

@nb.njit(parallel=True, cache=True)
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

#verify osampling for both I and Q
# fractional bandwidth 0.4 is two sided. For a zero-centered signal, 0.4*N/2 on +ve and rest on -ve freqs.
# but when you upconvert total bw is 0.4*N
re=narrowband(df=0.4) #conversion is 0.4 * N (0.4 is fraction of nyquist = N/2)
im=narrowband(df=0.4)
re.generate().osamp()
im.generate().osamp()

plt.loglog(np.abs(np.fft.rfft(re.bank[re.krig_len:-re.bw]))) #get 20000
plt.loglog(np.abs(np.fft.rfft(re.ybig))) #get 20000
plt.show()

plt.loglog(np.abs(np.fft.rfft(im.bank[im.krig_len:-im.bw]))) #get 20000
plt.loglog(np.abs(np.fft.rfft(im.ybig))) #get 20000
plt.show()

from scipy.interpolate import CubicSpline, make_interp_spline
t_orig=np.arange(len(re.ybig))
t_new=np.arange(len(re.ybig)) - 1.5

carrier = 0.3 #this is now for the oversampled one
ybig = re.ybig * np.cos(2*np.pi*carrier*t_orig) - im.ybig * np.sin(2*np.pi*carrier*t_orig)
# ybig2 = cubic_spline(t_new, t_orig, ybig)
ybig2 = cubic_spline(t_new, t_orig, re.ybig) * np.cos(2*np.pi*carrier*t_new) - cubic_spline(t_new, t_orig, im.ybig) * np.sin(2*np.pi*carrier*t_new)

# cs = make_interp_spline(t_orig,ybig,k=3)
# ybig2 = cs(t_new)
#new fractional bandwidth is df/L after upsampling so df/L * 20000 = 80
print(ybig.shape, ybig2.shape)
f1=np.fft.rfft(ybig.reshape(-1,20000),axis=1)
f2=np.fft.rfft(ybig2.reshape(-1,20000),axis=1)
# f1 = np.fft.rfft(ybig)
# plt.loglog(np.abs(f1))
# plt.show()
# f2 = np.fft.rfft(ybig2)

# plt.loglog(np.abs(f2))
# plt.show()
xc=f1*np.conj(f2)
print(xc.shape)
xc=np.mean(xc,axis=0)
plt.plot(np.abs(xc)[0:])
plt.show()
plt.plot(np.angle(xc)[0:])
plt.show()
# plt.plot(np.angle(xc))
# plt.xlim(0,4000) #signal will be present at 0.4/L * 10000