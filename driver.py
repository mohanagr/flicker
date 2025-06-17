import numpy as np
import generate_narrowband as gn
import generate_flicker as gf
import pfb
from matplotlib import pyplot as plt
import time
import numba as nb

re=gn.narrowband(df=0.48,length=2048,up=1000) #conversion is 0.4 * N (0.4 is fraction of nyquist = N/2)
im=gn.narrowband(df=0.48,length=2048,up=1000)
lblock=4096
nlevels=13
up = 10
f2 = 1 / 2
f1 = 0.993 * f2 / up
# nsamp=2048*500
nsamp=2048*1000
obj1=pfb.StreamingPFB(tsize=nsamp)
obj2=pfb.StreamingPFB(tsize=nsamp)
start_delay=0
zero_delay = np.zeros(nsamp,dtype=np.float64)
t_orig = np.arange(nsamp,dtype=np.float64)
t_new = np.empty(nsamp,dtype=np.float64)
ybig = np.empty(nsamp,dtype='float64')
ybig2 = np.empty(nsamp,dtype='float64')
csum = np.empty(nsamp,dtype='float64')
clock = gf.flicker(nlevels, nsamp, f1, f2)
nspec_per_run = nsamp//4096
niter=100
start_chan=1830
nchans=10
spectra1 = np.empty((nspec_per_run*niter,nchans),dtype='complex128')
spectra2=spectra1.copy()
delays = np.empty(nspec_per_run*niter,dtype=np.float64)
# delays2 = np.empty(nsamp*niter,dtype=np.float64)
# delay = -5.5*np.ones(nsamp)
delay = np.zeros(nsamp)
tag='drift_delay2'
3
@nb.njit(parallel=True,cache=True)
def apply_drift_delay(delay, t_new, t_orig, slope, offset):
    nn=len(t_orig)
    for i in nb.prange(nn):
        delay[i] = slope*t_orig[i] + offset
        t_new[i] = t_orig[i] + delay[i]

offset=0
slope=5/20e3/4096 #sample drift per 20k spectra
try:
    for ii in range(niter):
        t1=time.time()
        re.generate().osamp()
        im.generate().osamp()
        # clock.generate()
        # delay = gf.cumsum(csum,clock.ybig,start_delay,10**(-clock.nlevels/2))
        # start_delay=delay[-1]
        # delays2[ii*nsamp:(ii+1)*nsamp]=delay
        # print("first val of clock drift is", clock.ybig[0])
        # gn.get_delay(t_new,t_orig,delay)
        offset=ii*nsamp*slope
        apply_drift_delay(delay,t_new,t_orig,slope,offset)
        # plt.plot(t_new-t_orig)
        # plt.plot()
        # plt.show()
        carrier = 0.4478 #this is now for the oversampled one, 1835/4096
        
        I=re.ybig
        Q=im.ybig
        gn.upconvert_delay_noise(ybig,I,Q,t_orig,carrier,0.05) #last argument is sigma of noise you wanna add
        I=gn.cubic_spline(t_new, t_orig, re.ybig)
        Q=gn.cubic_spline(t_new, t_orig, im.ybig)
        gn.upconvert_delay_noise(ybig2,I,Q,t_new,carrier,0.05)
        obj1.pfb(ybig)
        obj2.pfb(ybig2)
        #save data
        spectra1[ii*nspec_per_run:(ii+1)*nspec_per_run,:]=obj1.spectra[:,start_chan:start_chan+nchans]
        spectra2[ii*nspec_per_run:(ii+1)*nspec_per_run,:]=obj2.spectra[:,start_chan:start_chan+nchans]
        delays[ii*nspec_per_run:(ii+1)*nspec_per_run] = np.mean(np.reshape(delay,(-1,4096)),axis=1)
        # print(np.mean(np.reshape(delay,(-1,4096)),axis=1))
        t2=time.time()
        print(ii, t2-t1)
        # xc=np.abs(obj1.spectra)**2#*np.conj(obj2.spectra)
        # # print(xc.shape)
        # xc=np.mean(xc,axis=0)
        # plt.plot(np.abs(xc)[0:])
        # plt.show()
        # plt.plot(np.angle(xc)[0:])
        # plt.show()
    # plt.plot(delays2)
    # plt.show()
except Exception as e:
    pass
finally:
    np.savez(f"./spectra_{start_chan}_{start_chan+nchans}_{lblock}_{tag}.npz",spectra1=spectra1,spectra2=spectra2,delays=delays)




