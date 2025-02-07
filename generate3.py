import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz
from scipy.special import sici
import time
import numba as nb
import sys

##
# rand bank can be very small - size of filter * 2
# krig bank can be very big - only need last len(filter) rand for the next million krig
# total time 15e-8 per samp
# 1.5e-8 per samp taken up by 64 element dot product, 
# 5e-8 per samp taken up by FFT + rng + data movement
# only rng is 2e-8 per samp
# bad memory access?
##

# x = np.random.randn(10*500).reshape(10,500)
# cc = np.random.randn(500)
# niter=100
# tot=0
# for j in range(niter):
#     t1=time.time()
#     roll_test(x,cc)
#     t2=time.time()
#     tot+=(t2-t1)
# print("roll of 100 takes", tot/niter)
# sys.exit(0)
def get_acf(tau,f1,f2):
    s1,c1=sici(2*np.pi*f1*tau)
    s2,c2=sici(2*np.pi*f2*tau)
    y=2*(c2-c1)
    y=np.nan_to_num(y,nan=2*np.log(f2/f1)+1e-3)
    return y

def get_impulse(x,coeffs):
    n=len(coeffs)
    y_big = np.zeros(len(x)+len(coeffs))
    for i in range(len(x)):
        y_big[i+n] = coeffs@y_big[i:n+i] + x[i]
    return y_big[n:]

@nb.njit()
def gen_krig(bank, rand_bank, level, hf, sigma, krig_bank_size):
    #len(fir) = krig_bank_size
    #rand bank is 3x krig size. 2x for next generation, 1x full of zeros
    # print(hf.shape, rand_bank[level].shape)
    bank[level,:] = np.fft.irfft(hf*np.fft.rfft(rand_bank[level]))[krig_bank_size:2*krig_bank_size]
    rand_bank[level, :krig_bank_size] = rand_bank[level, krig_bank_size:2*krig_bank_size]
    rand_bank[level, krig_bank_size:2*krig_bank_size] = sigma*np.random.randn(krig_bank_size)

@nb.njit()
def recurse(i,bank,samp_bank,rand_bank,level,coeffs,osamp_coeffs,krig_ptr,samp_ptr,hf,sigma, bw, krig_bank_size, samp_bank_size):
    # print("rec", i, "level", level)
    # global bw
    # global krig_len
    # global krig_bank_size
    # global samp_bank_size
    upper=2*bw
    rownum = i%10 - 1 #row 0 is 0.1
    retval = bank[level,krig_ptr[level]]
    krig_ptr[level] +=1
    if krig_ptr[level] == krig_bank_size:
        #used up all the krig'd values. generate next chunk
        # gen_krig(bank, rand_bank, level, hf, sigma)
        bank[level,:] = np.fft.irfft(hf*np.fft.rfft(rand_bank[level]))[krig_bank_size:2*krig_bank_size]
        rand_bank[level, :krig_bank_size] = rand_bank[level, krig_bank_size:2*krig_bank_size]
        rand_bank[level, krig_bank_size:2*krig_bank_size] = sigma*np.random.randn(krig_bank_size)
        krig_ptr[level] = 0
    if level == 0:
        return retval
    if rownum==-1: #this whole block seems to be taking 5e-8
        # samp_val = recurse(i//10, bank, samp_bank, level-1, coeffs, osamp_coeffs)
        samp_ptr[level-1] +=1
        if samp_ptr[level-1] > samp_bank_size - 2*bw:
            samp_bank[level-1,:bw+bw-1] = samp_bank[level-1, 1-bw-bw:]
            samp_ptr[level-1] = 0
        # samp_bank[level-1, samp_ptr[level-1]+bw+bw-1]=recurse(i//10, bank, samp_bank, rand_bank, level-1, coeffs, osamp_coeffs, krig_ptr, samp_ptr, hf, sigma, bw, krig_bank_size, samp_bank_size)

    retval += (osamp_coeffs[rownum,:]@samp_bank[level-1,samp_ptr[level-1]:samp_ptr[level-1]+upper])
    return retval

@nb.njit()
def generate(n, bank,samp_bank_small,rand_bank,start_level,coeffs,osamp_coeffs,krig_ptr,samp_ptr,hf,sigma,bw, krig_bank_size, samp_bank_size):
    y = np.empty(n+1)
    for i in range(1,n+1):
        y[i] = recurse(i,bank,samp_bank_small,rand_bank,start_level,coeffs,osamp_coeffs,krig_ptr,samp_ptr,hf,sigma,bw, krig_bank_size, samp_bank_size)
    return y

@nb.njit()
def generate_dot(n,x,y,z):
    for i in range(1,n+1):
        z[:] = x@y

@nb.njit()
def generate_rand(n, sigma):
    y = sigma*np.random.randn(n)
    return y

f1=0.05
f2=0.5
N=2*1000
ps=np.zeros(N//2+1,dtype='complex128')
ps[int(f1*N):int(f2*N)+1]=1/np.arange(int(f1*N),int(f2*N)+1) #N/2 is the scaling factor to line the two PS up.
# acf_dft=N*np.fft.irfft(ps)
# acf_anl=get_acf(np.arange(0,N//2+1),f1,f2)

krig_len = 2000
krig_bank_size = 8192
acf_anl=get_acf(np.arange(0,krig_len),f1,f2)
C=toeplitz(acf_anl)
Cinv=np.linalg.inv(C)
vec=get_acf(np.arange(0,krig_len)+1,f1,f2)
vec=vec[::-1]
coeffs=vec.T@Cinv
sigma = np.sqrt(C[0,0]-vec@Cinv@vec.T)
print(sigma)

nlevels=10

bank=np.zeros((nlevels,krig_bank_size),dtype='float64')
rand_bank = np.zeros((nlevels,3*krig_bank_size),dtype='float64')
rand_bank[:,:2*krig_bank_size] = sigma*np.random.randn(nlevels*2*krig_bank_size).reshape(nlevels,2*krig_bank_size)

delta = np.zeros(krig_bank_size) # as long as we want our usable bank of krig to be
delta[0]=1
fir = get_impulse(delta,coeffs) #size of krig coeffs can be different, don't matter.
# plt.plot(fir)
# plt.title("impulse response")
# plt.show()
hf = np.fft.rfft(np.hstack([fir,np.zeros(2*krig_bank_size)]))

#burn in the krig_bank
for level in range(nlevels):
    gen_krig(bank, rand_bank, level, hf, sigma, krig_bank_size)

# plt.loglog(np.abs(np.fft.rfft(bank[0])));
# plt.loglog(np.abs(np.fft.rfft(bank[1])));
# plt.loglog(np.abs(np.fft.rfft(bank[2])));
# plt.title("rand level 0");plt.show()
# sys.exit(0)

bw=32

taus=np.arange(-bw,bw)
print(len(taus))
# coeff=np.ones(len(taus))


t_n_diff = np.arange(1,10)/10
osamp_coeffs = np.zeros((len(t_n_diff), len(taus)),dtype='float64')
for i,dd in enumerate(t_n_diff):
    # print("saving coeffs for", dd)
    osamp_coeffs[i,:] = np.sinc(dd-taus)
print(osamp_coeffs.shape)
krig_ptr = np.zeros(nlevels,dtype='int64')
samp_ptr = np.zeros(nlevels,dtype='int64')
#forward generation first to enable later sampling
samp_bank = np.zeros(bank.shape,dtype=bank.dtype)
# samp_bank = bank.copy()
samp_bank[0,:] = bank[0,:].copy()

ctr=[0]*nlevels
# plt.loglog(np.abs(np.fft.rfft(samp_bank[1,:])));plt.title("before")

for ll in range(1,nlevels):
    print("processing level", ll, "parent", ll-1)
    for i in range(samp_bank.shape[1]):
        #generate level's own krig - already there!
    #         print("samp bank begin", samp_bank[ll,i])
        krig_samp_own = bank[level,krig_ptr[ll]]
        krig_ptr[ll] +=1
        if krig_ptr[ll] == krig_bank_size:
            #used up all the krig'd values. generate next chunk
            # print("resetting", ll)
            gen_krig(bank, rand_bank, level, hf, sigma, krig_bank_size)
            krig_ptr[ll] = 0
        samp_bank[ll,i] += krig_samp_own
        if i%10==0:
            ctr[ll-1]+=1
            samp_bank[ll,i] += samp_bank[ll-1,ctr[ll-1] + bw]
            continue
        rownum = i%10 - 1 #row 0 is 0.1
        samp_bank[ll,i] += (osamp_coeffs[rownum,:]@samp_bank[ll-1,ctr[ll-1]:ctr[ll-1]+2*bw])
print("pre-algo samp_bank fill counters", ctr)
print("krig counters")
print(krig_ptr)

samp_bank_size=256*krig_bank_size
samp_bank_small = np.zeros((samp_bank.shape[0], samp_bank_size), dtype='float64')
samp_bank_small[:,:2*bw] = samp_bank[:,200:200+2*bw].copy()

tot=0
# yy=np.empty(2000001,dtype='float64')
# for i in range(1,2000001):
#     t1=time.time()
#     yy[i] = recurse(i,bank,samp_bank_small,rand_bank,nlevels-1,coeffs,osamp_coeffs,krig_ptr,samp_ptr,hf,sigma)
#     t2=time.time()
#     tot+=(t2-t1)
generate(200,bank,samp_bank_small,rand_bank,nlevels-1,coeffs,osamp_coeffs,krig_ptr,samp_ptr,hf,sigma,bw, krig_bank_size, samp_bank_size)
t1=time.time()
yy = generate(2000000,bank,samp_bank_small,rand_bank,nlevels-1,coeffs,osamp_coeffs,krig_ptr,samp_ptr,hf,sigma, bw, krig_bank_size, samp_bank_size)
t2=time.time()
tot=t2-t1
print(tot/2000001)


xx=np.random.randn(64)
yy=np.random.randn(64)
zz=np.empty(64)

# yy = generate_rand(2000000,sigma)
# zz = generate_dot(200,xx,yy,zz)
yy = generate_rand(20,sigma)
t1=time.time()
yy = generate_rand(2000000,sigma)
t2=time.time()
tot=t2-t1
print(tot/2000000)

plt.loglog(np.abs(np.fft.rfft(yy[1:])));plt.title("power spectrum")
plt.show()
