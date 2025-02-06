import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz
from scipy.special import sici
import time
import numba as nb
import sys


global bw
global krig_len
global krig_bank_size
global samp_bank_size
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

def gen_krig(bank, rand_bank, level, hf, sigma):
    #len(fir) = krig_bank_size
    bank[level,:] = np.fft.irfft(hf*rand_bank)[krig_bank_size:]
    rand_bank[:krig_bank_size] = rand_bank[krig_bank_size:2*krig_bank_size]
    rand_bank[krig_bank_size:2*krig_bank_size] = sigma*np.random.randn(krig_bank_size)

@nb.njit()
def recurse(i,bank,samp_bank,level,coeffs,osamp_coeffs,krig_ptr,samp_ptr, hf):
    # print("rec", i, "level", level)
    global bw
    global krig_len
    global krig_bank_size
    global samp_bank_size
    retval=0

    krig_samp_own = bank[level,krig_ptr[level]]
    krig_ptr[level] +=1
    if krig_ptr[level] > krig_bank_size - krig_len:
        bank[level,:krig_len-1] = bank[level, -krig_len-1:]
        # bank[level, 499] = krig_samp_own
        krig_ptr[level] = 0
    bank[level, krig_ptr[level]+krig_len-1]=krig_samp_own
#     samp_bank[1,i] += krig_samp_own
    retval += krig_samp_own
    if level == 0:
        return retval
    if i%10==0:
        # print("recursing")
        # samp_val = recurse(i//10, bank, samp_bank, level-1, coeffs, osamp_coeffs)
        samp_ptr[level-1] +=1
        if samp_ptr[level-1] > samp_bank_size - 2*bw:
            samp_bank[level-1,:2*bw-1] = samp_bank[level-1, -2*bw+1:]
            samp_ptr[level-1] = 0
        samp_bank[level-1, samp_ptr[level-1]+2*bw-1]=recurse(i//10, bank, samp_bank, level-1, coeffs, osamp_coeffs, krig_ptr, samp_ptr)
        # samp_bank[level-1,:] = np.roll(samp_bank[level-1,:],shift=-1)
        # if samp_ptr[level-1] == 1500:
        #     samp_bank[level-1,:499] = samp_bank[level-1, -499:]
        #     samp_bank[level-1, 499] = krig_samp_own
        #     krig_ptr[level-1] = 0
        # krig_ptr[level-1] +=1
        # samp_bank[level-1, krig_ptr[level-1]+499]=krig_samp_own
        # samp_bank[level-1,-1] = recurse(i//10, bank, samp_bank, level-1, coeffs, osamp_coeffs, krig_ptr)
    # rownum = i%10 - 1 #row 0 is 0.1
    retval += (osamp_coeffs[i%10 - 1,:]@samp_bank[level-1,samp_ptr[level-1]:samp_ptr[level-1]+2*bw])
    return retval

@nb.njit()
def generate(n, bank,samp_bank_small,start_level,coeffs,osamp_coeffs,krig_ptr,samp_ptr):
    global bw
    y = np.empty(n+1)
    for i in range(1,n+1):
        y[i] = recurse(i,bank,samp_bank_small,start_level,coeffs,osamp_coeffs,krig_ptr,samp_ptr)
    return y


f1=0.05
f2=0.5
N=2*1000
ps=np.zeros(N//2+1,dtype='complex128')
ps[int(f1*N):int(f2*N)+1]=1/np.arange(int(f1*N),int(f2*N)+1) #N/2 is the scaling factor to line the two PS up.
# acf_dft=N*np.fft.irfft(ps)
# acf_anl=get_acf(np.arange(0,N//2+1),f1,f2)

nlevels=4
nsamps=500
bank=np.zeros((nlevels,krig_bank_size),dtype='float64')
rand_bank = np.zeros((nlevels,3*krig_bank_size),dtype='float64')
rand_bank[:,:2*krig_bank_size] = sigma*np.random.randn(nlevels*2*krig_bank_size).reshape(nlevels,2*krig_bank_size)

acf_anl=get_acf(np.arange(0,nsamps),f1,f2)
C=toeplitz(acf_anl)
Cinv=np.linalg.inv(C)
vec=get_acf(np.arange(0,nsamps)+1,f1,f2)
vec=vec[::-1]
coeffs=vec.T@Cinv
sigma = np.sqrt(C[0,0]-vec@Cinv@vec.T)
print(sigma)

bw=16
krig_len = nsamps
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

ctr=[0,0,0]
# plt.loglog(np.abs(np.fft.rfft(samp_bank[1,:])));plt.title("before")

for ll in range(1,nlevels):
    print("processing level", ll, "parent", ll-1)
    for i in range(samp_bank.shape[1]):
        #generate level's own krig - already there!
    #         print("samp bank begin", samp_bank[ll,i])
        krig_samp_own = coeffs@bank[ll,:] + sigma * np.random.randn(1)[0]
        bank[ll,:] = np.roll(bank[ll,:],shift=-1)
        bank[ll,-1] = krig_samp_own
        # krig_samp_own = coeffs@bank[ll,krig_ptr[ll]:krig_ptr[ll]+500] + sigma * np.random.randn(1)[0]
        # if krig_ptr[ll] == 1500:
        #     bank[ll,:499] = bank[ll, -499:]
        #     bank[ll, 499] = krig_samp_own
        #     krig_ptr[ll] = 0
        # krig_ptr[ll] +=1
        samp_bank[ll,i] += krig_samp_own
        if i%10==0:
            ctr[ll-1]+=1
            samp_bank[ll,i] += samp_bank[ll-1,ctr[ll-1] + bw]
            continue
        rownum = i%10 - 1 #row 0 is 0.1
        samp_bank[ll,i] += (osamp_coeffs[rownum,:]@samp_bank[ll-1,ctr[ll-1]:ctr[ll-1]+2*bw])

bank=np.zeros((nlevels,2000),dtype='float64')
for ll in range(nlevels):
    noise = N*np.fft.irfft(np.sqrt(ps/2) * (np.random.randn(N//2+1) + 1j * np.random.randn(N//2+1)))  
    bank[ll,:]=noise
samp_bank_small = np.zeros((samp_bank.shape[0], 2000), dtype='float64')
samp_bank_small[:,:2*bw] = samp_bank[:,200:200+2*bw].copy()
print(samp_bank_small.shape)
yy=np.zeros(2000001,dtype=np.float64)

# tot=0
# for i in range(1,2000001):
#     t1=time.time()
#     yy[i] = recurse(i,bank,samp_bank_small,nlevels-1,coeffs,osamp_coeffs,krig_ptr)
#     t2=time.time()
#     tot+=(t2-t1)
t1=time.time()
yy = generate(2000000,bank,samp_bank_small,nlevels-1,coeffs,osamp_coeffs,krig_ptr,samp_ptr)
t2=time.time()
tot=t2-t1
print(tot/2000001)
plt.loglog(np.abs(np.fft.rfft(yy[1:])));plt.title("power spectrum")
plt.show()
