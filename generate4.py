import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz
from scipy.special import sici
import time
import numba as nb
import sys
from rocket_fft import c2r
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
def gen_krig(bank, rand_bank, level, hf, sigma, krig_bank_size, krig_len):
    #len(fir) = krig_len
    norm=1/(krig_bank_size + krig_len)
    noise = sigma*np.random.randn(krig_bank_size + krig_len) #generate bigger
    # print("noise shape", noise.shape)
    noise[:krig_len] = rand_bank[level,:] #then replace first krig_len with stored last randn
    rand_bank[level,:] = noise[-krig_len:] #store the last krig_len noise for next generation
    # bank[level,:] = np.fft.irfft(hf*np.fft.rfft(noise))[krig_len:]
    # print("out array shape", bank.shape, "input shape", hf.shape)
    # np.fft.irfft(hf*np.fft.rfft(noise),out=bank[level,:]) # rocket fft doesn't support out kwarg
    c2r(hf*np.fft.rfft(noise),bank[level,:],np.asarray([0,],dtype='int64'),False,norm,16)
@nb.njit(parallel=True)
def copy_arr(x,y):
    nn=len(x)
    for i in nb.prange(nn):
        y[i] = x[i]

@nb.njit()
def recurse(i,bank,samp_bank,rand_bank,level,coeffs,osamp_coeffs,krig_ptr,samp_ptr,hf,sigma, bw, krig_bank_size, samp_bank_size, krig_len):
    # print("rec", i, "level", level)
    # global bw
    # global krig_len
    # global krig_bank_size
    # global samp_bank_size
    upper=2*bw
    norm=1/(krig_bank_size + krig_len)
    rownum = i%10 - 1 #row 0 is 0.1
    retval = bank[level,krig_len+krig_ptr[level]]
    krig_ptr[level] +=1
    if krig_ptr[level] == krig_bank_size:
        #used up all the krig'd values. generate next chunk
        noise = sigma*np.random.randn(krig_bank_size + krig_len) #generate bigger
        noise[:krig_len] = rand_bank[level,:] #then replace first krig_len with stored last randn
        rand_bank[level,:] = noise[-krig_len:] #store the last krig_len noise for next generation
        # temp = np.fft.irfft(hf*np.fft.rfft(noise))
        # copy_arr(temp,bank[level,:])
        c2r(hf*np.fft.rfft(noise),bank[level,:],np.asarray([0,],dtype='int64'),False,norm,16)
        krig_ptr[level] = 0
    if level == 0:
        return retval
    if rownum==-1: #this whole block seems to be taking 5e-8
        # samp_val = recurse(i//10, bank, samp_bank, level-1, coeffs, osamp_coeffs)
        samp_ptr[level-1] +=1
        if samp_ptr[level-1] > samp_bank_size - 2*bw:
            samp_bank[level-1,:bw+bw-1] = samp_bank[level-1, 1-bw-bw:]
            samp_ptr[level-1] = 0
        samp_bank[level-1, samp_ptr[level-1]+bw+bw-1]=recurse(i//10, bank, samp_bank, rand_bank, level-1, coeffs, osamp_coeffs, krig_ptr, samp_ptr, hf, sigma, bw, krig_bank_size, samp_bank_size, krig_len)
        # samp_bank[level-1, samp_ptr[level-1]+bw+bw-1]=2#recurse(i//10, bank, samp_bank, rand_bank, level-1, coeffs, osamp_coeffs, krig_ptr, samp_ptr, hf, sigma, bw, krig_bank_size, samp_bank_size)
        retval += samp_bank[level-1,samp_ptr[level-1]+bw-1] #center element of next chunk
        return retval
    # for jj in range(upper):
    #     retval += (osamp_coeffs[rownum,jj]*samp_bank[level-1,samp_ptr[level-1]+jj])
    retval += (osamp_coeffs[rownum,:]@samp_bank[level-1,samp_ptr[level-1]:samp_ptr[level-1]+upper])
    return retval

@nb.njit()
def generate(n, bank,samp_bank_small,rand_bank,start_level,coeffs,osamp_coeffs,krig_ptr,samp_ptr,hf,sigma,bw, krig_bank_size, samp_bank_size, krig_len):
    y = np.empty(n+1)
    for i in range(1,n+1):
        y[i] = recurse(i,bank,samp_bank_small,rand_bank,start_level,coeffs,osamp_coeffs,krig_ptr,samp_ptr,hf,sigma,bw, krig_bank_size, samp_bank_size, krig_len)
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

coeff_len = 2048
krig_len = 1024
krig_bank_size = 1024*63
acf_anl=get_acf(np.arange(0,coeff_len),f1,f2)
C=toeplitz(acf_anl)
Cinv=np.linalg.inv(C)
vec=get_acf(np.arange(0,coeff_len)+1,f1,f2)
vec=vec[::-1]
coeffs=vec.T@Cinv
sigma = np.sqrt(C[0,0]-vec@Cinv@vec.T)
print(sigma)

nlevels=3

bank=np.zeros((nlevels,krig_len+krig_bank_size),dtype='float64')
rand_bank = np.zeros((nlevels,krig_len),dtype='float64') #only gotta store the last krig_len rand 
rand_bank[:, :] = sigma*np.random.randn(nlevels*krig_len).reshape(nlevels,krig_len)

delta = np.zeros(krig_len) # as long as we want our usable bank of krig to be
delta[0]=1
fir = get_impulse(delta,coeffs) #size of krig coeffs can be different, don't matter.
# plt.plot(fir)
# plt.show()
# sys.exit(0)
# print("firt shape", fir.shape)
# plt.title("impulse response")
# plt.show()
hf = np.fft.rfft(np.hstack([fir,np.zeros(krig_bank_size)]))
# print(hf.shape)
#burn in the krig_bank
for level in range(nlevels):
    gen_krig(bank, rand_bank, level, hf, sigma, krig_bank_size, krig_len)

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
        krig_samp_own = bank[level,krig_len+krig_ptr[ll]]
        krig_ptr[ll] +=1
        if krig_ptr[ll] == krig_bank_size:
            #used up all the krig'd values. generate next chunk
            # print("resetting", ll)
            gen_krig(bank, rand_bank, level, hf, sigma, krig_bank_size, krig_len)
            krig_ptr[ll] = 0
        samp_bank[ll,i] += krig_samp_own
        if i%10==0:
            # print("level", ll, "i", i)
            ctr[ll-1]+=1
            samp_bank[ll,i] += samp_bank[ll-1,ctr[ll-1] + bw]
            continue
        rownum = i%10 - 1 #row 0 is 0.1
        samp_bank[ll,i] += (osamp_coeffs[rownum,:]@samp_bank[ll-1,ctr[ll-1]:ctr[ll-1]+2*bw])
print("krig counters", krig_ptr)
print("samp counters", ctr)
print("krig + len", np.log2(krig_bank_size+krig_len))
samp_bank_size=krig_bank_size
samp_bank_small = np.zeros((samp_bank.shape[0], samp_bank_size), dtype='float64')
samp_bank_small[:,:2*bw] = samp_bank[:,ctr[0]:ctr[0]+2*bw].copy()

tot=0
# yy=np.empty(2000001,dtype='float64')
# for i in range(1,2000001):
#     t1=time.time()
#     yy[i] = recurse(i,bank,samp_bank_small,rand_bank,nlevels-1,coeffs,osamp_coeffs,krig_ptr,samp_ptr,hf,sigma)
#     t2=time.time()
#     tot+=(t2-t1)
from scipy.signal import welch
generate(200,bank,samp_bank_small,rand_bank,nlevels-1,coeffs,osamp_coeffs,krig_ptr,samp_ptr,hf,sigma,bw, krig_bank_size, samp_bank_size, krig_len)
t1=time.time()
yy = generate(2000000,bank,samp_bank_small,rand_bank,nlevels-1,coeffs,osamp_coeffs,krig_ptr,samp_ptr,hf,sigma, bw, krig_bank_size, samp_bank_size, krig_len)
t2=time.time()
tot1=t2-t1
print(tot/2000001)
f1, Pxx1 = welch(yy, fs=2.0, nperseg=2*10**nlevels)
plt.loglog(f1[1:], Pxx1[1:])
plt.legend()
plt.title(r"$1/f^\alpha$ power spectrum")
plt.show()

# xx=np.random.randn(64)
# yy=np.random.randn(64)
# zz=np.empty(64)

# yy = generate_rand(2000000,sigma)
# # zz = generate_dot(200,xx,yy,zz)

yy = generate_rand(20,sigma)
t1=time.time()
yy = generate_rand(2000000,sigma)
t2=time.time()
tot2=t2-t1
print(tot/2000000)


print(tot1/tot2)
# plt.loglog(np.abs(np.fft.rfft(yy[1:])));plt.title("power spectrum")
# plt.show()
