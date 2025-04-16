import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz
from scipy.special import sici
import time
import numba as nb
import sys
from rocket_fft import c2r
from scipy.signal import welch
import concurrent.futures


def get_acf(tau,f1,f2):
    s1,c1=sici(2*np.pi*f1*tau)
    s2,c2=sici(2*np.pi*f2*tau)
    y=2*(c2-c1)
    y=np.nan_to_num(y,nan=2*np.log(f2/f1)+1e-5)
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

@nb.njit(nogil=True)
def generate(n, bank,samp_bank,rand_bank,nlevels,coeffs,osamp_coeffs,krig_ptr,samp_ptr,hf,sigma,bw, krig_bank_size, samp_bank_size, krig_len, enable):
    NUM_STACK = np.zeros(nlevels+1,dtype='int64') # recursion stack should not exceed number of levels but still +1 for good luck
    LEV_STACK = np.zeros(nlevels+1,dtype='int64')
    NUM_STACK[0]=1
    LEV_STACK[0]=nlevels-1
    y=np.zeros(n+1,dtype='float64')
    NUM_PTR=0
    norm=1/(krig_bank_size + krig_len)
    upper=2*bw
    while(NUM_PTR>=0):

        curpt=NUM_STACK[NUM_PTR]
        curll=LEV_STACK[NUM_PTR]
        NUM_PTR-=1
        rownum = curpt%10 - 1 #row 0 is 0.1
        # print("Popped point", curpt, "level", curll, "rownum is", rownum+1)
        tot = bank[curll,krig_len+krig_ptr[curll]]
        if curll>0:
            if rownum == -1: #%10 is zero
                # bank ptr starts at krig len because first krig_len numbers are trash from circular convolution
                tot =  tot + samp_bank[curll-1,samp_ptr[curll-1]+bw-1]
            else:
                tot = tot + osamp_coeffs[rownum,:]@samp_bank[curll-1,samp_ptr[curll-1]:samp_ptr[curll-1]+upper]
                # print("tot is", tot)
            if curll==nlevels-1:
                y[curpt]+=tot
                if enable:
                    y[curpt] = y[curpt] + y[curpt-1] + 1e-4*np.random.randn(1)[0]
                NUM_PTR+=1;NUM_STACK[NUM_PTR]=curpt+1;LEV_STACK[NUM_PTR]=nlevels-1
                futpt=curpt+1
                if futpt==n+1: break
                for ii in range(nlevels-1):
                    rem=futpt%10
                    quo=futpt//10
                    if rem>0: break
                    else:
                        futpt=quo
                        NUM_PTR+=1
                        NUM_STACK[NUM_PTR]=quo #20 is 2 for level previous to it
                        LEV_STACK[NUM_PTR]=nlevels-2-ii
        
        samp_ptr[curll] +=1
        if samp_ptr[curll] > samp_bank_size - 2*bw: #for bottommost level samp_ptr should never move
            samp_bank[curll,:bw+bw-1] = samp_bank[curll, 1-bw-bw:]
            samp_ptr[curll] = 0
        samp_bank[curll, samp_ptr[curll]+bw+bw-1] = tot #next element after 2bw-1 is 2*bw but we moved sampt ptr ahead earlier

        #handle bank rotations
        krig_ptr[curll] +=1
        if krig_ptr[curll] == krig_bank_size:
            #used up all the krig'd values. generate next chunk
            noise = sigma*np.random.randn(krig_bank_size + krig_len) #generate bigger
            noise[:krig_len] = rand_bank[curll,:] #then replace first krig_len with stored last randn
            rand_bank[curll,:] = noise[-krig_len:] #store the last krig_len noise for next generation
            # np.fft.irfft(hf*np.fft.rfft(noise),out=bank[curll,:])
            c2r(hf*np.fft.rfft(noise),bank[curll,:],np.asarray([0,],dtype='int64'),False,norm,16)
            krig_ptr[curll] = 0
    # print("final y is", y)
    return y

@nb.njit(nogil=True)
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
acf_anl=get_acf(np.arange(0,coeff_len),f1,f2)
C=toeplitz(acf_anl)
Cinv=np.linalg.inv(C)
vec=get_acf(np.arange(0,coeff_len)+1,f1,f2)
vec=vec[::-1]
coeffs=vec.T@Cinv
sigma = np.sqrt(C[0,0]-vec@Cinv@vec.T)
print("krig stddev", sigma)

delta = np.zeros(krig_len)
delta[0]=1
fir = get_impulse(delta,coeffs) #size of krig coeffs can be different, don't matter.

krig_bank_size = 100*2000
hf = np.fft.rfft(np.hstack([fir,np.zeros(krig_bank_size+200)])) #transfer function
# print(hf.shape)

bw=32
taus=np.arange(-bw,bw)
t_n_diff = np.arange(1,10)/10
osamp_coeffs = np.zeros((len(t_n_diff), len(taus)),dtype='float64')
for i,dd in enumerate(t_n_diff):
    print((dd-taus)[bw])
    osamp_coeffs[i,:] = np.sinc(dd-taus)

print(len(taus))

rn = np.random.randn(len(fir) + krig_bank_size + 200) #extra hundred to make my oversampling easy
yy = np.fft.irfft(np.fft.rfft(rn)*hf)[krig_len:]
bigyy = np.zeros(krig_bank_size*10,dtype=yy.dtype)

curpt=99
for i in range(len(bigyy)):
    rownum = i%10
    if rownum == 0:
        curpt+=1
        bigyy[i] = yy[curpt]
    else:
        # print(np.arange(curpt-bw,curpt+bw)[bw])
        # print(rownum-1)
        bigyy[i] = osamp_coeffs[rownum-1]@yy[curpt-bw:curpt+bw]
        # sys.exit()

yy=yy[100:-100]
f,P=welch(yy,nperseg=2000,noverlap=1000)
spec=yy.reshape(-1,200)
spec=np.mean(np.abs(np.fft.rfft(spec,axis=1))**2,axis=0)
plt.loglog(np.abs(np.fft.rfft(yy))**2)
plt.show()
plt.loglog(spec)
plt.show()
plt.loglog(P)
plt.show()

f,P=welch(bigyy,nperseg=2000,noverlap=1000)
spec=bigyy.reshape(-1,2000)
spec=np.mean(np.abs(np.fft.rfft(spec,axis=1))**2,axis=0)
plt.loglog(np.abs(np.fft.rfft(bigyy))**2)
plt.show()
plt.loglog(spec)
plt.show()
plt.loglog(P)
plt.show()

#make some more and add to oversampled one
hf = np.fft.rfft(np.hstack([fir,np.zeros(krig_bank_size*10)]))
rn = np.random.randn(len(fir) + krig_bank_size*10)
yy2 = np.fft.irfft(np.fft.rfft(rn)*hf)[krig_len:]

yytot = bigyy + yy2

#calc spectra again
plt.loglog(np.abs(np.fft.rfft(yytot))**2)
plt.show()
spec=yytot.reshape(-1,2000)
spec=np.mean(np.abs(np.fft.rfft(spec,axis=1))**2,axis=0)
plt.loglog(spec)
plt.show()