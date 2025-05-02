import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz
from scipy.special import sici
import time
import numba as nb
import sys
from rocket_fft import c2r
from scipy.signal import welch, firwin, resample_poly,upfirdn
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

@nb.njit(nogil=True,cache=True)
def generate(n, bank,samp_bank,rand_bank,nlevels,osamp_coeffs,krig_ptr,samp_ptr,hf,sigma,bw, krig_bank_size, samp_bank_size, krig_len):
    NUM_STACK = np.zeros(nlevels+1,dtype='int64') # recursion stack should not exceed number of levels but still +1 for good luck
    LEV_STACK = np.zeros(nlevels+1,dtype='int64')
    NUM_STACK[0]=1
    LEV_STACK[0]=nlevels-1
    y=np.zeros(n+1,dtype='float64')
    NUM_PTR=0
    norm=1/(krig_bank_size + krig_len)
    upper=2*bw
    up = osamp_coeffs.shape[0]
    while(NUM_PTR>=0):
        curpt=NUM_STACK[NUM_PTR]
        curll=LEV_STACK[NUM_PTR]
        NUM_PTR-=1
        rownum = curpt%up
        # print("Popped point", curpt, "level", curll, "rownum is", rownum+1)
        tot = bank[curll,krig_len+krig_ptr[curll]] #bank of krigs
        if curll>0:
            tot = tot + osamp_coeffs[rownum,:]@samp_bank[curll-1,samp_ptr[curll-1]:samp_ptr[curll-1]+upper]
            # print("tot is", tot)
            if curll==nlevels-1:
                y[curpt]+=tot
                NUM_PTR+=1;
                NUM_STACK[NUM_PTR]=curpt+1
                LEV_STACK[NUM_PTR]=nlevels-1
                futpt=curpt+1
                if futpt==n+1: break
                for ii in range(nlevels-1):
                    rem=futpt%up
                    quo=futpt//up
                    if rem>0: break
                    else:
                        futpt=quo
                        NUM_PTR+=1
                        NUM_STACK[NUM_PTR]=quo #20 is 2 for level previous to it
                        LEV_STACK[NUM_PTR]=nlevels-2-ii
        
        samp_ptr[curll] +=1
        if samp_ptr[curll] > samp_bank_size - 2*bw:
            samp_bank[curll,:bw+bw-1] = samp_bank[curll, 1-bw-bw:] 
            #say bw = 64. consider final 64 elements: idx 0-63. when we at 0, we still have 64 avl.
            # when we at 1, we have only 63 remaining.
            # copy these 63 back to beginning
            samp_ptr[curll] = 0
        #and point 64, idx 63 is now the tot we evalulated
        samp_bank[curll, samp_ptr[curll]+bw+bw-1] = tot

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

def plot_spectra(y,size):
    f,P=welch(y,nperseg=size,noverlap=size//2)
    spec=y.reshape(-1,size)
    spec=np.mean(np.abs(np.fft.rfft(spec,axis=1))**2,axis=0)
    f=plt.gcf()
    f.set_size_inches(10,4)
    plt.subplot(121)
    plt.title("Stacked FFT PS")
    plt.loglog(spec)
    plt.subplot(122)
    plt.title("Welch PS w/ windowing & overlap")
    plt.loglog(P)
    plt.tight_layout()
    plt.show()


up=10
f2=0.5
f1=0.5/up
# N=2*1000
# ps=np.zeros(N//2+1,dtype='complex128')
# ps[int(f1*N):int(f2*N)+1]=1/np.arange(int(f1*N),int(f2*N)+1) #N/2 is the scaling factor to line the two PS up.
# acf_dft=N*np.fft.irfft(ps)
# acf_anl=get_acf(np.arange(0,N//2+1),f1,f2)

coeff_len = 2048
krig_len = 2048
acf_anl=get_acf(np.arange(0,coeff_len),f1,f2) #+ 500*np.cos(np.arange(0,coeff_len)*2*np.pi*f2)
C=toeplitz(acf_anl)
Cinv=np.linalg.inv(C)
vec=get_acf(np.arange(0,coeff_len)+1,f1,f2)
vec=vec[::-1]
coeffs=vec.T@Cinv
sigma = np.sqrt(C[0,0]-vec@Cinv@vec.T)
print("krig stddev", sigma)
plt.plot(coeffs)
plt.show()
delta = np.zeros(krig_len)
delta[0]=1
fir = get_impulse(delta,coeffs) #size of krig coeffs can be different, don't matter.
plt.plot(fir)
plt.show()
krig_bank_size = 63*2048
hf = np.fft.rfft(np.hstack([fir,np.zeros(krig_bank_size)])) #transfer function
# hf = np.fft.rfft(np.hstack([fir,np.zeros(krig_bank_size+200)])) #transfer function + 200 for manual osamp later
#design a filter to replace osamp_coeffs
import upsample_poly as upsamp
half_size = 64
bw=half_size
h=up*firwin(2*half_size*up+1, 0.1,window=('kaiser',10))
print("len h", len(h))
plt.title("Filter response function")
plt.loglog(2*np.arange(0,len(h)//2+1)/len(h),np.abs(np.fft.rfft(h))**2)
plt.show()

osamp_coeffs = h[:-1].reshape(-1,up).T[:,::-1].copy() #refer to notes. (last column is h[0], h[1], h[2]. h[3])

nlevels=4

rand_bank = np.zeros((nlevels,krig_len),dtype='float64') #only gotta store the last krig_len rand 
rand_bank[:, :] = sigma*np.random.randn(nlevels*krig_len).reshape(nlevels,krig_len) #white noise bank

krig_ptr = np.zeros(nlevels,dtype='int64')
samp_ptr = np.zeros(nlevels,dtype='int64')
bank=np.zeros((nlevels,krig_len+krig_bank_size),dtype='float64') #krig bank
for jj in range(nlevels):
    gen_krig(bank, rand_bank, jj, hf, sigma, krig_bank_size, krig_len)
# plot_spectra(bank[0,krig_len:],200)
# plot_spectra(bank[1,krig_len:],200)
# sys.exit()
samp_bank = np.zeros((nlevels, krig_bank_size),dtype=bank.dtype) #krig + white bank
samp_bank[0,:] = bank[0,krig_len:].copy() #topmost level just krig
samp_bank_size = samp_bank.shape[1]
ctr=[0]*nlevels
#let's try to forward generate two levels, long timestream and look at spectra.
for ll in range(1,nlevels):
    # print("processing level", ll, "parent", ll-1)
    for i in range(samp_bank.shape[1]):
        #generate level's own krig - already there!
    #         print("samp bank begin", samp_bank[ll,i])
        krig_samp_own = bank[ll,krig_len+krig_ptr[ll]]
        krig_ptr[ll] +=1
        if krig_ptr[ll] == krig_bank_size:
            #used up all the krig'd values. generate next chunk
            print("ran out of krig. resetting", ll)
            gen_krig(bank, rand_bank, ll, hf, sigma, krig_bank_size, krig_len)
            krig_ptr[ll] = 0
        samp_bank[ll,i] += krig_samp_own
        rownum = i%up
        samp_bank[ll,i] += (osamp_coeffs[rownum,:]@samp_bank[ll-1,ctr[ll-1]:ctr[ll-1]+2*bw])

# print("ctrs",ctr)
# # plot_spectra(samp_bank[1,:],2048)
# # plot_spectra(samp_bank[2,:],2048)
yy=generate(20000000, bank,samp_bank,rand_bank,nlevels,osamp_coeffs,krig_ptr,samp_ptr,hf,sigma,bw, krig_bank_size, samp_bank_size, krig_len)
# plot_spectra(yy[1:],20000)
# sys.exit()

plot_spectra(yy[1:],20000)

# sys.exit()

krig_bank_size = 2000*100
hf = np.fft.rfft(np.hstack([fir,np.zeros(krig_bank_size+2*half_size)]))
rn = sigma*np.random.randn(len(fir) + krig_bank_size+2*half_size) #extra hundred to make my oversampling easy
yy = np.fft.irfft(np.fft.rfft(rn)*hf)[krig_len:]
bigyy = np.zeros(krig_bank_size*up,dtype=yy.dtype)
bigyy_new = np.zeros(krig_bank_size*up,dtype=yy.dtype)
print("len yy", len(yy))

upsamp.big_interp(yy,h,up,bigyy,half_size)

print("len bigyy", len(bigyy))

plot_spectra(bigyy,2000)
print("done")
# print("osamp coeffs shape", osamp_coeffs.shape)

# print("resamp shape",bigyy_resamp.shape)

# for i in range(len(bigyy)):
#     rownum = i%10
#               #osamp_coeffs2 uses h, 10 rows.
#     if rownum == 0:
#         curpt+=1
#         # bigyy[i] = yy[curpt]
#         bigyy[i] = osamp_coeffs[rownum]@yy[curpt-bw:curpt+bw]
#         bigyy_new[i] = osamp_coeffs2[rownum,:-1]@yy[curpt-bw:curpt+bw]
#     else:
#         # print(np.arange(curpt-bw,curpt+bw)[bw])
#         # print(rownum-1)
#         # print(i)
#         # bigyy[i] = osamp_coeffs[rownum-1]@yy[curpt-bw:curpt+bw]
#         bigyy[i] = osamp_coeffs[rownum]@yy[curpt-bw:curpt+bw]
#         bigyy_new[i] = osamp_coeffs2[rownum,:-1]@yy[curpt-bw:curpt+bw]

#         # sys.exit()
# # beta = 0.1102*(60 - 8.7)

# yy=yy[100:-100]
# bigyy2= resample_poly(yy,up=up,down=1,window=('kaiser',beta))
# bigyy2= resample_poly(yy,up=10,down=1,window=('kaiser',2))

# plot_spectra(yy,2000)
# plot_spectra(bigyy,2000)
# plot_spectra(bigyy_new,2000)
# plot_spectra(bigyy2,2000)
# sys.exit()

#make some more and add to oversampled one
hf = np.fft.rfft(np.hstack([fir,np.zeros(krig_bank_size*up)]))
rn = sigma*np.random.randn(len(fir) + krig_bank_size*up)
yy2 = np.fft.irfft(np.fft.rfft(rn)*hf)[krig_len:]

yytot = bigyy + yy2
#calc spectra again
plot_spectra(yytot,2000)
# yytot = bigyy2 + yy2
# plot_spectra(yytot,2000)