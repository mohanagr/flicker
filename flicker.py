import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz
from scipy.special import sici
import time
import numba as nb
import sys
from rocket_fft import c2r
import concurrent.futures
from scipy.signal import welch

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
    #I think the following data movement is not necessary for a single thread.
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
            #I think the following data movement is not necessary for a single thread.
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

if len(sys.argv[1:]) < 2:
    print("usage flicker.py nlevels npoints")
    sys.exit(1)

nlevels = int(sys.argv[1])
npoints = int(sys.argv[2])
coeff_len = 2048
krig_len = 2048
krig_bank_size = 2048*63
acf_anl=get_acf(np.arange(0,coeff_len),f1,f2)
C=toeplitz(acf_anl)
Cinv=np.linalg.inv(C)
vec=get_acf(np.arange(0,coeff_len)+1,f1,f2)
vec=vec[::-1]
coeffs=vec.T@Cinv
sigma = np.sqrt(C[0,0]-vec@Cinv@vec.T)
print("krig stddev", sigma)

bank=np.zeros((nlevels,krig_len+krig_bank_size),dtype='float64')
rand_bank = np.zeros((nlevels,krig_len),dtype='float64') #only gotta store the last krig_len rand 
rand_bank[:, :] = sigma*np.random.randn(nlevels*krig_len).reshape(nlevels,krig_len)

delta = np.zeros(krig_len)
delta[0]=1
fir = get_impulse(delta,coeffs) #size of krig coeffs can be different, don't matter.
# plt.plot(fir)
# plt.show()
# sys.exit(0)
# print("firt shape", fir.shape)
# plt.title("impulse response")
# plt.show()
hf = np.fft.rfft(np.hstack([fir,np.zeros(krig_bank_size)])) #transfer function
# print(hf.shape)

#burn in the krig_bank
for level in range(nlevels):
    gen_krig(bank, rand_bank, level, hf, sigma, krig_bank_size, krig_len)

# plt.loglog(np.abs(np.fft.rfft(bank[0])));
# plt.loglog(np.abs(np.fft.rfft(bank[1])));
# plt.loglog(np.abs(np.fft.rfft(bank[2])));
# plt.title("rand level 0");plt.show()
# sys.exit(0)

bw=64 #bandwidth of sinc used as oversampling weights

taus=np.arange(-bw,bw)
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
print("setting up random number banks...")
for ll in range(1,nlevels):
    # print("processing level", ll, "parent", ll-1)
    for i in range(samp_bank.shape[1]):
        #generate level's own krig - already there!
    #         print("samp bank begin", samp_bank[ll,i])
        krig_samp_own = bank[ll,krig_len+krig_ptr[ll]]
        krig_ptr[ll] +=1
        if krig_ptr[ll] == krig_bank_size:
            #used up all the krig'd values. generate next chunk
            # print("resetting", ll)
            gen_krig(bank, rand_bank, ll, hf, sigma, krig_bank_size, krig_len)
            krig_ptr[ll] = 0
        samp_bank[ll,i] += krig_samp_own
        if i%10==0:
            # print("level", ll, "i", i)
            ctr[ll-1]+=1
            samp_bank[ll,i] += samp_bank[ll-1,ctr[ll-1] + bw]
            continue
        rownum = i%10 - 1 #row 0 is 0.1
        samp_bank[ll,i] += (osamp_coeffs[rownum,:]@samp_bank[ll-1,ctr[ll-1]:ctr[ll-1]+2*bw])
# print("krig counters", krig_ptr)
# print("samp counters", ctr)
# print("krig + len", np.log2(krig_bank_size+krig_len))
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
generate(200,bank,samp_bank_small,rand_bank,nlevels,coeffs,osamp_coeffs,krig_ptr,samp_ptr,hf,sigma,bw, krig_bank_size, samp_bank_size, krig_len,False)
generate_rand(20,sigma)
# plt.loglog(np.abs(np.fft.rfft(yy[1:])));plt.title("power spectrum")
# plt.show()
# sys.exit(0)
t1=time.time()
args=[npoints,bank,samp_bank_small,rand_bank,nlevels,coeffs,osamp_coeffs,krig_ptr,samp_ptr,hf,sigma, bw, krig_bank_size, samp_bank_size, krig_len, False]
# with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
#     fu1=executor.submit(generate, *args)
#     while True:
#         yy=fu1.result()
#         fu1=executor.submit(generate, *args)
#         fu2=executor.submit(generate_rand, 4*npoints, sigma)
#         fu3=executor.submit(generate_rand, 4*npoints, sigma)
#         fu4=executor.submit(generate_rand, 4*npoints, sigma)
#         yy2=fu2.result()
#         yy2=fu3.result()
#         yy2=fu4.result()
        
yy = generate(*args)
f1, Pxx1 = welch(yy, fs=2.0, nperseg=2*10**nlevels)
# yy2 = generate(npoints,bank,samp_bank_small,rand_bank,nlevels,coeffs,osamp_coeffs,krig_ptr,samp_ptr,hf,sigma, bw, krig_bank_size, samp_bank_size, krig_len, True)
t2=time.time()
tot1=t2-t1
print("exectime flicker", tot1/npoints)
# print("exectime flicker", tot1/npoints)
plt.loglog(f1[1:], Pxx1[1:])
plt.legend()
plt.title(r"$1/f^\alpha$ power spectrum")
plt.show()
# plt.loglog(np.abs(np.fft.rfft(yy[1:])), label='$alpha$ = -1')
# plt.legend()
# plt.title(r"$1/f^\alpha$ power spectrum")
# plt.show()
# xx=np.random.randn(64)
# yy=np.random.randn(64)
# zz=np.empty(64)

# yy = generate_rand(2000000,sigma)
# zz = generate_dot(200,xx,yy,zz)
yy = generate_rand(20,sigma)
t1=time.time()
yy = generate_rand(npoints,sigma)
t2=time.time()
tot2=t2-t1
print("exectime white", tot2/npoints)

print("exectime ratio flicker/white:", tot1/tot2)

