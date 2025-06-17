import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz
from scipy.special import sici
import time
import os
os.environ['NUMBA_OPT']='3'
os.environ['NUMBA_GDB_BINARY']='/usr/bin/gdb'
os.environ['NUMBA_LOOP_VECTORIZE']='1'
os.environ['NUMBA_ENABLE_AVX']='1'
# os.environ['NUMBA_DEBUG']='1'
# os.environ['NUMBA_FULL_TRACEBACKS']='1'

import numba as nb
import sys
from rocket_fft import c2r
from scipy.signal import welch, firwin, resample_poly, upfirdn
import concurrent.futures



def get_acf(tau, f1, f2):
    s1, c1 = sici(2 * np.pi * f1 * tau)
    s2, c2 = sici(2 * np.pi * f2 * tau)
    y = 2 * (c2 - c1)
    y = np.nan_to_num(y, nan=2 * np.log(f2 / f1) + 1e-4)  # was 1e-5
    return y


def get_impulse(x, coeffs):
    n = len(coeffs)
    y_big = np.zeros(len(x) + len(coeffs))
    for i in range(len(x)):
        y_big[i + n] = coeffs @ y_big[i : n + i] + x[i]
    return y_big[n:]


@nb.njit()
def gen_krig(bank, rand_bank, level, hf, sigma, krig_bank_size, krig_len):
    # len(fir) = krig_len
    norm = 1 / (krig_bank_size + krig_len)
    noise = sigma * np.random.randn(krig_bank_size + krig_len)  # generate bigger
    # print("noise shape", noise.shape)
    noise[:krig_len] = rand_bank[
        level, :
    ]  # then replace first krig_len with stored last randn
    rand_bank[level, :] = noise[
        -krig_len:
    ]  # store the last krig_len noise for next generation
    # bank[level,:] = np.fft.irfft(hf*np.fft.rfft(noise))[krig_len:]
    # print("out array shape", bank.shape, "input shape", hf.shape)
    # np.fft.irfft(hf*np.fft.rfft(noise),out=bank[level,:]) # rocket fft doesn't support out kwarg
    c2r(
        hf * np.fft.rfft(noise),
        bank[level, :],
        np.asarray(
            [
                0,
            ],
            dtype="int64",
        ),
        False,
        norm,
        16,
    )


@nb.njit(nogil=True,cache=True)
# @nb.njit(debug=True)
def generate(
    n,
    krig_bank,
    samp_bank,
    rand_bank,
    nlevels,
    osamp_coeffs,
    krig_ptr,
    samp_ptr,
    hf,
    sigma,
    bw,
    krig_bank_size,
    samp_bank_size,
    krig_len,
):
    NUM_STACK = np.zeros(
        nlevels + 1, dtype="int64"
    )  # recursion stack should not exceed number of levels but still +1 for good luck
    LEV_STACK = np.zeros(nlevels + 1, dtype="int64")
    NUM_STACK[0] = 0
    LEV_STACK[0] = nlevels - 1
    circular_buffer_offsets = np.zeros(nlevels,dtype='int64')
    y = np.zeros(
        n, dtype="float64"
    )  # we have krig_bank_size values already from forward generation
    NUM_PTR = 0
    norm = 1 / (krig_bank_size + krig_len)
    upper = 2 * bw
    up = osamp_coeffs.shape[0]
    # nb.gdb_init()
    while NUM_PTR >= 0:
        # peek
        lev = LEV_STACK[NUM_PTR]
        pt = NUM_STACK[NUM_PTR]
        quo = pt // up
        rem = pt % up
        cbop = circular_buffer_offsets[lev - 1]
        # print(f"lev {lev} pt {pt} quo {quo} rem {rem} cbo of prev lev {cbop}")
        # if this level doesn't have enough krig vals left, regenerate

        # check visited status
        if lev == 0:
            if krig_ptr[lev] == krig_bank_size:
                # print(f"regenerating krig for lev {lev}")
                noise = sigma*np.random.randn(krig_bank_size + krig_len) #generate bigger
                noise[:krig_len] = rand_bank[lev,:] #then replace first krig_len with stored last randn
                rand_bank[lev,:] = noise[-krig_len:] #store the last krig_len noise for next generation
                # np.fft.irfft(hf*np.fft.rfft(noise),out=bank[lev,:])
                c2r(hf*np.fft.rfft(noise),krig_bank[lev,:],np.asarray([0,],dtype='int64'),False,norm,16)
                krig_ptr[lev] = 0
            tot = krig_bank[lev, krig_len+krig_ptr[lev]]
            krig_ptr[lev]+=1
            if samp_ptr[lev] > krig_bank_size-1:
                circular_buffer_offsets[lev] = pt - upper + 1
                # print(f"cbo for lev {lev} set to {circular_buffer_offsets[lev]}")
                samp_ptr[lev] = upper - 1 #this just indicates the tail of moving window; point is INCLUDED
                samp_bank[lev, : upper - 1] = samp_bank[lev, -upper + 1 :]
                # print(f"last elements included were {samp_bank[lev, -upper+1]}, {samp_bank[lev, -1]}")
                # print(f"First and 2bw-1'th element {samp_bank[lev, 0]}, {samp_bank[lev, upper-2]}")
            samp_bank[lev, samp_ptr[lev]] = tot
            NUM_PTR-=1 #pop
        elif quo + upper - 1 > samp_ptr[lev - 1] + cbop: #keep checking ahead
            # print(f"checking quo+upper {quo+upper -1} > {samp_ptr[lev-1]+cbop}")
            # parent hasn't been visted yet
            # push
            NUM_PTR += 1
            LEV_STACK[NUM_PTR] = lev - 1
            NUM_STACK[NUM_PTR] = quo + upper - 1
            samp_ptr[lev-1] += 1
            # print(f"pushed lev {lev-1} and {quo + upper - 1}, and samp ptr prev levl at {samp_ptr[lev -1]}")
        else:
            # make sure we have enough room left in circular buffer
            # print(f"krig ptr of lev {lev} at {krig_ptr[lev]}")
            if krig_ptr[lev] == krig_bank_size:
                # print(f"regenerating krig for lev {lev}")
                noise = sigma*np.random.randn(krig_bank_size + krig_len) #generate bigger
                noise[:krig_len] = rand_bank[lev,:] #then replace first krig_len with stored last randn
                rand_bank[lev,:] = noise[-krig_len:] #store the last krig_len noise for next generation
            #     # np.fft.irfft(hf*np.fft.rfft(noise),out=bank[lev,:])
                c2r(hf*np.fft.rfft(noise),krig_bank[lev,:],np.asarray([0,],dtype='int64'),False,norm,16)
                krig_ptr[lev] = 0
            tot = krig_bank[lev, krig_len+krig_ptr[lev]]+osamp_coeffs[rem, :]@samp_bank[lev - 1, quo - cbop : quo - cbop + upper]
            if lev == nlevels-1:
                # print("adding pt + 1 to stack")
                # print(f"dot product for lev {lev}")
                # print(f"lev {lev-1} starting {quo}-{cbop} = {quo - cbop}")
                y[pt] = tot
                if pt == n-1: break
                # NUM_PTR+=1 #pop and push combined
                LEV_STACK[NUM_PTR]=nlevels-1
                NUM_STACK[NUM_PTR]=pt+1
            else:
                if samp_ptr[lev] > krig_bank_size-1:
                    circular_buffer_offsets[lev] = pt - upper + 1
                    # print(f"cbo for lev {lev} set to {circular_buffer_offsets[lev]}")
                    samp_ptr[lev] = upper - 1 #this just indicates the tail of moving window; point is INCLUDED
                    # print("the first point is now,", np.arange(0,200)[-upper+1])
                    samp_bank[lev, : upper - 1] = samp_bank[lev, -upper + 1 :]
                    # print(f"last elements included were {samp_bank[lev, -upper+1]}, {samp_bank[lev, -1]}")
                    # print(f"First and 2bw-1'th element {samp_bank[lev, 0]}, {samp_bank[lev, upper-2]}")
                # print(f"setting {samp_ptr[lev]}")
                # print(f"dot product for lev {lev}")
                # print(f"lev {lev-1} starting {quo}-{cbop} = {quo - cbop}")
                samp_bank[lev, samp_ptr[lev]] = tot
                NUM_PTR-=1 #pop
            krig_ptr[lev]+=1
        # print("-----------------------------------------------------------")
    return y

@nb.njit(nogil=True,cache=True)
# @nb.njit(debug=True)
def generate2(
    n,
    krig_bank,
    samp_bank,
    rand_bank,
    nlevels,
    osamp_coeffs,
    krig_ptr,
    samp_ptr,
    hf,
    sigma,
    bw,
    krig_bank_size,
    samp_bank_size,
    krig_len,
):
    NUM_STACK = np.zeros(
        nlevels + 1, dtype="int64"
    )  # recursion stack should not exceed number of levels but still +1 for good luck
    LEV_STACK = np.zeros(nlevels + 1, dtype="int64")
    NUM_STACK[0] = 0
    LEV_STACK[0] = nlevels - 1
    circular_buffer_offsets = np.zeros(nlevels,dtype='int64')
    y = np.zeros(
        n, dtype="float64"
    )  # we have krig_bank_size values already from forward generation
    NUM_PTR = 0
    norm = 1 / (krig_bank_size + krig_len)
    upper = 2 * bw
    up = osamp_coeffs.shape[0]
    # nb.gdb_init()
    while NUM_PTR >= 0:
        # peek
        lev = LEV_STACK[NUM_PTR]
        pt = NUM_STACK[NUM_PTR]
        quo = pt // up
        rem = pt % up
        cbop = circular_buffer_offsets[lev - 1]
        # print(f"lev {lev} pt {pt} quo {quo} rem {rem} cbo of prev lev {cbop}")
        # if this level doesn't have enough krig vals left, regenerate
        if krig_ptr[lev] == krig_bank_size:
            # print(f"regenerating krig for lev {lev}")
            noise = sigma*np.random.randn(krig_bank_size + krig_len) #generate bigger
            noise[:krig_len] = rand_bank[lev,:] #then replace first krig_len with stored last randn
            rand_bank[lev,:] = noise[-krig_len:] #store the last krig_len noise for next generation
        #     # np.fft.irfft(hf*np.fft.rfft(noise),out=bank[lev,:])
            c2r(hf*np.fft.rfft(noise),krig_bank[lev,:],np.asarray([0,],dtype='int64'),False,norm,16)
            krig_ptr[lev] = 0
        # check visited status
        if lev > 0 and ((quo + upper - 1) > (samp_ptr[lev - 1] + cbop)): #keep checking ahead
            # print(f"checking quo+upper {quo+upper -1} > {samp_ptr[lev-1]+cbop}")
            # parent hasn't been visted yet
            # push
            NUM_PTR += 1
            LEV_STACK[NUM_PTR] = lev - 1
            NUM_STACK[NUM_PTR] = quo + upper - 1
            samp_ptr[lev-1] += 1
            continue
        
        tot = krig_bank[lev, krig_len+krig_ptr[lev]]
        if lev > 0:
            tot += osamp_coeffs[rem, :]@samp_bank[lev - 1, quo - cbop : quo - cbop + upper]
        krig_ptr[lev]+=1

        if lev == nlevels-1:
            # print("adding pt + 1 to stack")
            # print(f"dot product for lev {lev}")
            # print(f"lev {lev-1} starting {quo}-{cbop} = {quo - cbop}")
            y[pt] = tot
            if pt == n-1: 
                print("final stack ptr at", NUM_PTR)
                break
            # NUM_PTR+=1 #pop and push combined
            LEV_STACK[NUM_PTR]=nlevels-1
            NUM_STACK[NUM_PTR]=pt+1
        else:
            if samp_ptr[lev] > krig_bank_size-1:
                circular_buffer_offsets[lev] = pt - upper + 1
                # print(f"cbo for lev {lev} set to {circular_buffer_offsets[lev]}")
                samp_ptr[lev] = upper - 1 #this just indicates the tail of moving window; point is INCLUDED
                # print("the first point is now,", np.arange(0,200)[-upper+1])
                samp_bank[lev, : upper - 1] = samp_bank[lev, -upper + 1 :]
            samp_bank[lev, samp_ptr[lev]] = tot
            NUM_PTR-=1 #pop
        # print("-----------------------------------------------------------")
    return y

@nb.njit(nogil=True)
def generate_rand(n, sigma):
    y = sigma * np.random.randn(n)
    return y


def plot_spectra(y, size,both=True):
    spec = y.reshape(-1, size)
    spec = np.mean(np.abs(np.fft.rfft(spec, axis=1)) ** 2, axis=0)
    if both:
        f, P = welch(y, nperseg=size, noverlap=size // 2)
        f = plt.gcf()
        f.set_size_inches(10, 4)
        plt.subplot(121)
        plt.title("Stacked FFT PS")
        plt.loglog(spec)
        plt.subplot(122)
        plt.title("Welch PS w/ windowing & overlap")
        plt.loglog(P)
        plt.tight_layout()
        plt.show()
    else:
        plt.title("Averaged Power Spectrum")
        plt.loglog(np.arange(0,size//2+1)/(size//2),spec)
        plt.xlabel("Frequency (fraction of nyquist)")
        plt.show()



@nb.njit(parallel=True, cache=True)
def square_add(x, y):
    nn = len(x)
    for i in nb.prange(nn):
        y[i] += np.abs(x[i]) ** 2


############################
tt = np.random.randn(10)
tt2 = np.random.randn(10)
square_add(tt, tt2)
del tt
del tt2
############################

up = 10
f2 = 1 / 2
f1 = 0.993 * f2 / up
# f1=f2/up
N = 2 * 1000



acf_anl=get_acf(np.arange(0,N//2+1),f1,f2)
Ndft=2*10
ps=np.zeros(Ndft//2+1,dtype='complex128')
ps[int(f1*Ndft):int(f2*Ndft)+1]=Ndft/np.arange(int(f1*Ndft),int(f2*Ndft)+1)
f=plt.gcf()
f.set_size_inches(12,4)
plt.subplot(121)
plt.title("Discrete PS")
plt.loglog(np.arange(0,Ndft//2+1)/(Ndft//2), ps,marker='o',ls='',label='16-point FFT PS')
Ndft=2*1000
ps=np.zeros(Ndft//2+1,dtype='complex128')
ps[int(f1*Ndft):int(f2*Ndft)+1]=Ndft/np.arange(int(f1*Ndft),int(f2*Ndft)+1)
plt.title("Analytical PS")
plt.loglog(np.arange(0,Ndft//2+1)/(Ndft//2), ps,c='orange',label='Analytical PS')
plt.xlabel("Frequency (fraction of Nyquist)")
plt.legend()

plt.subplot(122)
plt.title("Analytical vs Discrete correlation functions")
Ndfts = np.asarray([100, 500])*2
plt.plot(acf_anl,label='Analytical corrfunc',lw=2)
markerstyle = ['o', 'd']
markercolors = ['orange','yellow']
for i,Ndft in enumerate(Ndfts):
    ps=np.zeros(Ndft//2+1,dtype='complex128')
    ps[int(f1*Ndft):int(f2*Ndft)+1]=1/np.arange(int(f1*Ndft),int(f2*Ndft)+1) #N/2 is the scaling factor to line the two PS up.
    acf_dft=Ndft*np.fft.irfft(ps)
    plt.plot(np.arange(Ndft)[::3], acf_dft[::3], label=f"FFT corrfunc N={Ndft}",ls='',marker=markerstyle[i],markersize=5, markeredgecolor='black',color=markercolors[i],markeredgewidth=1)
plt.xlim(0,100)
plt.xlabel("Time (sample number)")
plt.legend()
plt.tight_layout()
plt.show()



coeff_len = 2048
krig_len = 2048
acf_anl = get_acf(
    np.arange(0, coeff_len), f1, f2
)  # + 500*np.cos(np.arange(0,coeff_len)*2*np.pi*f2)
C = toeplitz(acf_anl)
Cinv = np.linalg.inv(C)
vec = get_acf(np.arange(0, coeff_len) + 1, f1, f2)
vec = vec[::-1]
coeffs = vec.T @ Cinv
sigma = np.sqrt(C[0, 0] - vec @ Cinv @ vec.T)
print("krig stddev", sigma)


plt.show()
delta = np.zeros(krig_len)
delta[0] = 1
fir = get_impulse(delta, coeffs)  # size of krig coeffs can be different, don't matter.

plt.title("IIR impulse response")
plt.plot(fir)
plt.xlabel("Time (sample number)")
plt.show()

krig_bank_size = 200000
hf = np.fft.rfft(np.hstack([fir, np.zeros(krig_bank_size)]))  # transfer function
# hf = np.fft.rfft(np.hstack([fir,np.zeros(krig_bank_size+200)])) #transfer function + 200 for manual osamp later
# design a filter to replace osamp_coeffs
import upsample_poly as upsamp

half_size = 32
bw = half_size
h = up* firwin(2 * half_size * up + 1, 1 / up, window=("kaiser", 1),scale=True)

f=plt.gcf()
f.set_size_inches(10,4)
plt.subplot(121)
plt.title("Filter response")
print(len(h))
print(len(np.fft.rfft(h)))
print(np.arange(len(h)//2+1)/(len(h)//2+1))
plt.loglog(np.arange(len(h)//2+1)/(len(h)//2+1), np.abs(np.fft.rfft(firwin(2 * half_size * up + 1, 1 / up, window=("kaiser", 1),scale=True)))**2, label=r"Kaiser $\beta=1$")
plt.loglog(np.arange(len(h)//2+1)/(len(h)//2+1), np.abs(np.fft.rfft(firwin(2 * half_size * up + 1, 1 / up, window=("kaiser", 16),scale=True)))**2, label=r"Kaiser $\beta=10$")
plt.legend()
plt.xlabel("Frequency (fraction of nyquist)")
plt.subplot(122)
plt.plot(np.arange(-len(h)//2,len(h)//2), h)
plt.title(r"Filter coefficients Kaiser $\beta=1$")
plt.xlabel("Sample number (m = 10*n)")
plt.tight_layout()
plt.show()
# sys.exit()
# print("len h", len(h))
# plt.title("Filter response function")
# plt.plot(2*up*np.arange(0,len(h)//2+1)/len(h),np.abs(np.fft.rfft(h))**2)
# plt.yscale("log",base=10)
# plt.xscale("log",base=2)
# plt.axvline(1,ls='dashed',c='grey')
# plt.axvline(2*f1/up,ls='dashed',c='blue')
# plt.axvline(2*f2/up,ls='dashed',c='blue')
# plt.show()

# sys.exit()

osamp_coeffs = (
    h[:-1].reshape(-1, up).T[:, ::-1].copy()
)  # refer to notes. (last column is h[0], h[1], h[2]. h[3])

nlevels = 2

rand_bank = np.zeros(
    (nlevels, krig_len), dtype="float64"
)  # only gotta store the last krig_len rand
rand_bank[:, :] = sigma * np.random.randn(nlevels * krig_len).reshape(
    nlevels, krig_len
)  # white noise bank

krig_ptr = np.zeros(nlevels, dtype="int64")
samp_ptr = np.zeros(nlevels, dtype="int64")
bank = np.zeros((nlevels, krig_len + krig_bank_size), dtype="float64")  # krig bank
for jj in range(nlevels):
    gen_krig(bank, rand_bank, jj, hf, sigma, krig_bank_size, krig_len)
# plot_spectra(bank[0,krig_len:],200)
# plot_spectra(bank[1,krig_len:],200)
# sys.exit()
samp_bank = np.zeros((nlevels, krig_bank_size), dtype=bank.dtype)  # krig + white bank
samp_bank[0, :] = bank[0, krig_len:].copy()  # topmost level just krig
samp_bank_size = samp_bank.shape[1]
ctr = [0] * nlevels
# ctr=[0]*nlevels
# Forward generate all but the topmost and the bottommost level
for ll in range(1, nlevels):
    print("processing level", ll, "parent", ll - 1)
    for i in range(samp_bank.shape[1]):
        # generate level's own krig - already there!
        #         print("samp bank begin", samp_bank[ll,i])

        krig_samp_own = bank[ll, krig_len + krig_ptr[ll]]
        if krig_ptr[ll] == krig_bank_size:
            # used up all the krig'd values. generate next chunk
            # print("ran out of krig. resetting", ll)
            gen_krig(bank, rand_bank, ll, hf, sigma, krig_bank_size, krig_len)
            krig_ptr[ll] = 0
        samp_bank[ll, i] += krig_samp_own
        rownum = i % up
        if i > 0 and rownum == 0:
            ctr[ll - 1] += 1
        # print("level", ll, "samp", i, "krig ptr", krig_ptr[ll], "counters", ctr)
        samp_bank[ll, i] += (
            osamp_coeffs[rownum, :]
            @ samp_bank[ll - 1, ctr[ll - 1] : ctr[ll - 1] + 2 * bw]
        )
        krig_ptr[ll] += 1
samp_ptr[:]=krig_bank_size-1 #rolling pointer that tracks the end of the 2*bw window
samp_ptr[-1]=123456 #test value to make sure final level's ptr is not touched.
krig_ptr[-1]=0
krig_ptr[0]=krig_bank_size
print(krig_ptr)
y=np.zeros(krig_bank_size,dtype='float64')
# y=resample_poly(samp_bank[0,:20000],up=10,down=1,window=('kaiser',1))

# y=upfirdn(h,samp_bank[0,:20000+2*bw],up=10) #this was to test if scipy's filter also has that 1/sqrt(up) issue. yes, that's a plotting problem when using different length. normalize for different df
# print("resample shape", y.shape)
upsamp.big_interp(samp_bank[0,:20000+2*bw],h,10,y,bw)

# plot_spectra(samp_bank[1,:], 2000)
plot_spectra(samp_bank[0,:], 2000,both=False)
plot_spectra(y[:200000], 2000,both=False)
plot_spectra(samp_bank[1,:], 2000,both=False)



# plt.plot(np.cumsum(samp_bank[2,:]))
# plt.show()
# sys.exit()
# print(samp_ptr)
# sys.exit()
# print("ctrs",ctr)
# # plot_spectra(samp_bank[1,:],2048)
# # plot_spectra(samp_bank[2,:],2048)
# navg=2
# spec=np.zeros(10**nlevels+1,dtype='float64')
# for i in range(navg):
#     yy=generate(2*10**nlevels, bank,samp_bank,rand_bank,nlevels,osamp_coeffs,krig_ptr,samp_ptr,hf,sigma,bw, krig_bank_size, samp_bank_size, krig_len, i*2000)
#     spec[:] += np.abs(np.fft.rfft(yy))**2
#     print(samp_ptr)
#     # square_add(np.fft.rfft(yy),spec)
#     print("done",i)
# spec[:]=spec/navg
# plt.loglog(spec)
# plt.show()
# sys.exit()
t1=time.time()
yy = generate2(
    2000000,
    bank,
    samp_bank,
    rand_bank,
    nlevels,
    osamp_coeffs,
    krig_ptr,
    samp_ptr,
    hf,
    sigma,
    bw,
    krig_bank_size,
    samp_bank_size,
    krig_len,
)
t2=time.time()
print("time taken", (t2-t1)/2000000)

t1=time.time()
yy = generate2(
    2000000,
    bank,
    samp_bank,
    rand_bank,
    nlevels,
    osamp_coeffs,
    krig_ptr,
    samp_ptr,
    hf,
    sigma,
    bw,
    krig_bank_size,
    samp_bank_size,
    krig_len,
)
t2=time.time()
print("time taken", (t2-t1)/2000000)

t1=time.time()
generate_rand(2000000,2)
t2=time.time()
print("time taken", (t2-t1)/2000000)

t1=time.time()
generate_rand(2000000,2)
t2=time.time()
print("time taken", (t2-t1)/2000000)


plt.clf()
f=plt.gcf()
f.set_size_inches(10,4)
plt.title('Zoomed-in noise')
plt.plot(yy[:2000])
plt.show()

# plt.clf()
# plt.loglog(np.abs(np.fft.rfft(yy)))
# plt.show()
# plot_spectra(yy, 20000)

# # print(yy-samp_bank[2,:])
# plt.title(f"CUMSUM of {nlevels} decades, 2M points")
# plt.plot(np.cumsum(yy))
# plt.show()
