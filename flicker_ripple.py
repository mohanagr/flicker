import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz
from scipy.special import sici
import time
import numba as nb
import sys
from rocket_fft import c2r
from scipy.signal import welch, firwin, resample_poly, upfirdn
import concurrent.futures

import time

np.random.seed(int(time.time()))


def get_acf(tau, f1, f2):
    s1, c1 = sici(2 * np.pi * f1 * tau)
    s2, c2 = sici(2 * np.pi * f2 * tau)
    y = 2 * (c2 - c1)
    y = np.nan_to_num(y, nan=2 * np.log(f2 / f1) + 1e-5)  # was 1e-5
    return y  # *1e-15


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


@nb.njit(nogil=True, cache=True)
def generate(
    n,
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
):
    NUM_STACK = np.zeros(
        nlevels + 1, dtype="int64"
    )  # recursion stack should not exceed number of levels but still +1 for good luck
    LEV_STACK = np.zeros(nlevels + 1, dtype="int64")
    NUM_STACK[0] = krig_bank_size
    LEV_STACK[0] = nlevels - 1
    y = np.zeros(n, dtype="float64")
    NUM_PTR = 0
    norm = 1 / (krig_bank_size + krig_len)
    upper = 2 * bw
    up = osamp_coeffs.shape[0]
    while NUM_PTR >= 0:
        # print("NUM STACK", NUM_STACK)
        # print("LEV STACK", LEV_STACK)
        # print("NUM PTR", NUM_PTR)
        curpt = NUM_STACK[NUM_PTR]
        curll = LEV_STACK[NUM_PTR]
        NUM_PTR -= 1
        rownum = curpt % up
        # print("Popped point", curpt, "level", curll, "rownum is", rownum, "krig ptr @ ", krig_ptr[ll])
        tot = bank[curll, krig_len + krig_ptr[curll]]  # bank of krigs
        if curll > 0:
            tot = (
                tot
                + osamp_coeffs[rownum, :]
                @ samp_bank[
                    curll - 1, samp_ptr[curll - 1] : samp_ptr[curll - 1] + upper
                ]
            )
            # print("tot is", tot)
            # print("curll>0, inside if block")
            if curll == nlevels - 1:
                # print("NUM STACK", NUM_STACK)
                # print("LEV STACK", LEV_STACK)
                # print("NUM PTR", NUM_PTR)
                y[curpt - krig_bank_size] += tot
                NUM_PTR += 1
                NUM_STACK[NUM_PTR] = curpt + 1
                LEV_STACK[NUM_PTR] = nlevels - 1
                futpt = curpt + 1
                if futpt == krig_bank_size + n:
                    break
                for ii in range(nlevels - 1):
                    rem = futpt % up
                    quo = futpt // up
                    if rem > 0:
                        break
                    else:
                        futpt = quo
                        NUM_PTR += 1
                        NUM_STACK[NUM_PTR] = quo  # 20 is 2 for level previous to it
                        LEV_STACK[NUM_PTR] = nlevels - 2 - ii

        samp_ptr[curll] += 1
        if samp_ptr[curll] > samp_bank_size - 2 * bw:
            samp_bank[curll, : bw + bw - 1] = samp_bank[curll, 1 - bw - bw :]
            # say bw = 64. consider final 64 elements: idx 0-63. when we at 0, we still have 64 avl.
            # when we at 1, we have only 63 remaining.
            # copy these 63 back to beginning
            samp_ptr[curll] = 0
        # and point 64, idx 63 is now the tot we evalulated
        samp_bank[curll, samp_ptr[curll] + bw + bw - 1] = tot
        # print(f"For {curll} samp_bank[{samp_ptr[curll]}]")
        # handle bank rotations
        krig_ptr[curll] += 1
        # print("krig_ptr for", curll, "+1")
        if krig_ptr[curll] == krig_bank_size:
            # used up all the krig'd values. generate next chunk
            noise = sigma * np.random.randn(
                krig_bank_size + krig_len
            )  # generate bigger
            noise[:krig_len] = rand_bank[
                curll, :
            ]  # then replace first krig_len with stored last randn
            rand_bank[curll, :] = noise[
                -krig_len:
            ]  # store the last krig_len noise for next generation
            # np.fft.irfft(hf*np.fft.rfft(noise),out=bank[curll,:])
            c2r(
                hf * np.fft.rfft(noise),
                bank[curll, :],
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
            krig_ptr[curll] = 0
    # print("final y is", y)
    return y


@nb.njit(nogil=True)
def generate_rand(n, sigma):
    y = sigma * np.random.randn(n)
    return y


def plot_spectra(y, size):
    f, P = welch(y, nperseg=size, noverlap=size // 2)
    spec = y.reshape(-1, size)
    spec = np.mean(np.abs(np.fft.rfft(spec, axis=1)) ** 2, axis=0)
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
f1 = 0.995 * f2 / up
# f1=f2/up
N = 2 * 1000
# ps=np.zeros(N//2+1,dtype='complex128')
# ps[int(f1*N):int(f2*N)+1]=1/np.arange(int(f1*N),int(f2*N)+1) #N/2 is the scaling factor to line the two PS up.
# acf_dft=N*np.fft.irfft(ps)
# acf_anl=get_acf(np.arange(0,N//2+1),f1,f2)

coeff_len = 2048
krig_len = 2048
acf_anl = get_acf(
    np.arange(0, coeff_len), f1, f2
)  # + 500*np.cos(np.arange(0,coeff_len)*2*np.pi*f2)
C = toeplitz(acf_anl)
Cinv = np.linalg.inv(C)
print(Cinv @ C)
print(C @ Cinv)
vec = get_acf(np.arange(0, coeff_len) + 1, f1, f2)
vec = vec[::-1]
coeffs = vec.T @ Cinv
sigma = np.sqrt(C[0, 0] - vec @ Cinv @ vec.T)
print("krig stddev", sigma)
# plt.plot(coeffs)
# plt.show()
delta = np.zeros(krig_len)
delta[0] = 1
fir = get_impulse(delta, coeffs)  # size of krig coeffs can be different, don't matter.
# plt.plot(fir)
# plt.show()
krig_bank_size = 2048
hf = np.fft.rfft(np.hstack([fir, np.zeros(krig_bank_size)]))  # transfer function
# hf = np.fft.rfft(np.hstack([fir,np.zeros(krig_bank_size+200)])) #transfer function + 200 for manual osamp later
# design a filter to replace osamp_coeffs
import upsample_poly as upsamp

half_size = 128
bw = half_size
h = up * firwin(2 * half_size * up + 1, 1 / up, window=("kaiser", 1))
print("len h", len(h))
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

nlevels = 3

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
# let's try to forward generate two levels, long timestream and look at spectra.
for ll in range(1, nlevels):
    print("processing level", ll, "parent", ll - 1)
    for i in range(samp_bank.shape[1]):
        # generate level's own krig - already there!
        #         print("samp bank begin", samp_bank[ll,i])
        # print(f"filling level {ll}, sample {i}, krig_ptr self @ {krig_ptr[ll]}, ctr prev at {ctr[ll-1]}")
        krig_samp_own = bank[ll, krig_len + krig_ptr[ll]]
        krig_ptr[ll] += 1
        if krig_ptr[ll] == krig_bank_size:
            # used up all the krig'd values. generate next chunk
            # print("ran out of krig. resetting", ll)
            gen_krig(bank, rand_bank, ll, hf, sigma, krig_bank_size, krig_len)
            krig_ptr[ll] = 0
        samp_bank[ll, i] += krig_samp_own
        rownum = i % up
        if i > 0 and rownum == 0:
            ctr[ll - 1] += 1
            samp_ptr[ll - 1] += 1
        samp_bank[ll, i] += (
            osamp_coeffs[rownum, :]
            @ samp_bank[ll - 1, ctr[ll - 1] : ctr[ll - 1] + 2 * bw]
        )

krig_ptr[0] = samp_ptr[0]
print("ctrs", ctr)
print("krig pts", krig_ptr)
print("krig pts", krig_ptr)
# sys.exit()
# plot_spectra(samp_bank[1,:],2048)
# plot_spectra(samp_bank[2,:],2048)
# navg=100
# spec=np.zeros(10**nlevels+1,dtype='float64')
# for top in range(navg):
#     rand_bank[:, :] = sigma*np.random.randn(nlevels*krig_len).reshape(nlevels,krig_len) #white noise bank
#     krig_ptr[:]=0
#     samp_ptr[:]=0
#     bank[:]=0
#     for jj in range(nlevels):
#         gen_krig(bank, rand_bank, jj, hf, sigma, krig_bank_size, krig_len)
#     # plot_spectra(bank[0,krig_len:],200)
#     # plot_spectra(bank[1,krig_len:],200)
#     # sys.exit()
#     samp_bank[:]=0
#     samp_bank[0,:] = bank[0,krig_len:].copy() #topmost level just krig
#     samp_bank_size = samp_bank.shape[1]
#     ctr=[0]*nlevels
#     #let's try to forward generate two levels, long timestream and look at spectra.
#     for ll in range(1,nlevels):
#         # print("processing level", ll, "parent", ll-1)
#         for i in range(samp_bank.shape[1]):
#             #generate level's own krig - already there!
#         #         print("samp bank begin", samp_bank[ll,i])
#             krig_samp_own = bank[ll,krig_len+krig_ptr[ll]]
#             krig_ptr[ll] +=1
#             if krig_ptr[ll] == krig_bank_size:
#                 #used up all the krig'd values. generate next chunk
#                 # print("ran out of krig. resetting", ll)
#                 gen_krig(bank, rand_bank, ll, hf, sigma, krig_bank_size, krig_len)
#                 krig_ptr[ll] = 0
#             samp_bank[ll,i] += krig_samp_own
#             rownum = i%up
#             if i > 0 and rownum == 0:
#                 ctr[ll-1]+=1
#                 samp_ptr[ll-1]+=1
#             samp_bank[ll,i] += (osamp_coeffs[rownum,:]@samp_bank[ll-1,ctr[ll-1]:ctr[ll-1]+2*bw])
#     yy=generate(2*10**nlevels, bank,samp_bank,rand_bank,nlevels,osamp_coeffs,krig_ptr,samp_ptr,hf,sigma,bw, krig_bank_size, samp_bank_size, krig_len)
#     # spec[:] += np.abs(np.fft.rfft(yy))**2
#     square_add(np.fft.rfft(yy),spec)
#     print("done",top)
#     # if i%10==0: print("done", i)
# spec[:]=spec/navg
# plt.loglog(spec)
# plt.show()
# sys.exit()

yy = generate(
    10 ** (5),
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
# plot_spectra(yy[1:],20000)
# yy=yy-np.mean(yy)
# plt.plot(yy)
# plt.show()
plt.title("Cumsum of 1/f, 5 decades")
plt.plot(np.cumsum(yy))
plt.show()

# plt.plot(np.cumsum(yy[200000+1:]))
# plt.show()
# plt.loglog(spec)
# plt.show()

sys.exit()

krig_bank_size = 20000 * 100
hf = np.fft.rfft(np.hstack([fir, np.zeros(krig_bank_size + 4 * half_size)]))
rn = sigma * np.random.randn(
    len(fir) + krig_bank_size + 4 * half_size
)  # extra hundred to make my oversampling easy
yy = np.fft.irfft(np.fft.rfft(rn) * hf)[krig_len:]
bigyy = np.zeros(krig_bank_size * up + 2 * half_size, dtype=yy.dtype)
print("len yy", len(yy))

upsamp.big_interp(yy, h, up, bigyy, half_size)

print("len bigyy", len(bigyy))

# plot_spectra(bigyy,2000)
# print("done")

# make some more and add to oversampled one
hf = np.fft.rfft(np.hstack([fir, np.zeros(krig_bank_size * up + 2 * half_size)]))
rn = sigma * np.random.randn(len(fir) + krig_bank_size * up + 2 * half_size)
yy2 = np.fft.irfft(np.fft.rfft(rn) * hf)[krig_len:]

yytot = bigyy + yy2

bigyy2 = np.zeros(krig_bank_size * up, dtype=yy.dtype)

upsamp.big_interp(yytot, h, up, bigyy2, half_size)

hf = np.fft.rfft(np.hstack([fir, np.zeros(krig_bank_size * up)]))
rn = sigma * np.random.randn(len(fir) + krig_bank_size * up)
yy2 = np.fft.irfft(np.fft.rfft(rn) * hf)[krig_len:]

yytot = bigyy2 + yy2
# calc spectra again
plot_spectra(yytot, 2000)
# yytot = bigyy2 + yy2
# plot_spectra(yytot,2000)
