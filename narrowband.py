import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz
from scipy.special import sici
import time
import numba as nb
import sys
from rocket_fft import c2r
import concurrent.futures
import sys

np.random.seed(42)


def get_acf(tau, df):
    y = np.sinc(df * tau)
    y[0] += 1e-6
    return y


def get_impulse(x, coeffs):
    n = len(coeffs)
    y_big = np.zeros(len(x) + len(coeffs))
    for i in range(len(x)):
        y_big[i + n] = coeffs @ y_big[i : n + i] + x[i]
    return y_big[n:]


# @nb.njit()
def gen_krig(bank, rand_bank, hf, sigma, krig_bank_size, krig_len):
    # FIRST krig_len points in the OUTPUT will be crap. only use after that.
    # len(fir) = krig_len
    norm = 1 / (krig_bank_size + krig_len)
    noise = sigma * np.random.randn(krig_bank_size + krig_len)  # generate bigger
    # print("noise shape", noise.shape)
    print("end of noise", noise[-krig_len:])
    noise[:krig_len] = rand_bank[
        :
    ]  # then replace first krig_len with stored last randn
    print("beg of noise", noise[:krig_len])
    rand_bank[:] = noise[-krig_len:]  # save the last krig_len noise for next generation
    # print("out array shape", bank.shape, "input shape", hf.shape)
    np.fft.irfft(hf * np.fft.rfft(noise), out=bank[:])
    # c2r(hf*np.fft.rfft(noise),bank[:],np.asarray([0,],dtype='int64'),False,norm,16)


@nb.njit(nogil=True)
def generate(
    n,
    bank,
    samp_bank,
    rand_bank,
    nlevels,
    coeffs,
    osamp_coeffs,
    krig_ptr,
    samp_ptr,
    hf,
    sigma,
    bw,
    krig_bank_size,
    samp_bank_size,
    krig_len,
    enable,
):
    NUM_STACK = np.zeros(
        nlevels + 1, dtype="int64"
    )  # recursion stack should not exceed number of levels but still +1 for good luck
    LEV_STACK = np.zeros(nlevels + 1, dtype="int64")
    NUM_STACK[0] = 1
    LEV_STACK[0] = nlevels - 1
    y = np.zeros(n + 1, dtype="float64")
    NUM_PTR = 0
    norm = 1 / (krig_bank_size + krig_len)
    upper = 2 * bw
    while NUM_PTR >= 0:

        curpt = NUM_STACK[NUM_PTR]
        curll = LEV_STACK[NUM_PTR]
        NUM_PTR -= 1
        rownum = curpt % 10 - 1  # row 0 is 0.1
        # print("Popped point", curpt, "level", curll, "rownum is", rownum+1)
        tot = bank[curll, krig_len + krig_ptr[curll]]
        if curll > 0:
            if rownum == -1:  # %10 is zero
                # bank ptr starts at krig len because first krig_len numbers are trash from circular convolution
                tot = tot + samp_bank[curll - 1, samp_ptr[curll - 1] + bw - 1]
            else:
                tot = (
                    tot
                    + osamp_coeffs[rownum, :]
                    @ samp_bank[
                        curll - 1, samp_ptr[curll - 1] : samp_ptr[curll - 1] + upper
                    ]
                )
                # print("tot is", tot)
            if curll == nlevels - 1:
                y[curpt] += tot
                if enable:
                    y[curpt] = y[curpt] + y[curpt - 1] + 1e-4 * np.random.randn(1)[0]
                NUM_PTR += 1
                NUM_STACK[NUM_PTR] = curpt + 1
                LEV_STACK[NUM_PTR] = nlevels - 1
                futpt = curpt + 1
                if futpt == n + 1:
                    break
                for ii in range(nlevels - 1):
                    rem = futpt % 10
                    quo = futpt // 10
                    if rem > 0:
                        break
                    else:
                        futpt = quo
                        NUM_PTR += 1
                        NUM_STACK[NUM_PTR] = quo  # 20 is 2 for level previous to it
                        LEV_STACK[NUM_PTR] = nlevels - 2 - ii

        samp_ptr[curll] += 1
        if (
            samp_ptr[curll] > samp_bank_size - 2 * bw
        ):  # for bottommost level samp_ptr should never move
            samp_bank[curll, : bw + bw - 1] = samp_bank[curll, 1 - bw - bw :]
            samp_ptr[curll] = 0
        samp_bank[curll, samp_ptr[curll] + bw + bw - 1] = (
            tot  # next element after 2bw-1 is 2*bw but we moved sampt ptr ahead earlier
        )

        # handle bank rotations
        krig_ptr[curll] += 1
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
    return y


@nb.njit(nogil=True)
def generate_rand(n, sigma):
    y = sigma * np.random.randn(n)
    return y


def get_next_y(y_past, x, coeffs):
    n = len(y_past)
    print("x passed", x)
    print("y passed", y_past)
    y_big = np.hstack([y_past, np.zeros(len(x))])
    print("y big shape", y_big.shape)
    print("coeffs shape", coeffs.shape)
    for i in range(len(x)):
        y_big[i + n] = coeffs @ y_big[i : n + i] + x[i]
    # plt.plot(y_big)
    # plt.show()
    return y_big[n:]


df = 0.4
N = 2 * 1000
ps = np.zeros(N // 2 + 1, dtype="complex128")
ps[: int(df / 2 * N) + 1] = 1 / (df * N + 1)
acf_dft = N * np.fft.irfft(ps)
acf_anl = get_acf(
    np.arange(0, N // 2 + 1), df
)  # make sure both match bc we will generate initial realization from ps

plt.plot(acf_dft[:1000])
plt.plot(acf_anl[:1000])
plt.show()

coeff_len = 1024
krig_len = 1024
krig_bank_size = 1024 * 63
acf_anl = get_acf(np.arange(0, coeff_len), df)
C = toeplitz(acf_anl)
Cinv = np.linalg.inv(C)

vec = get_acf(np.arange(0, coeff_len) + 1, df)
vec = vec[::-1]
coeffs = vec.T @ Cinv
sigma = np.sqrt(C[0, 0] - vec @ Cinv @ vec.T)
print("krig stddev", sigma)

# sys.exit()

bank = np.zeros(krig_len + krig_bank_size, dtype="float64")
rand_bank = np.zeros(
    krig_len, dtype="float64"
)  # only gotta store the last krig_len rand
rand_bank[:] = sigma * np.random.randn(krig_len)

delta = np.zeros(
    krig_len
)  # how far back you wanna convolve, length of FIR filter essentially
delta[0] = 1
fir = get_impulse(delta, coeffs)  # size of krig coeffs can be different, don't matter.

# plt.plot(fir)
# # plt.show()
# # sys.exit(0)
# print("firt shape", fir.shape)
# plt.title("impulse response")
# plt.show()
hf = np.fft.rfft(np.hstack([fir, np.zeros(krig_bank_size)]))  # transfer function
# print(hf.shape)

# burn in the krig_bank
# print("end of rand bank",rand_bank[-krig_len:])
# gen_krig(bank, rand_bank, hf, sigma, krig_bank_size, krig_len)
# print("end of rand bank",rand_bank[-krig_len:])
# a1=bank[-krig_len:].copy()
# print("a1",a1)
# gen_krig(bank, rand_bank, hf, sigma, krig_bank_size, krig_len)
# a2=bank[krig_len:2*krig_len].copy()
# print("a2", a2)
print(len(fir), krig_len)
# plt.plot(np.hstack([fir,np.zeros(krig_bank_size)]));plt.show()
myrand = sigma * np.random.randn(krig_len + krig_bank_size)
# 1024     1024     1024    1024 ... 1024
# ign      coeff     out
# first out block has output of last in block
# second out block has output of first in block
# myrand[-krig_len]=0
krig_out = np.fft.irfft(hf * np.fft.rfft(myrand))
# krig_manual = np.zeros(krig_bank_size + krig_len)
krig_manual = np.zeros(krig_bank_size + krig_len)
# krig_manual[krig_len:2*krig_len] = krig_out[krig_len:2*krig_len]
# krig_manual[:krig_len] = krig_out[krig_len:2*krig_len] #second block has output of first krig_len randn's

for ii in range(0, krig_bank_size):
    krig_manual[ii + krig_len] = (
        krig_manual[ii : ii + krig_len] @ coeffs + myrand[ii + krig_len]
    )


# plt.plot(fir[::-1])
# plt.plot(coeffs)
# plt.show()
# print("fir way", fir[::-1]@myrand[1025:2049])
# print("krig way", krig_manual[:krig_len]@coeffs + myrand[2049])
# print(len(krig_manual), len(krig_out)-krig_len)

# following totally equal
print(krig_manual[:krig_len])
plt.plot(krig_out[krig_len:], label="out")
plt.plot(krig_manual[krig_len:], label="manual")
plt.legend()

plt.show()


plt.plot(np.abs(np.fft.rfft(krig_out[krig_len:])))
plt.show()
sys.exit()

noise[-krig_len:] = myrand
noise[:krig_len] = myrand
plt.plot(noise)
plt.show()
res = np.fft.irfft(hf * np.fft.rfft(noise))
print(res[krig_len : 2 * krig_len])
print(res[:krig_len])
sys.exit()
# prev=bank[krig_len:2*krig_len]
# test_ts = np.zeros(krig_len,dtype='float64')
# for i in range(krig_len):
#     test_ts[i]=

# plt.title("rand level 0");plt.show()
# sys.exit(0)
