import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz
from scipy.special import sici
import time
import numba as nb
import sys


##
#  FFT method works for caclulating contribution of random numbers
#  Still need to get the contribution of past realization working
#
##


def get_acf(tau, f1, f2):
    s1, c1 = sici(2 * np.pi * f1 * tau)
    s2, c2 = sici(2 * np.pi * f2 * tau)
    y = 2 * (c2 - c1)
    y = np.nan_to_num(y, nan=2 * np.log(f2 / f1) + 1e-3)
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


# def get_next_y(y_past,x,coeffs):
#     n=len(y_past)
#     print("x passed", x)
#     print("y passed", y_past)
#     y_big = np.hstack([y_past,np.zeros(len(x))])
#     print(y_big.shape)
#     print(coeffs.shape)
#     for i in range(len(x)):
#         y_big[i+n] = coeffs@y_big[i:n+i] + x[i]
#     # plt.plot(y_big)
#     # plt.show()
#     return y_big[n:]


def fir_krig(x0, coeffs):
    x = np.zeros(len(coeffs))
    x[0] = x0
    for i in range(1, len(x)):
        # print(x[:i], coeffs[:i])
        x[i] = coeffs[-i:] @ x[:i]
        print(x[i])
    return x


np.random.seed(42)
f1 = 0.05
f2 = 0.5
N = 2 * 1000
ps = np.zeros(N // 2 + 1, dtype="complex128")
ps[int(f1 * N) : int(f2 * N) + 1] = 1 / np.arange(
    int(f1 * N), int(f2 * N) + 1
)  # N/2 is the scaling factor to line the two PS up.
# acf_dft=N*np.fft.irfft(ps)
# acf_anl=get_acf(np.arange(0,N//2+1),f1,f2)

nsamps = 2000
acf_anl = get_acf(np.arange(0, nsamps), f1, f2)
C = toeplitz(acf_anl)
Cinv = np.linalg.inv(C)
vec = get_acf(np.arange(0, nsamps) + 1, f1, f2)
vec = vec[::-1]
coeffs = vec.T @ Cinv
sigma = np.sqrt(C[0, 0] - vec @ Cinv @ vec.T)

# plt.plot(coeffs);plt.show()
noise = N * np.fft.irfft(
    np.sqrt(ps / 2) * (np.random.randn(N // 2 + 1) + 1j * np.random.randn(N // 2 + 1))
)

rand_bank = sigma * np.random.randn(nsamps)

yy = np.zeros(2 * nsamps, dtype="float64")
yy[:nsamps] = noise.copy()
# coeffs=np.asarray([0.5])
for i in range(nsamps):
    yy[nsamps + i] = (
        coeffs @ yy[nsamps + i - len(coeffs) : i + nsamps]
    )  # + rand_bank[i]


fk = fir_krig(1, coeffs)

plt.plot(fk)
plt.show()
# y_filt = get_next_y(noise,0*noise,coeffs)
plt.title("just one")
plt.plot(yy[nsamps:])

plt.show()

print("coeffs shape", coeffs.shape)
# coeff_rev = coeffs[::-1]
# cc=np.hstack([1,coeff_rev[:-1]])
# imp = np.fft.irfft(1/np.fft.rfft(cc))
# plt.plot(imp)
# plt.show()
# sys.exit(0)
delta = np.zeros(nsamps)
delta[0] = 1
fir = get_next_y(np.zeros(len(coeffs)), delta, coeffs)

plt.title("they same?")
plt.plot(fir)
plt.plot(fk)
plt.show()

pred = 0 * noise
# for i in range(nsamps):
#     pred[i] =
# fir2 = fir_krig(np.zeros(nsamps),delta,coeffs)
# plt.plot(fir2);plt.title("fir2");plt.show()

# coeffs_rev=coeffs[::-1]
# plt.plot(fir2[1:]-coeffs_rev[:-1])
# plt.show()


hf = np.fft.rfft(np.hstack([fir, 0 * fir]))
# hf2 = np.fft.rfft(np.hstack([fir2,0*fir2]))

randf = np.fft.rfft(np.hstack([rand_bank, np.zeros(len(rand_bank))]))
# prevf = np.fft.rfft(np.hstack([rand_bank]))
# print(hf.shape,prevf.shape)

yy_next = np.fft.irfft(hf * randf)[:nsamps]  # + extra[:nsamps]

# yy_next2 = get_next_y(noise,rand_bank,coeffs)

# print("error", np.sum(np.abs(yy_next-yy_next2)))

# plt.plot(yy_next2)
plt.plot(yy_next)
plt.plot(yy[nsamps:])
plt.show()
