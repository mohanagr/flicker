import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz
from scipy.special import sici
import time
import numba as nb
import sys
from rocket_fft import c2r
import concurrent.futures
from scipy.signal import firwin,welch
import sys
from upsample_poly import big_interp_c
from scipy.interpolate import CubicSpline, splrep, splev
import os

os.environ["NUMBA_OPT"] = "3"
os.environ["NUMBA_LOOP_VECTORIZE"] = "1"
os.environ["NUMBA_ENABLE_AVX"] = "1"

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

@nb.njit(cache=True)
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

@nb.njit(cache=True,nogil=True)
def _generate(
    y,
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
    krig_len,
    STACK,
    STACK_PTR,
    circular_buffer_offsets,
):
    start_pt = STACK[0,1]
    tot=0
    norm = 1 / (krig_bank_size + krig_len)
    upper = 2 * bw
    up = osamp_coeffs.shape[0]
    # nb.gdb_init()
    while STACK_PTR >= 0:
        # peek
        lev, pt = STACK[STACK_PTR, :]
        quo = pt // up
        rem = pt % up
        cbop = circular_buffer_offsets[lev - 1]
        # print(f"lev {lev} pt {pt} quo {quo} rem {rem} cbo of prev lev {cbop}")
        # if this level doesn't have enough krig vals left, regenerate
        if krig_ptr[lev] == krig_bank_size:
            # print(f"regenerating krig for lev {lev}")
            noise = sigma * np.random.randn(
                krig_bank_size + krig_len
            )  # generate bigger
            noise[:krig_len] = rand_bank[
                lev, :
            ]  # then replace first krig_len with stored last randn
            rand_bank[lev, :] = noise[
                -krig_len:
            ]  # store the last krig_len noise for next generation
            #     # np.fft.irfft(hf*np.fft.rfft(noise),out=bank[lev,:])
            c2r(
                hf * np.fft.rfft(noise),
                krig_bank[lev, :],
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
            krig_ptr[lev] = 0
        # check visited status
        if lev > 0 and (
            (quo + upper - 1) > (samp_ptr[lev - 1] + cbop)
        ):  # keep checking ahead
            # print(f"checking quo+upper {quo+upper -1} > {samp_ptr[lev-1]+cbop}")
            # parent hasn't been visted yet
            # push
            STACK_PTR += 1
            STACK[STACK_PTR, 0] = lev - 1
            STACK[STACK_PTR, 1] = quo + upper - 1
            samp_ptr[lev - 1] += 1
            continue

        tot = krig_bank[lev, krig_len + krig_ptr[lev]]
        if lev > 0:
            tot += (
                osamp_coeffs[rem, :]
                @ samp_bank[lev - 1, quo - cbop : quo - cbop + upper]
            )
        krig_ptr[lev] += 1

        if lev == nlevels - 1:
            # print("adding pt + 1 to stack")
            # print(f"dot product for lev {lev}")
            # print(f"lev {lev-1} starting {quo}-{cbop} = {quo - cbop}")
            y[pt-start_pt] = tot
            if pt == start_pt + n - 1:
                break
            # STACK_PTR+=1 #pop and push combined
            STACK[STACK_PTR, 0] = nlevels - 1
            STACK[STACK_PTR, 1] = pt + 1
        else:
            if samp_ptr[lev] > krig_bank_size - 1:
                circular_buffer_offsets[lev] = pt - upper + 1
                # print(f"cbo for lev {lev} set to {circular_buffer_offsets[lev]}")
                samp_ptr[lev] = (
                    upper - 1
                )  # this just indicates the tail of moving window; point is INCLUDED
                # print("the first point is now,", np.arange(0,200)[-upper+1])
                samp_bank[lev, : upper - 1] = samp_bank[lev, -upper + 1 :]
            samp_bank[lev, samp_ptr[lev]] = tot
            STACK_PTR -= 1  # pop
        # print("-----------------------------------------------------------")
    return y

class flicker:
    def __init__(self, nlevels, nsamp, f1, f2, cache_file=None):

        if cache_file:
            pass
        else:
            self.nlevels = nlevels
            self.nsamp = nsamp
            self.f1 = f1
            self.f2 = f2
            
            coeff_len = 1024
            krig_len = 1024  # number of coeffs in the FIR filter
            acf_anl = self.get_acf(np.arange(0, coeff_len), f1, f2)
            C = toeplitz(acf_anl)
            Cinv = np.linalg.inv(C)
            vec = self.get_acf(np.arange(0, coeff_len) + 1, f1, f2)
            vec = vec[::-1]
            coeffs = vec.T @ Cinv
            sigma = np.sqrt(C[0, 0] - vec @ Cinv @ vec.T)
            print("krig stddev", sigma)
            self.sigma = sigma
            up = 10
            bw = 32
            krig_bank_size = 20000
            self.krig_len = krig_len
            self.up = up
            self.bw = bw
            self.start_pt = 0
            self.krig_bank_size = krig_bank_size
            self.krig_bank = np.zeros(
                (nlevels, krig_len + krig_bank_size), dtype="float64"
            )  # KRIG BANK
            self.krig_ptr = np.zeros(nlevels, dtype="int64")
            self.samp_ptr = np.zeros(nlevels, dtype="int64")
            self.ybig = np.empty(
                nsamp, dtype="float64"
            )  # generation starts from sample no. 1
            self.rand_bank = np.zeros(
                (nlevels, krig_len), dtype="float64"
            )  # only gotta store the last krig_len rand for future kring generation
            self.rand_bank[:] = sigma * np.random.randn(nlevels * krig_len).reshape(
                nlevels, krig_len
            )

            delta = np.zeros(
                krig_len
            )  # how far back you wanna convolve, length of FIR filter essentially
            delta[0] = 1
            fir = self.get_impulse(
                delta, coeffs
            )  # size of krig coeffs can be different, don't matter.

            self.hf = np.fft.rfft(
                np.hstack([fir, np.zeros(self.krig_bank_size)])
            )  # krig transfer function

            # first len(fir) is garbage. after len(fir), krig timestream continues.

            self.h = up*firwin(2 * bw * up + 1, 1 / up, window=("kaiser", 1))
            
            self.osamp_coeffs = self.h[:-1].reshape(-1, up).T[:, ::-1].copy()
            self.samp_bank = np.zeros(
                (nlevels, krig_bank_size), dtype=self.krig_bank.dtype
            )  # krig + white bank
            
            self.init_banks()
            self.init_pointers()

    def init_banks(self):
        for jj in range(self.nlevels):
            gen_krig(
                self.krig_bank,
                self.rand_bank,
                jj,
                self.hf,
                self.sigma,
                self.krig_bank_size,
                self.krig_len,
            )
        self.samp_bank[0, :] = self.krig_bank[0, self.krig_len:]  # topmost level just krig
        ctr = [0] * self.nlevels
        # Forward generate all but the topmost and the bottommost level
        for ll in range(
            1, self.nlevels
        ):  # generating bottomost but it doesnt matter, we wont use it
            # print("processing level", ll, "parent", ll - 1)
            for i in range(self.samp_bank.shape[1]):
                krig_samp_own = self.krig_bank[ll, self.krig_len + self.krig_ptr[ll]]
                if self.krig_ptr[ll] == self.krig_bank_size:
                    # used up all the krig'd values. generate next chunk
                    # print("ran out of krig. resetting", ll)
                    gen_krig(
                        self.bank,
                        self.rand_bank,
                        ll,
                        self.hf,
                        self.sigma,
                        self.krig_bank_size,
                        self.krig_len,
                    )
                    self.krig_ptr[ll] = 0
                self.samp_bank[ll, i] += krig_samp_own
                rownum = i % self.up
                if i > 0 and rownum == 0:
                    ctr[ll - 1] += 1
                # print("level", ll, "samp", i, "krig ptr", krig_ptr[ll], "counters", ctr)
                self.samp_bank[ll, i] += (
                    self.osamp_coeffs[rownum, :]
                    @ self.samp_bank[ll - 1, ctr[ll - 1] : ctr[ll - 1] + 2 * self.bw]
                )
                self.krig_ptr[ll] += 1
        self.samp_ptr[:] = (
            self.krig_bank_size - 1
        )  # rolling pointer that tracks the end of the 2*bw window
        self.samp_ptr[-1] = (
            123456  # test value to make sure final level's ptr is not touched.
        )
        self.krig_ptr[-1] = 0
        self.krig_ptr[0] = self.krig_bank_size

    def init_pointers(self):
        #at the end of each generate call, stack_ptr will always end up back at zero.
        #generating final level's point means that all parent levels have been generated already
        self.STACK = np.zeros((self.nlevels + 1, 2), dtype="int64") #col 0 - levels, col 1 - point number
        self.STACK_PTR = 0
        self.circular_buffer_offsets = np.zeros(self.nlevels, dtype="int64")
        self.STACK[0,0]=self.nlevels-1
        self.STACK[0,1]=self.start_pt

    def get_acf(self, tau, f1, f2):
        s1, c1 = sici(2 * np.pi * f1 * tau)
        s2, c2 = sici(2 * np.pi * f2 * tau)
        y = 2 * (c2 - c1)
        y = np.nan_to_num(y, nan=2 * np.log(f2 / f1) + 1e-4)  # was 1e-5
        return y

    def get_impulse(self, x, coeffs):
        n = len(coeffs)
        y_big = np.zeros(len(x) + len(coeffs))
        for i in range(len(x)):
            y_big[i + n] = coeffs @ y_big[i : n + i] + x[i]
        return y_big[n:]

    def generate(self):
        _generate(self.ybig,
        self.nsamp,
        self.krig_bank,
        self.samp_bank,
        self.rand_bank,
        self.nlevels,
        self.osamp_coeffs,
        self.krig_ptr,
        self.samp_ptr,
        self.hf,
        self.sigma,
        self.bw,
        self.krig_bank_size,
        self.krig_len,
        self.STACK,
        self.STACK_PTR,
        self.circular_buffer_offsets,)
        assert self.STACK_PTR == 0
        self.start_pt += self.nsamp
        self.STACK[0,0]=self.nlevels-1
        self.STACK[0,1]=self.start_pt
@nb.njit(cache=True)
def cumsum(y,x,start,scale):
    nn=len(y)
    y[0] = x[0]*scale + start
    for i in range(1,nn):
        y[i]=x[i]*scale+y[i-1]
    return y

if __name__ == "__main__":

    nlevels=13
    up = 10
    f2 = 1 / 2
    f1 = 0.993 * f2 / up
    # nsamp=2048*500
    nsamp=2000000
    
    clock = flicker(nlevels, nsamp, f1, f2)
    clock.generate()
    # plt.loglog(np.abs(np.fft.rfft(clock.h)))
    # plt.show()
    plot_spectra(clock.ybig,200000)
    # plt.loglog(np.abs(np.fft.rfft(clock.ybig)))
    # plt.show()
    # navg=100
    # spec=np.zeros(10**nlevels+1,dtype='float64')
    # for i in range(navg):
    #     clock = flicker(nlevels, nsamp, f1, f2)
    #     clock.generate()
    #     spec[:] += np.abs(np.fft.rfft(clock.ybig))**2
    #     # print(samp_ptr)
    #     # square_add(np.fft.rfft(yy),spec)
    #     print("done",i)
    # spec[:]=spec/navg
    # plt.loglog(spec)
    # plt.show()

    # clock.generate()
    csum=np.empty(nsamp,dtype='float64')
    csum2 = np.cumsum(clock.ybig)*10**(-0.5*nlevels)
    cumsum(csum,clock.ybig,0,10**(-clock.nlevels/2))
    print(clock.ybig)
    print(csum)
    print(csum2)
    plt.title(f"CUMSUM of {nlevels} decades, 2M points")
    plt.plot(csum)
    plt.show()
    plt.title(f"CUMSUM of {nlevels} decades, 2M points")
    plt.plot(csum2)
    plt.show()

