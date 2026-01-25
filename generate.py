import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz
from scipy.special import sici
import time
import numba as nb


def get_acf(tau, f1, f2):
    s1, c1 = sici(2 * np.pi * f1 * tau)
    s2, c2 = sici(2 * np.pi * f2 * tau)
    y = 2 * (c2 - c1)
    y = np.nan_to_num(y, nan=2 * np.log(f2 / f1) + 1e-4)
    return y


@nb.njit()
def get_legendre(x, deg):
    # P0 = 1, P1 = x
    y = np.zeros(deg + 1, dtype="float64")
    for i in range(deg + 1):
        if i == 0:
            y[i] = 1
        elif i == 1:
            y[i] = x
        else:
            y[i] = (2 * i - 1) * x * y[i - 1] - (i - 1) * y[i - 2]
            y[i] /= i
    return y


def generate_data(legcoeffs, vsT, bank, ndata, nlevels):
    mult = 10  # 10x every level
    spec = np.zeros(ndata)
    maxs = 8
    polyord = 8
    for i in range(0, nlevels + 1):
        dt = mult ** (i - nlevels)
        rand_bank = np.random.randn(
            int(ndata * dt) + 1
        )  # bank of random numbers for this level
        rand_ptr = 0
        print("level", i, dt)
        for j in range(ndata):
            frac = (j * dt) % 1
            pt = 1 if frac == 0 else frac
            # print("pt is", pt)
            coeffs = vsT @ legcoeffs @ get_legendre(pt, polyord)
            pred = coeffs @ bank[i]
            # if integer multiple
            if frac == 0.0:  # integers exactly rep. in floats
                # add noise
                krig_err = rand_bank[rand_ptr] * sigma
                pred += krig_err / np.sqrt(dt)
                bank[i] = np.roll(bank[i], -1)
                bank[i][-1] = pred
                rand_ptr += 1
            spec[j] += pred
    return spec


# @nb.njit()
def generate_data2(legcoeffs, vsT, bank, ndata, nlevels, coeff_arr):
    mult = 10.0  # 10x every level
    spec = np.zeros(ndata)
    maxs = 8
    polyord = 8
    nlevels = nlevels
    npoints = 1000  # cached data
    for j in range(ndata):
        # if j > 3: break
        for i in range(0, nlevels + 1):
            dt = mult ** (i - nlevels)
            # rand_bank = np.random.randn(int(ndata*dt)+1) #bank of random numbers for this level
            # rand_ptr=0

            frac = (j * dt) % 1
            pt = 1 if frac == 0 else frac
            if dt >= 0.001:
                # print(dt, pt, pt*npoints)
                pred = coeff_arr[int(np.round(pt * npoints)), :] @ bank[i]
            else:
                coeffs = (
                    vsT @ legcoeffs @ get_legendre(pt, polyord)
                )  ## OOM slower than dot ahead, 1e-5
                pred = coeffs @ bank[i]
            # print("dot", t2-t1)
            # if integer multiple
            if frac == 0.0:  # integers exactly rep. in floats
                # add noise
                krig_err = np.random.randn(1)[0] * sigma  # 7e-7
                pred += krig_err / np.sqrt(dt)
                bank[i] = np.roll(bank[i], -1)  # 4e-6
                bank[i][-1] = pred
            spec[j] += pred
            # print("data, level", j, i)
    return spec


nlevels = 4
nsamps = 1000
N = 2 * nsamps
f1 = 0.05
f2 = 0.5
bank = np.zeros((nlevels + 1, nsamps), dtype="float64")
ps = np.zeros(N // 2 + 1, dtype="complex128")
ps[int(f1 * N) : int(f2 * N) + 1] = 1 / np.arange(
    int(f1 * N), int(f2 * N) + 1
)  # N/2 is the scaling factor to line the two PS up.
for ll in range(nlevels + 1):
    # factor=np.sqrt(10**(nlevels-ll))
    # print(factor*10000)
    factor = 1
    noise = (
        factor
        * N
        * np.fft.irfft(
            np.sqrt(ps / 2)
            * (np.random.randn(N // 2 + 1) + 1j * np.random.randn(N // 2 + 1))
        )
    )
    bank[ll, :] = noise[:nsamps]

acf_anl = get_acf(np.arange(0, nsamps), f1, f2)
C = toeplitz(acf_anl)
Cinv = np.linalg.inv(C)
vec = get_acf(np.arange(0, nsamps) + 1, f1, f2)
vec = vec[::-1]
coeffs = vec.T @ Cinv
sigma = np.sqrt(C[0, 0] - vec @ Cinv @ vec.T)
print("krig sigma", sigma)

print("generating polynomials...")
npoints = 1000
dtaus = np.arange(0, npoints + 1) / npoints
coeff_arr = np.zeros((len(dtaus), nsamps))
my_tau = np.arange(0, nsamps)
dtaus[0] = 1e-14
for i, dtau in enumerate(dtaus):
    tau = my_tau + dtau
    vec = get_acf(tau, f1, f2)
    vec = vec[::-1]
    coeff = Cinv @ vec
    coeff_arr[i, :] = coeff
u, s, vT = np.linalg.svd(coeff_arr)
maxs = 8
svT = np.diag(s[:maxs]) @ vT[:maxs, :]
vsT = svT.T.copy()
polyord = 8
legcoeffs = np.zeros((maxs, polyord + 1), dtype="float64")  # singvals x order of poly+1

for i in range(maxs):
    colnum = i
    legcoeffs[i, :] = (
        np.linalg.pinv(
            np.polynomial.legendre.legvander(np.linspace(-1, 1, 1001), polyord)
        )
        @ u[:, colnum]
    )
print(legcoeffs)
print(legcoeffs.base, vsT.base, bank.base)
ndata = 2000000
print("generating noise...")
for i in range(1):
    t1 = time.time()
    spectrum = generate_data2(legcoeffs, vsT, bank, ndata, nlevels, coeff_arr)
    t2 = time.time()
    print(t2 - t1)

f = plt.gcf()
f.set_size_inches(10, 4)
plt.loglog(np.abs(np.fft.rfft(spectrum)) ** 2 / N**2)
plt.xlabel("freq (arb. units)")
plt.ylabel("PSD (arb. units)")
plt.show()
