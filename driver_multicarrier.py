import datetime
import time
import sys
import numba as nb
import numpy as np
from matplotlib import pyplot as plt

import generate_flicker as gf
import generate_narrowband as gn
import pfb

# ============================================================
# Helper functions for multi-carrier upconversion
# ============================================================


@nb.njit(parallel=True, cache=True)
def upconvert(y, I, Q, t, freq, block_num, block_len, accumulate=True):
    """Upconvert I/Q to carrier frequency. If accumulate=True, adds to y; otherwise overwrites."""
    nn = len(y)
    omega = 2 * np.pi * freq
    phi0 = ((omega * block_num) % (2 * np.pi) * (omega * block_len) % (2 * np.pi)) % (
        2 * np.pi
    )
    if not accumulate:
        for i in nb.prange(nn):
            phi = omega * t[i] + phi0
            y[i] = I[i] * np.cos(phi) - Q[i] * np.sin(phi)
    else:
        for i in nb.prange(nn):
            phi = omega * t[i] + phi0
            y[i] += I[i] * np.cos(phi) - Q[i] * np.sin(phi)


@nb.njit(parallel=True, cache=True)
def add_noise(y, sigma):
    """Add Gaussian noise to y in-place."""
    nn = len(y)
    noise = np.random.randn(nn)
    for i in nb.prange(nn):
        y[i] += sigma * noise[i]


# ============================================================

np.random.seed(42)

# ============================================================
# Multi-carrier configuration
# ============================================================
# Each entry specifies a carrier frequency and its narrowband
# bandwidth (df). All carriers share a common clock delay.
#   freq: carrier frequency (fractional, e.g. channel/4096)
#   df:   fractional bandwidth of the narrowband signal

carriers = [
    {"freq": 0.4478, "df": 0.48},  # chan 1835, 120 kHz wide
    {"freq": 0.4504, "df": 0.096},  # chan 1845, 24 kHz wide  (~610 kHz from 1835)
    {"freq": 0.4529, "df": 0.096},  # chan 1855, 24 kHz wide  (~1.22 MHz from 1835)
    {"freq": 0.4553, "df": 0.096},  # chan 1865, 24 kHz wide  (~1.83 MHz from 1835)
    {"freq": 0.4578, "df": 0.096},  # chan 1875, 24 kHz wide  (~2.44 MHz from 1835)
    {"freq": 0.4602, "df": 0.096},  # chan 1885, 24 kHz wide  (~2.44 MHz from 1835)
]
ncarriers = len(carriers)

# Shared narrowband parameters (length and upsample factor)
nb_length = 2048
nb_up = 1000

# ============================================================
# Create independent I/Q narrowband generators per carrier
# ============================================================
re_list = [gn.narrowband(df=c["df"], length=nb_length, up=nb_up) for c in carriers]
im_list = [gn.narrowband(df=c["df"], length=nb_length, up=nb_up) for c in carriers]

lblock = 4096
nlevels = 13
up = 10
f2 = 1 / 2
f1 = 0.993 * f2 / up
nsamp = 2048 * 1000

obj1 = pfb.StreamingPFB(tsize=nsamp)
obj2 = pfb.StreamingPFB(tsize=nsamp)
start_delay = 0

t_orig = np.arange(nsamp, dtype=np.float64)
t_new = np.empty(nsamp, dtype=np.float64)
bufsize = 10 * 4096  # how many samples
t_buf = np.hstack([np.arange(-bufsize, 0), t_orig])

# Per-carrier I/Q buffers (each needs its own buffer for spline continuity)
Ibufs = [np.zeros(nsamp + bufsize, dtype=np.float64) for _ in range(ncarriers)]
Qbufs = [np.zeros(nsamp + bufsize, dtype=np.float64) for _ in range(ncarriers)]

ybig = np.empty(nsamp, dtype="float64")  # undelayed combined signal
ybig2 = np.empty(nsamp, dtype="float64")  # delayed combined signal
csum = np.empty(nsamp, dtype="float64")
clock = gf.flicker(nlevels, nsamp, f1, f2)
nspec_per_run = nsamp // 4096
niter = 24000
start_chan = 1830
nchans = 50
spectra1 = np.empty((nspec_per_run * niter, nchans), dtype="complex128")
spectra2 = spectra1.copy()
delays = np.empty(nspec_per_run * niter, dtype=np.float64)
delay = np.zeros(nsamp)


@nb.njit(parallel=True, cache=True)
def apply_drift_delay(delay, t_new, t_orig, slope, offset):
    nn = len(t_orig)
    for i in nb.prange(nn):
        delay[i] = slope * t_orig[i] + offset
        t_new[i] = t_orig[i] + delay[i]


scale = 1e-10
time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
sigma = 0.05

# this is if you want to do a constant drift instead of noise
offset = 5 * 4096 + 200
slope = 5 / 20e3 / 4096  # sample drift per 20k spectra
#############################################################
write_data = True

try:
    for ii in range(niter):
        block_num = ii
        block_len = nsamp

        t1 = time.time()

        # Generate all narrowband I/Q signals
        for cc in range(ncarriers):
            re_list[cc].generate().osamp()
            im_list[cc].generate().osamp()

        # Common clock delay
        clock.generate()
        delay = gf.cumsum(csum, clock.ybig, start_delay, scale)
        start_delay = delay[-1]
        gn.get_delay(t_new, t_orig, delay, offset)

        # --------------------------------------------------------
        # Undelayed path: accumulate all carriers into ybig
        # --------------------------------------------------------
        ybig[:] = 0.0
        for cc in range(ncarriers):
            carrier = carriers[cc]["freq"]
            Ibufs[cc][bufsize:] = re_list[cc].ybig
            Qbufs[cc][bufsize:] = im_list[cc].ybig
            # first carrier overwrites, rest accumulate
            upconvert(
                ybig,
                Ibufs[cc][bufsize:],
                Qbufs[cc][bufsize:],
                t_orig,
                carrier,
                block_num,
                block_len,
                accumulate=(cc > 0),
            )
        # Add noise once to combined undelayed signal
        add_noise(ybig, sigma)

        # --------------------------------------------------------
        # Delayed path: spline-interpolate each carrier's I/Q at
        # delayed times, upconvert, accumulate into ybig2
        # --------------------------------------------------------
        ybig2[:] = 0.0
        for cc in range(ncarriers):
            carrier = carriers[cc]["freq"]
            I2 = gn.cubic_spline(t_new, t_buf, Ibufs[cc])
            Q2 = gn.cubic_spline(t_new, t_buf, Qbufs[cc])
            upconvert(
                ybig2,
                I2,
                Q2,
                t_new,
                carrier,
                block_num,
                block_len,
                accumulate=(cc > 0),
            )
            # Rotate this carrier's buffers
            Ibufs[cc][:bufsize] = Ibufs[cc][-bufsize:]
            Qbufs[cc][:bufsize] = Qbufs[cc][-bufsize:]
        # Add noise once to combined delayed signal
        add_noise(ybig2, sigma)

        # --------------------------------------------------------
        # PFB and store spectra
        # --------------------------------------------------------
        obj1.pfb(ybig)
        obj2.pfb(ybig2)
        spectra1[ii * nspec_per_run : (ii + 1) * nspec_per_run, :] = obj1.spectra[
            :, start_chan : start_chan + nchans
        ]
        spectra2[ii * nspec_per_run : (ii + 1) * nspec_per_run, :] = obj2.spectra[
            :, start_chan : start_chan + nchans
        ]
        delays[ii * nspec_per_run : (ii + 1) * nspec_per_run] = np.mean(
            np.reshape(delay, (-1, 4096)), axis=1
        )  # one number per 4096 delay samples.
        t2 = time.time()
        print(ii, t2 - t1)

except Exception as e:
    pass
finally:
    if write_data:
        np.savez(
            f"./data/spectra_multicarrier_{start_chan}_{start_chan + nchans}_{lblock}_{ii + 1}_{sigma}_{scale:.2e}_{time_tag}.npz",
            spectra1=spectra1,
            spectra2=spectra2,
            delays=delays,
        )
