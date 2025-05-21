import os
os.environ['NUMBA_OPT']='3'
os.environ['NUMBA_LOOP_VECTORIZE']='1'
os.environ['NUMBA_ENABLE_AVX']='1'
import numpy as np
import numba as nb
from rocket_fft import c2r

@nb.njit(cache=True, parallel=True)
def forward_pfb(timestream, win, spectra=None, out=None,nchan=2049, ntap=4):
    lblock = 2*(nchan-1)
    nblock = timestream.size // lblock - (ntap - 1)
    norm=1/lblock
    if out is not None:
        assert out.shape == (nblock, lblock)
        out[:] = 0. #can be multithreaded
    else:
        out = np.empty((nblock, lblock), dtype=timestream.dtype)
    if spectra is not None:
        assert spectra.shape == (nblock, nchan)
    else:
        spectra = np.empty((nblock, nchan), dtype="complex128")
    for i in nb.prange(nblock):
        for j in range(ntap):
            for k in range(lblock):
                out[i,k] += win[j*lblock + k]*timestream[i*lblock + j * lblock + k]
    c2r(out,spectra,np.asarray([1,]),True, norm, 0)
    return spectra

class PFB():
    def __init__(self, tsize=2000000,nchan=2049, ntap=4, window='hamming'):
        self.nchan = nchan
        self.ntap = ntap
        self.lblock = 2*(nchan-1)
        N = self.lblock * ntap
        w = np.arange(0, N) - N // 2
        rem = (ntap-1)*self.lblock
        if tsize % self.lblock > 0:
            rem += tsize % self.lblock
        self.nblock = tsize // self.lblock - (ntap - 1)
        self.win = np.__dict__[window](N) * np.sinc(w / self.lblock)
        self.spectra = np.empty((self.nblock, nchan), dtype="complex128")
        self.out = np.empty((self.nblock, self.lblock), dtype="float64")
        self.timestream = np.empty(tsize + rem, dtype="float64")
        self.timestream.fill(0)
    def pfb(self, timestream):
        #may be JIT this thing too
        self.timestream[(self.ntap-1)*self.lblock:] = timestream #could be multithreaded
        forward_pfb(self.timestream, self.win, self.spectra, self.out, self.nchan, self.ntap)
        self.timestream[:(self.ntap-1)*self.lblock] = timestream[-(self.ntap-1)*self.lblock:]
        