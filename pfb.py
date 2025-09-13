import os
os.environ['NUMBA_OPT']='3'
os.environ['NUMBA_LOOP_VECTORIZE']='1'
os.environ['NUMBA_ENABLE_AVX']='1'
import numpy as np
import numba as nb
from rocket_fft import r2c

def sinc_hamming(ntap,lblock):
    N=ntap*lblock
    w=np.arange(0,N)-N/2
    return np.hamming(ntap*lblock)*np.sinc(w/lblock)

def sinc_hanning(ntap,lblock):
    N=ntap*lblock
    w=np.arange(0,N)-N/2
    return np.hanning(ntap*lblock)*np.sinc(w/lblock)

def forward_pfb(timestream, nchan=2048, ntap=4, window=sinc_hanning):

    # number of samples in a sub block
    lblock = 2*(nchan)
    # number of blocks
    nblock = timestream.size / lblock - (ntap - 1)
    if nblock==int(nblock): nblock=int(nblock)
    else: raise Exception("nblock is {}, should be integer".format(nblock))

    # initialize array for spectrum 
    spec = np.zeros((nblock,nchan+1), dtype=np.complex128)

    # window function
    w = window(ntap, lblock)

    def s(ts_sec):
        return np.sum(ts_sec.reshape(ntap,lblock),axis=0) # this is equivalent to sampling an ntap*lblock long fft - M


    # iterate over blocks and perform PFB
    for bi in range(nblock):
        # cut out the correct timestream section
        ts_sec = timestream[bi*lblock:(bi+ntap)*lblock].copy()

        spec[bi] = np.fft.rfft(s(ts_sec * w)) 

    return spec

def inverse_pfb(dat,ntap,window=sinc_hamming,thresh=0.0):
    dd=np.fft.irfft(dat,axis=1)
    win=window(ntap,dd.shape[1])
    win=np.reshape(win,[ntap,len(win)//ntap])
    mat=np.zeros(dd.shape,dtype=dd.dtype)
    mat[:ntap,:]=win
    matft=np.fft.rfft(mat,axis=0)
    ddft=np.fft.rfft(dd,axis=0)
    if thresh>0:
        filt=np.abs(matft)**2/(thresh**2+np.abs(matft)**2)*(1+thresh**2)
        ddft=ddft*filt
    return np.fft.irfft(ddft/np.conj(matft),axis=0)

# @nb.njit(parallel=True)
# def forward_pfb(timestream, win=None, spectra=None, scratch=None,nchan=2049, ntap=4):
#     lblock = 2*(nchan-1)
#     nblock = timestream.size // lblock - (ntap - 1)
#     norm=1.
#     if scratch is not None:
#         assert scratch.shape == (nblock, lblock)
#         scratch[:] = 0. #can be multithreaded
#     else:
#         scratch = np.empty((nblock, lblock), dtype=timestream.dtype)
#     if spectra is not None:
#         assert spectra.shape == (nblock, nchan)
#     else:
#         spectra = np.empty((nblock, nchan), dtype="complex128")
#     if win is None:
#         N = lblock * ntap
#         w = np.arange(0, N) - N // 2
#         win = np.hamming(N)*np.sinc(w / lblock)
#     for i in nb.prange(nblock):
#         for j in range(ntap):
#             for k in range(lblock):
#                 scratch[i,k] += win[j*lblock + k]*timestream[i*lblock + j * lblock + k]
#     r2c(scratch,spectra,np.asarray([1,]),True, norm, 0)
#     return spectra

@nb.njit(parallel=True,cache=True)
def _fpfb(timestream, win, spectra, scratch, nchan, ntap):
    lblock = 2*(nchan-1)
    nblock = timestream.size // lblock - (ntap - 1)
    for i in nb.prange(nblock):
        for j in range(lblock):
            scratch[i,j]=0.
    for i in nb.prange(nblock):
        for j in range(ntap):
            for k in range(lblock):
                scratch[i,k] += win[j*lblock + k]*timestream[i*lblock + j * lblock + k]
    r2c(scratch,spectra,np.asarray([1,]),True, 1., 0)
    return spectra

def _fill(y,x):
    nn=len(y)
    for i in nb.prange(nn):
        y[i]=x[i]

class StreamingPFB():
    def __init__(self, tsize=4096*500, nchan=2049, ntap=4, window='hamming'):
        #only accepts multiple of lblock for now.
        #could implement a buffered solution later for arbitrary sizes
        self.nchan = nchan
        self.ntap = ntap
        self.lblock = 2*(nchan-1)
        assert tsize%self.lblock==0
        N = self.lblock * ntap
        w = np.arange(0, N) - N // 2
        rem = (ntap-1)*self.lblock
        self.nblock = tsize // self.lblock #buffer the last 3 blocks from previous chunk
        self.win = np.__dict__[window](N) * np.sinc(w / self.lblock)
        self.spectra = np.empty((self.nblock, nchan), dtype="complex128")
        self.scratch = np.empty((self.nblock, self.lblock), dtype="float64")
        self.timestream = np.zeros(tsize + rem, dtype="float64")
    def pfb(self, timestream):
        #input timestream, fill spectra
        #timestream = external timestream that is ONLY tsize long
        #may be JIT this thing too
        _fill(self.timestream[(self.ntap-1)*self.lblock:],timestream)
        # self.timestream[(self.ntap-1)*self.lblock:] = timestream #could be multithreaded
        _fpfb(self.timestream, self.win, self.spectra, self.scratch, self.nchan, self.ntap)
        _fill(self.timestream[:(self.ntap-1)*self.lblock],timestream[-(self.ntap-1)*self.lblock:])
        # self.timestream[:(self.ntap-1)*self.lblock] = timestream[-(self.ntap-1)*self.lblock:]

class StreamingIPFB():
    def __init__(self, nblock=100, lblock=4096, ntap=4, window='hamming', cut=10):
        self.lblock = lblock
        self.nblock = nblock
        N = self.lblock * ntap
        w = np.arange(0, N) - N // 2
        self.win = np.__dict__[window](N) * np.sinc(w / self.lblock)
        self.cut = cut
        self.nchans=lblock//2+1
        self.specbuf = np.empty((2*cut + nblock, self.nchans), dtype='complex128')
        mat=np.zeros((2*cut + nblock, lblock),dtype="float64")
        mat[:ntap,:]=np.reshape(self.win,[ntap,len(self.win)//ntap])
        mat=mat.T.copy()
        print("mat shape", mat.shape)
        self.matft = np.fft.rfft(mat,axis=1)
        print("matft shape", self.matft.shape)
    
    def ipfb(self, spectra, chans, thresh=0.):
        #input spectra, fill timestream
        self.specbuf[2*self.cut:, chans] = spectra
        print(self.specbuf[0,1829:1842])
        dd=np.fft.irfft(self.specbuf,axis=1)
        print("dd shape", dd.shape)
        self.specbuf[:2*self.cut, chans] = spectra[-2*self.cut:, :]
        dd2=dd.T.copy()
        ddft=np.fft.rfft(dd2,axis=1)
        if thresh>0.:
            filt=np.abs(self.matft)**2/(thresh**2+np.abs(self.matft)**2)*(1+thresh**2)
            ddft=ddft*filt
        out = np.fft.irfft(ddft/np.conj(self.matft),axis=1)
        out = out.T.copy()[self.cut:-self.cut].ravel()
        return out
        
if __name__=='__main__':
    # nchan=2049
    # ntap=4
    # # timestream = np.cos(2*np.pi*1830.1*np.arange(0,2048*500)/4096)
    # timestream = np.random.randn(2048*500)
    # lblock = 2*(nchan-1)
    # nblock = timestream.size // lblock - (ntap - 1)
    # scratch = np.empty((nblock, lblock), dtype=timestream.dtype)
    # spectra = np.empty((nblock, nchan), dtype="complex128")
    # N = lblock * ntap
    # w = np.arange(0, N) - N // 2
    # win = np.hamming(N)*np.sinc(w / lblock)
    # f1 = forward_pfb(timestream,2048,4,sinc_hamming)
    # f2 = _fpfb(timestream,win,spectra,scratch,2049,4)
    # print(np.abs(f1-f2).sum())
    # print(f1.shape)
    # from matplotlib import pyplot as plt
    # plt.imshow(np.abs(f),aspect='auto',interpolation='none')
    # plt.colorbar()
    # plt.show()

    spectra = np.random.randn(100*2049) + 1j*np.random.randn(100*2049)
    spectra = spectra.reshape(100,-1)
    print("spectra shape", spectra.shape)
    ipfb =  StreamingIPFB(nblock=100)
    print("final two rows of spec", spectra[-2*ipfb.cut:])
    ts = ipfb.ipfb(spectra,thresh=0.1)
    print("first two rows of specbuf", ipfb.specbuf[:2*ipfb.cut])
    print(ts.shape)