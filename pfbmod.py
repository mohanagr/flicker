import numpy as np


class StreamingIPFB():
    def __init__(self, nblock=4096*100, lblock=4096, ntap=4, window='hamming', cut=10):
        N = self.lblock * ntap
        w = np.arange(0, N) - N // 2
        self.win = np.__dict__[window](N) * np.sinc(w / self.lblock)
        self.cut = cut
        self.nchans=lblock//2+1
        self.specbuf = np.empty((2*cut + nblock, nchans), dtype='complex128')
        mat=np.zeros((2*cut + nblock, lblock),dtype=specbuf.dtype)
        mat[:ntap,:]=np.reshape(self.win,[ntap,len(self.win)//ntap])
        mat=mat.T.copy()
        matft=np.fft.rfft(mat,axis=1)
    
    def ipfb(self, spectra,thresh=0.):
        #input spectra, fill timestream
        self.specbuf[2*cut:, :] = spectra
        
        dd=np.fft.irfft(self.specbuf,axis=1)
        self.specbuf[:2*cut, :] = spectra[-2*cut:, :]
        dd2=dd.T.copy()
        ddft=np.fft.rfft(dd,axis=1)
        if thresh>0.:
            filt=np.abs(matft)**2/(thresh**2+np.abs(matft)**2)*(1+thresh**2)
            ddft=ddft*filt
        out = np.fft.irfft(ddft/np.conj(matft),axis=1)
        out = out.T.copy()[cut:-cut].ravel()
        return out


