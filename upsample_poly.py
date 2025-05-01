import numpy as np
from scipy.signal import resample_poly, upfirdn, firwin
from matplotlib import pyplot as plt
import numba as nb
import time
@nb.njit(cache=True)
def interp_poly(x,h,L,y):
    bw = (len(h)-1)//(2*L)
    center = (len(h)-1)//2
    # print("Len h", len(h))
    # print("BW", bw)
    # print("center idx and vals", center, h[center], h[center-1], h[center+1])
    M = 2*bw #len(x)
    for i in range(0,M):
        # print("i=",i)
        for j in range(0,L):
            y[j] += x[M-i-1]*h[L*i+j]
            # print(f"y[{j}] += x[{M-i-1}]*h[{L*i+j}]")
    return y

@nb.njit()
def interp_poly2(x,h,L,y): #this version is slower than previous one
    bw = (len(h)-1)//(2*L)
    center = (len(h)-1)//2
    # print("Len h", len(h))
    # print("BW", bw)
    # print("center idx and vals", center, h[center], h[center-1], h[center+1])
    M = 2*bw #len(x)
    for i in range(0,M):
            y[:] += x[M-i-1]*h[L*i:L*(i+1)]
    return y

@nb.njit(parallel=True,cache=True)
def big_interp(x,h,L,y,bw):
    N = len(x)-2*bw
    for i in nb.prange(N):
        interp_poly(x[i : 2*bw+i],h,L,y[i*L:(i+1)*L])

if __name__=='__main__':
    L=10
    bw=32
    np.random.seed(42)
    x = np.random.randn(2*bw)
    h = firwin(L*2*bw+1,1/L,window=('kaiser', 10))
    print("Filter len", len(h))
    plt.title("filter response")
    plt.loglog(2*np.arange(0,len(h)//2+1)/len(h),np.abs(np.fft.rfft(h))**2)
    plt.show()
    y = np.zeros(L,dtype='float64')
    interp_poly(x,h,L,y)
    # y = np.zeros(L,dtype='float64')
    # interp_poly2(x,h,L,y)

    # t1=time.time()
    # for ni in range(1000):
    #     interp_poly(x,h,L,y)
    # t2=time.time()
    # print((t2-t1)/1000)

    # t1=time.time()
    # for ni in range(1000):
    #     interp_poly2(x,h,L,y)
    # t2=time.time()
    # print((t2-t1)/1000)

    lenx = 20000
    leny = lenx*L
    xbig = np.random.randn(lenx + 2*bw) #pad front and back so we get exactly len y out
    ybig = np.zeros(leny,dtype='float64')
    big_interp(xbig,h,L,ybig,bw)
    print("compiled big")


    t1=time.time()
    for ni in range(10):
        yf = np.zeros(leny//2+1,dtype='complex128') #about 15 us. isnt much
        yf[:lenx//2+1] = np.fft.rfft(xbig[bw:-bw])
        ybigfft = np.fft.irfft(yf) #most time
    t2=time.time()
    print("FFT upsample", (t2-t1)/10)


    t1=time.time()
    for ni in range(10):
        big_interp(xbig,h,L,ybig,bw)
    t2=time.time()
    print("polyphase upsample", (t2-t1)/10)


    fs=250e3
    frac_bw = 0.4
    nn=lenx + 2*bw
    df = fs/nn
    BW=int(frac_bw*nn) #bandwidth
    print(bw, bw*df)

    ps_realization=np.zeros(nn//2+1,dtype='complex128')
    ps_realization[:BW+1] = np.random.randn(BW+1) + 1j * np.random.randn(BW+1)
    ts = np.fft.irfft(ps_realization)

    print("len ts", len(ts))
    plt.plot(np.real(ps_realization))
    plt.show()

    print(len(ts[bw:-bw]))
    plt.plot(100*np.abs(np.fft.rfft(ts[bw:-bw])))
    plt.show()
    ybig = np.zeros(leny,dtype='float64')
    big_interp(ts,h,L,ybig,bw)

    ybig2 = resample_poly(ts,up=10,down=1,window=('kaiser',10))
    plt.title("upsampled spectra")
    plt.loglog(np.abs(np.fft.rfft(ybig)))
    plt.loglog(np.abs(np.fft.rfft(ybig2)))
    plt.plot(np.abs(np.fft.rfft(ts[bw:-bw])))
    plt.show()


    # print(len(ts2))
    # print(len(ts3))

    # plt.plot(10*ts2)
    # plt.plot(ts3)
    # plt.show()

