import numpy as np
import matplotlib.pyplot as plt
import generate_flicker as gf
import numba as nb

def get_adev(x,tau,stidx=0,endidx=None,fs=250e6):
    dt=1/fs
    delta=int(tau/dt)
#     print(delta)
    sl=slice(stidx,endidx,delta)
    samps=x[sl]
#     print(samps)
    adev=dt*np.sqrt(np.mean((samps[2:]-2*samps[1:-1]+samps[:-2])**2)/(2*tau**2))
    return adev

# N = 100000000
# print("total time =", N * 4e-9, "sec")
# # delays1=np.random.randn(N)
# delays2=2e-13*np.cumsum(np.random.randn(N))#+np.random.randn(N)
# plt.plot(delays2[:1000000])
# plt.show()
# taus=10**np.linspace(-3,1,101) #in seconds
# print(taus)
# # taus=np.linspace(0.1,2,1001)
# adevs=np.zeros(len(taus))
# for i,tau in enumerate(taus):
#     adevs[i]=get_adev(delays2,tau)
# m,c=np.polyfit(np.log10(taus),np.log10(adevs),1)
# print("slope is", m)
# plt.loglog(taus,adevs,label='adev of rw sim noise')
# plt.legend()
# plt.show()

@nb.njit(parallel=True,cache=True)
def add_white(y,scale1, scale2):
    n=len(y)
    for i in nb.prange(n):
        y[i] = scale1 * y[i] + scale2 * np.random.randn(1)[0]

_=add_white(np.zeros(10),1,1)

taus=10**np.linspace(-3,1,101) #in seconds
adevs=np.zeros(len(taus))


nsamp=100000000
nlevels=13

up = 10
f2 = 1 / 2
f1 = 0.993 * f2 / up
clock = gf.flicker(nlevels, nsamp, f1, f2)
clock.generate()
# plt.plot(clock.ybig)
# plt.plot(np.cumsum(clock.ybig))
# plt.show()
alpha=1
print(clock.ybig[0])
add_white(clock.ybig, alpha*1e-8,0)
print(clock.ybig[0])
noise=np.cumsum(clock.ybig)
plt.plot(noise[:1000])
plt.show()
for i,tau in enumerate(taus):
    adevs[i]=get_adev(noise,tau,fs=250e6)
print(adevs)
idx=~np.isnan(adevs)
m,c=np.polyfit(np.log10(taus[idx]),np.log10(adevs[idx]),1)
print("slope is", m)
plt.loglog(taus,adevs,label='adev of rw sim noise')
plt.legend()
plt.show()