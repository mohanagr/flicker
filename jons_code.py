import numpy as np
from matplotlib import pyplot as plt
import sys

def onepole(x,c):
    y=0*x
    y[0]=x[0]
    for i in range(1,len(x)):
        y[i]=x[i]+(1-c)*y[i-1]
    return y

def get_next_y(y_past,x,coeffs):
    n=len(y_past)
    print("x passed", x)
    print("y passed", y_past)
    y_big = np.hstack([y_past,np.zeros(len(x))])
    print(y_big.shape)
    print(coeffs.shape)
    for i in range(len(x)):
        y_big[i+n] = coeffs@y_big[i:n+i] + x[i]
    # plt.plot(y_big)
    # plt.show()
    return y_big[n:]

c=0.5
x=np.random.randn(1000)
y=onepole(x,c)

xx=0*x;
xx[0]=1
yy=onepole(xx,c)

delta = np.zeros(len(x))
delta[0]=1
coeffs=np.asarray([c])
fir = get_next_y(np.zeros(len(coeffs)),delta,coeffs)

print("response", np.sum(fir-yy))
sys.exit(0)
myx=np.hstack([x,0*x])
myc=np.hstack([yy,0*yy])
y2=np.fft.irfft(np.fft.rfft(myx)*np.fft.rfft(myc))[:len(x)]

plt.plot(y)
plt.plot(y2)
plt.show()