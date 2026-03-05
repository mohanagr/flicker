import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize

def f(freq,tau):
    return np.hstack([np.cos(2*tau*freq*np.pi/4096), np.sin(2*tau*freq*np.pi/4096)])

def res(tau, data, freq):
    res = np.exp(1j*2*tau*freq*np.pi/4096)-np.exp(1j*data)
    return np.hstack([np.real(res), np.imag(res)])

def jac2(tau, data, freq):
    y = np.exp(2j*np.pi*freq*tau/4096)
    dy = y * 2j * np.pi*freq/4096
    n_eval = len(freq)
    J = np.zeros((2*len(freq), 1), dtype='float64')
    J[:n_eval,0] = np.real(dy)
    J[n_eval:,0] = np.imag(dy)
    return J  

def jac(freq, tau):
    y = np.exp(2j*np.pi*freq*tau/4096)
    dy = y * 2j * np.pi*freq/4096
    n_eval = len(freq)
    J = np.zeros((2*len(freq), 1), dtype='float64')
    J[:n_eval,0] = np.real(dy)
    J[n_eval:,0] = np.imag(dy)
    return J

def wrap(ang):
    return (ang + np.pi)%(2*np.pi) - np.pi

# carrier=1834
carrier=1830
osamp=64
full_freq = np.arange(0,2048*64)/64

freq = full_freq[carrier*64:carrier*64+85]
# ALONG TIME AXIS, IF delay jumps by more than a cycle, it's not good. 
print("for carrier of", carrier, " wrap at", 4096/carrier)
print("for bw of", freq[-1]-freq[0], "wrap at", 4096/(freq[-1]-freq[0]), "samples")

tau =  np.arange(0,4)*4096/carrier/5+ 11200 #samp

phase = 2*np.pi*freq[None,:]*tau[:,None]/4096
vis = np.exp(1j*phase)
# phase_fill = 2*np.pi*full_freq[None,:]*tau[:,None]/4096

print("phase shape", phase.shape)
wp_phase = np.angle(np.exp(1j*phase))
uwp_phase = np.unwrap(wp_phase,axis=1)
uwp_phase = np.unwrap(uwp_phase, axis=0)

# plt.plot(phase[0])
plt.plot(uwp_phase[0])
plt.plot(uwp_phase[1])
plt.show()

lsq_tau1 = 4096*freq @ uwp_phase[0]/(freq@freq)/(2*np.pi)
lsq_tau2 = 4096*freq @ uwp_phase[1]/(freq@freq)/(2*np.pi)

print("lsq_tau1", lsq_tau1, "lsq_tau2", lsq_tau2, "diff", lsq_tau2-lsq_tau1)

# plt.plot(uwp_phase[0])
# plt.plot(2*np.pi*freq*lsq_tau/4096)
# plt.show()
data = np.angle(vis[0])
ydata = np.hstack([np.real(vis[0]), np.imag(vis[0])])
output=optimize.curve_fit(f,freq,ydata,p0=[1,],jac=jac,full_output=True)
output2=optimize.least_squares(res,[1,],args=(data, freq), jac=jac2)
print(output2['x'])
tau_scipy1 = output[0]
data = np.angle(vis[1])
ydata = np.hstack([np.real(vis[1]), np.imag(vis[1])])
output=optimize.curve_fit(f,freq,ydata,p0=[1,],jac=jac,full_output=True)
output2=optimize.least_squares(res,[1,],args=(data, freq), jac=jac2)
print(output2['x'])
print("fjac, infodict", output[2]['fjac'].shape)
tau_scipy2 = output[0]
print("tau_scipy1", tau_scipy1, "tau_scipy2", tau_scipy2, "diff", tau_scipy2-tau_scipy1)

# plt.plot(np.real(vis)[0])
# plt.plot(np.real(np.exp(2j*np.pi*freq*tau_scipy/4096)))
# plt.show()