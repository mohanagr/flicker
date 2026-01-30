import numpy as np
from matplotlib import pyplot as plt

def wrap(ang):
    return (ang + np.pi)%(2*np.pi) - np.pi

colors = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow (use carefully on white)
    "#000000",  # black
    "#999999",  # grey
    "#332288",  # deep indigo (extra, still CB-friendly-ish)
]
linestyles = ["-","--","-.",":"]
pair_styles = [(colors[i], linestyles[i % len(linestyles)]) for i in range(10)]

# carrier=1834
carrier=2
osamp=64
full_freq = np.arange(0,2048*64)/64

freq = full_freq[carrier*64:carrier*64+85]
# ALONG TIME AXIS, IF delay jumps by more than a cycle, it's not good. 
print("for carrier of", carrier, " wrap at", 4096/carrier)
print("for bw of", freq[-1]-freq[0], "wrap at", 4096/(freq[-1]-freq[0]), "samples")

# tau =  np.arange(0,4)*500+ 11000 #samp
tau =  np.arange(0,4)*4096/carrier/5+ 9000 #samp

phase = 2*np.pi*freq[None,:]*tau[:,None]/4096
phase_fill = 2*np.pi*full_freq[None,:]*tau[:,None]/4096

print("phase shape", phase.shape)
wp_phase = np.angle(np.exp(1j*phase))
uwp_phase = np.unwrap(wp_phase,axis=1)
uwp_phase = np.unwrap(uwp_phase, axis=0)

# plt.plot(np.unwrap(wp_phase[-1,:]))
# plt.plot(uwp_phase[-1,:])
# plt.show()
center_phase = uwp_phase[0,0]

dphi = uwp_phase[0,-1]-uwp_phase[0,0]
print("dphi", dphi, "est", 2*np.pi*(freq[-1]-freq[0])*tau[0]/4096)

print("phase at carrier after unwrap", uwp_phase[0,0], "tau", tau[0])
tau_derot = 4096*center_phase/(2*np.pi*freq[0])
tau_derot2 = 4096*uwp_phase[0,-1]/(2*np.pi*freq[-1])
print("derot tau start", tau_derot)
print("derot tau end", tau_derot2)

x=(freq[-1]-freq[0])/(freq[0])
print("frac bw", x)
print("center_phase", center_phase, "dphi", dphi)
print("estimate of slope error", 4096*((1-x)*dphi - x*center_phase)/(2*np.pi*freq[0]), "even rougher", x * 5, "actual error", tau_derot2-tau_derot)

lsq_slope = np.sum(freq*uwp_phase[0,:])/np.sum(freq**2)/(2*np.pi)
print(lsq_slope)
fig=plt.gcf()
fig.set_size_inches(10,8)
for k in range(len(tau)):
    c,ls=pair_styles[k]
    plt.plot(full_freq[:carrier*osamp], phase_fill[k,:carrier*osamp], color=c, linestyle=ls,alpha=0.4)
    plt.plot(freq, phase[k,:], color=c, linestyle=ls)
    plt.plot(freq, uwp_phase[k,:], color=c, linestyle=ls, label=f'time={k+1}')
# plt.plot(full_freq[:carrier*osamp+85], 2*np.pi*tau_derot*full_freq[:carrier*osamp+85], c='black')
x0,y0=freq[0],uwp_phase[0,0]
plt.plot([x0], [y0],
        marker="o", markersize=10,
        markerfacecolor="yellow", markeredgecolor="black",
        markeredgewidth=1, linestyle="None", zorder=5)
plt.axvspan(freq[0], freq[-1], color='gray', alpha=0.3, label=r'signal width $\Delta \nu=$'+f'{freq[-1]-freq[0]:4.2f} chans')
plt.xlim(0,4)
plt.text(1,30,r"$\phi_{actual} = 2 \pi \nu \tau(t)$", fontsize=14)
plt.text(2.5,5,r"$\phi_{meas} = [2 \pi \nu \tau(t)] \% 2 \pi$,"+"\n" +r"unwrapped along $\nu,t$", fontsize=12)
plt.annotate(r"$[2 \pi \nu_0 \tau_0)] \% 2 \pi$",
            xy=(x0, y0),
            xytext=(x0 - 0.75, y0+10),  # in data units
            arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=10)
plt.grid(True)
plt.title(f"Visibility phase illustration, carrier at chan {carrier}",fontsize=12)
plt.xlabel("Freq. channel",fontsize=12)
plt.ylabel("Phase (rad)",fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

#now let's just shift it along freq a lot.



# for k in range(len(tau)):
#     c,ls=pair_styles[k]
#     plt.plot(full_freq[:carrier*osamp], phase_fill[k,:carrier*osamp], color=c, linestyle=ls,alpha=0.4)
#     plt.plot(freq, phase[k,:], color=c, linestyle=ls)
#     plt.plot(freq, uwp_phase[k,:], color=c, linestyle=ls, label=f'time={k+1}')