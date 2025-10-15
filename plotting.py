from matplotlib import pyplot as plt
import numpy as np
plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.titlesize": 22
    })
with np.load("/home/mohan/Downloads/vis_1024_16_0.08.npz") as f:
        vis = f['data']
        chans = f['channels']
print(vis.shape, vis.dtype)

# plt.plot(chans[:140]/16, np.log10(np.abs(vis[0,0,:140,250])))
# for i in range(1830, 1840):
#     plt.axvline(i-0.5,c='r')
df=0.061/16
plt.title("Visibility amplitude. Up-res by 16x.")
myext=[chans[50]*df,chans[85]*df,vis.shape[-1]*1024*16e-6*16,0]
fig=plt.gcf()
fig.set_size_inches(8,4)
ax=plt.gca()
im=plt.imshow(np.log10(np.abs(vis[0,0,50:85,:-1].T)), aspect='auto',extent=myext)
cbar = fig.colorbar(im, ax=ax, orientation="vertical")
cbar.ax.tick_params(labelsize=14)
cbar.set_label("log(Power)", fontsize=16)
plt.xlabel("Freq (MHz)")
plt.ylabel("Time (s)")
plt.tight_layout()
plt.savefig("./up_res_visiblity.png",dpi=300)
# print("/scratch/thomasb/mohan/power.png")