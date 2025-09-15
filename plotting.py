from matplotlib import pyplot as plt
import numpy as np

with np.load("/scratch/thomasb/mohan/vis_1024_16_0.08.npz") as f:
        vis = f['data']
        chans = f['channels']
print(vis.shape, vis.dtype)

# plt.plot(chans[:140]/16, np.log10(np.abs(vis[0,0,:140,250])))
# for i in range(1830, 1840):
#     plt.axvline(i-0.5,c='r')
plt.imshow(np.log10(np.abs(vis[0,0,50:85,:-1])), aspect='auto')
plt.colorbar()
plt.savefig("/scratch/thomasb/mohan/power.png",dpi=300)
print("/scratch/thomasb/mohan/power.png")