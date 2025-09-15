from matplotlib import pyplot as plt
import numpy as np
import sys
from os import path
sys.path.insert(0, path.expanduser("~"))
from albatros_analysis.src.utils import pfb_utils as pu
import cupy as cp
import time

with np.load("./data/spectra_1830_1840_4096_24000_0.05_clock_1overf_5e-9.npz") as f:
        spec1=f['spectra1']
        spec2=f['spectra2']
        delays=f['delays']


nant = 1
npol = 2
osamp = 16
pfb_size = 65536
channels = np.arange(1830,1840)
lblock = 4096
cutsize = 10
read_size = pfb_size - 2*cutsize
timestream_size = read_size * lblock
nchan = len(channels)
new_channels = np.arange(osamp) + channels[:, None] * osamp
new_channels = new_channels.ravel()
new_nchan = len(new_channels)
#needs channels that are in the read data
ipfb = pu.StreamingIPFB(nant, npol, channels, nblock=pfb_size, lblock=4096, ntap=4, window='hamming', cut=cutsize)
fpfb = pu.StreamingPFB(nant, npol,timestream_size = timestream_size, lblock = lblock*osamp)
#needs channels you want to cross-correlate in re-PFB'd data
nchunks = spec1.shape[0]//pfb_size
new_acclen = 125
nrows_total = nchunks * pfb_size // (osamp * new_acclen)
xcorr = pu.StreamingCorrelator(nant, npol, new_acclen, new_channels, bufsize_frac = 50)
# vis = np.zeros((nant*npol, nant*npol, new_nchan, nrows_total), dtype="complex64", order="F")
new_spec1 = np.zeros((spec1.shape[0]//osamp, 30), dtype='complex64')
new_spec2 = new_spec1.copy()
print("new_nchan", new_nchan, "nrows total", nrows_total)
rowidx=0
filt_thresh = 0.08
idx1=0
idx2=0
start_event = cp.cuda.Event()
end_event = cp.cuda.Event()
for chunk_idx in range(nchunks):
    pol0 = cp.asarray(spec1[chunk_idx*read_size : (chunk_idx + 1)*read_size, :], dtype='complex64')
    pol1 = cp.asarray(spec2[chunk_idx*read_size : (chunk_idx + 1)*read_size, :], dtype='complex64')
    start_event.record()
    ts0=ipfb.ipfb(0,0,pol0,thresh=filt_thresh)
    ts1=ipfb.ipfb(0,1,pol1,thresh=filt_thresh)
    end_event.record()
    end_event.synchronize()
    # print("tot ipfb time", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
    start_event.record()
    pol0_new = fpfb.pfb(0,0, ts0)
    end_event.record()
    end_event.synchronize()
    # print("pfb time", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
    pol1_new = fpfb.pfb(0,1, ts1)
    n = pol0_new.shape[0]
    new_spec1[idx1:idx1+n,:] = cp.asnumpy(pol0_new[:, new_channels[53]:new_channels[83]])
    new_spec2[idx2:idx2+n,:] = cp.asnumpy(pol1_new[:, new_channels[53]:new_channels[83]])
    idx1+=n
    idx2+=n
    # print("PFB returned shape", pol0_new.shape, pol1_new.shape)
    # start_event.record()
    # xcorr.load(0, 0, pol0_new)
    # end_event.record()
    # end_event.synchronize()
    # # print("load time", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
    # xcorr.load(0, 1, pol1_new)
    # start_event.record()
    # rows = xcorr.xcorr()
    # end_event.record()
    # end_event.synchronize()
    # # print("xcorr time", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
    # n = len(rows)
    # if n > 0:
    #     print(f"chunk {chunk_idx+1}/{nchunks}, rcv {n}")
    #     for row in rows:
    #         t1=time.time()
    #         vis[:,:,:,rowidx] = cp.asnumpy(row) #dev to host
    #         t2=time.time()
    #         # print("dev to host time", t2-t1)
    #         rowidx+=1
print("final idxs", idx1, idx2)
# np.savez(f"/scratch/thomasb/mohan/vis_{new_acclen}_{osamp}_{filt_thresh}.npz", data=vis, channels=new_channels)
np.savez(f"/scratch/thomasb/mohan/raw_{osamp}_{filt_thresh}.npz", spec1 = new_spec1, spec2 = new_spec2, channels=new_channels[53:83])
# plt.imshow(np.abs(vis[0,0,:,:]), aspect='auto')
# plt.colorbar()
# plt.plot(np.log10(np.abs(vis[0,0,:,100])))
# plt.savefig("/scratch/thomasb/mohan/power.png",dpi=300)
# print("/scratch/thomasb/mohan/power.png")