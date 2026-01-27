from matplotlib import pyplot as plt
import numpy as np
import sys
from scipy.stats import median_abs_deviation as mad
import cg
import timing

def phase2timenoise(sigma_phase, nchan, chanwidth):
    return np.sqrt(12) * sigma_phase / chanwidth / np.sqrt(nchan * (nchan**2 - 1))


def average_rows(x, nblock=100):
    nr = x.shape[0] // nblock
    print(x.shape[0], nr)
    nc = x.shape[1]
    y = np.zeros((nr, nc), dtype=x.dtype)
    for i in range(nr):
        y[i, :] = np.mean(x[i * nblock : (i + 1) * nblock], axis=0)
    return y


def get_adev(x, tau, stidx=0, endidx=None):
    dt = 4096 / 250e6
    delta = int(tau / dt)
    #     print(delta)
    sl = slice(stidx, endidx, delta)
    samps = x[sl]
    #     print(samps)
    adev = dt * np.sqrt(
        np.mean((samps[2:] - 2 * samps[1:-1] + samps[:-2]) ** 2) / (2 * tau**2)
    )
    return adev


def lmsolver(
    xc, alpha, chan, scale=1, lamda=16, xtol=1e-6, ftol=1e-6, niter=10, debug=False
):
    c = 2 * np.pi * chan
    N = len(xc)
    n = np.arange(N)
    lamda = lamda
    for ii in range(niter):
        if debug:
            print(
                f"------------------------------ LM iter {ii} -------------------------------"
            )
        xc_phased = 1e-5 * xc * np.exp(1j * c * n * alpha)  # n is specnum
        f = -np.abs(np.mean(xc_phased)) ** 2
        S0conj = np.conj(np.mean(xc_phased))
        S1 = np.mean(xc_phased * n * c)
        S2 = np.mean(xc_phased * n**2 * c**2)
        # print(np.abs(S0conj), np.abs(S1), np.abs(S2))
        df = np.imag(S0conj * S1)
        ddf = np.real(S0conj * S2) - np.abs(S1) ** 2
        step = -df / (ddf + lamda * np.abs(ddf))
        # step = - 1e-3 * df
        alpha2 = alpha + step
        #
        # f22 = -np.mean(np.abs(xc_phased)**2)
        xc_phased = 1e-5 * xc * np.exp(1j * c * n * alpha2)
        f2 = -np.abs(np.mean(xc_phased)) ** 2
        # print(f,f2,alpha,alpha2)
        if debug:
            print(
                f"lamda: {lamda:5.3e} alpha: {alpha:5.3e} f: {f:5.3e}, df: {df:5.3e}, ddf: {ddf:5.3e}, step: {step:5.3e}, alpha2: {alpha2:5.3e} f2: {f2:5.3e}"
            )
        if f2 < f:  # proceeding towards minimization
            # accept

            if debug:
                print("accepting...")
            rel_alpha = (alpha2 - alpha) / alpha
            rel_f = (f2 - f) / f
            if debug:
                print(
                    f"rel alpha: {np.abs(rel_alpha):5.3e} rel f: {np.abs(rel_f):5.3e}"
                )
            alpha = alpha2
            lamda /= 2
            if np.abs(rel_alpha) < xtol or np.abs(rel_f) < ftol:
                if debug:
                    print("converged.")
                return alpha
        else:
            lamda *= 2
    raise Exception(f"Failed to converge in {niter} iterations.")
    return alpha


if __name__ == "__main__":

    # with np.load("./data/spectra_1830_1840_4096_24000_0.05_clock_1overf_5e-9.npz") as f:
    #     spec1 = f["spectra1"]
    #     spec2 = f["spectra2"]
    #     delays = f["delays"]
    # with np.load("/home/mohan/Downloads/raw_16_0.08.npz") as f:
    #     new_spec1 = f["spec1"]
    #     new_spec2 = f["spec2"]
    #     new_channels = f["channels"]

    with np.load("./data/spectra_1830_1840_4096_24000_0.05_clock_1overf_1e-10.npz") as f:
        spec1 = f["spectra1"]
        spec2 = f["spectra2"]
        delays = f["delays"]
    
    with np.load("/home/mohan/Projects/flicker/data/raw_1e-10_16_0.08.npz") as f:
        new_spec1 = f["spec1"]
        new_spec2 = f["spec2"]
        new_channels = f["channels"]

    osamp = 16
    # Global font size settings
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "figure.titlesize": 22,
        }
    )
    print("len delays", len(delays))
    plt.plot(-delays)
    plt.show()
    # print("delay at midpoint is", delays[4096*2])
    # print("len new spec", new_spec1.shape[0])
    # sys.exit(0)
    avglen = 8192
    # blocksize = avglen
    # st = 0
    # nblocks = spec1.shape[0] // blocksize
    # # nblocks=20
    # print("Num blocks are", nblocks)
    # fit_delta = np.zeros(nblocks, dtype="float64")
    # actual_delta = np.zeros(nblocks, dtype="float64")
    # r_val = np.zeros(nblocks, dtype="float64")
    # for i in range(nblocks):
    #     # fft and get a coarse guess
    #     ix = st + i * blocksize
    #     # print("taking data from ", ix, ix+blocksize)
    #     y1 = spec1[ix : ix + blocksize, 4]
    #     y2 = spec2[ix : ix + blocksize, 4]
    #     xc_small = y1 * np.conj(y2)
    #     xc_fft = np.fft.fftshift(np.abs(np.fft.fft(xc_small)))
    #     mm = np.argmax(xc_fft)
    #     M = len(xc_fft)
    #     # print(M)
    #     expected_delta = -(mm - M / 2) / M / 1834
    #     if np.abs(expected_delta) < 1e-15:
    #         expected_delta = 1e-15
    #     # print("Starting point", expected_delta)
    #     # print(f"Block {i} Expected delta={expected_delta}")
    #     alpha = lmsolver(
    #         xc_small,
    #         expected_delta,
    #         1834.1888,
    #         lamda=10,
    #         ftol=1e-8,
    #         xtol=1e-8,
    #         niter=100,
    #     )
    #     fit_delta[i] = alpha
    #     # plt.title(f"Block {i}")
    #     # plt.plot(delays[ix:ix+blocksize])
    #     # plt.show()
    #     xx = np.arange(ix, ix + blocksize)
    #     yy = delays[ix : ix + blocksize]
    #     yymean = np.mean(yy)
    #     m, c = np.polyfit(xx, yy, 1)
    #     yypred = np.polyval([m, c], xx)
    #     SSreg = np.sum((yypred - yymean) ** 2)
    #     SStot = np.sum((yy - yymean) ** 2)
    #     # print(f"SSreg, SStot for {i}", SSreg,SStot)
    #     r_val[i] = np.sqrt(SSreg / SStot)
    #     # m,c=np.polyfit(np.arange(0,en-st),delays[st:en]-delays[st],1)
    #     # print(f"block {i} slope", m/4096, "pred slope", alpha)
    #     actual_delta[i] = m / 4096

    # xc=new_spec1[:,:]*np.conj(new_spec2[:,:])
    # xc_true=xc[:,:]*np.exp(2j*np.pi*new_channels*delays[:osamp*new_spec1.shape[0]:osamp,None]/4096/osamp)
    # xc_true_avg = average_rows(xc_true,nblock=1000)
    # xc_avg = average_rows(xc,nblock=1000)

    # # plt.plot(auto_avg1[:,0],label='autos1')
    # plt.plot(np.abs(xc_true_avg[:,4]),label='cross corrected true')
    # plt.plot(np.real(xc_true_avg[:,4]),label='cross corrected true real')
    # plt.plot(np.abs(xc_avg[:,4]),label='cross uncorrected true')
    # plt.legend()
    # plt.show()
    # sys.exit()

    st = 0
    blocksize = avglen // osamp

    print("new channels are", new_channels / osamp)
    nblocks = new_spec1.shape[0] // blocksize
    # nblocks = 300
    print("new blocksize=", blocksize, "nblocks=",nblocks)
    n = np.arange(0, blocksize)
    xc_avg_corrected = np.zeros((nblocks, len(new_channels)), dtype="complex128")
    xc_avg_uncorrected = np.zeros((nblocks, len(new_channels)), dtype="complex128")
    slopes = np.zeros(nblocks, dtype="float64")
    ph_noises = np.zeros(nblocks, dtype="float64")
    phases = np.zeros((nblocks, len(new_channels)), dtype="float64")
    
    # st = 4 * blocksize
    true_drift = -delays[st:: blocksize * osamp][: len(slopes)]
    
    plt.plot(true_drift)
    plt.title("true drift")
    plt.show()
    for i in range(nblocks):
        # fft and get a coarse guess
        ix = st + i * blocksize
        y1 = new_spec1[ix : ix + blocksize, :]
        y2 = new_spec2[ix : ix + blocksize, :]
        xc_small = y1 * np.conj(y2)
        # xc_corrected1 = xc_small * np.exp(2j*np.pi*1834*n*fit_delta[i])
        # xc_corrected1 = xc_small * np.exp(
        #     2j * np.pi * new_channels * n[:, None] * fit_delta[i]
        # )
    
        # xc_avg_corrected[i, :] = np.mean(xc_corrected1, axis=0)
        xc_avg_uncorrected[i, :] = np.mean(xc_small, axis=0)
        # ph = np.unwrap(np.angle(xc_avg_corrected[i, :]))
        ph = np.unwrap(np.angle(xc_avg_uncorrected[i, :]))
        phases[i, :] = ph
        slope, const = np.polyfit(2 * np.pi * new_channels / 4096 / osamp, ph, 1)
        slopes[i] = slope
        ph_noises[i] = np.std(
            ph - (2 * slope * np.pi * new_channels / 4096 / osamp + const)
        )
    # plt.plot(ph_noises)
    # plt.show()
    nu = new_channels / 4096 / osamp 
    print("nu is", nu)
    nant=2
    nbl = nant*(nant-1)//2
    ntime,nfreq=xc_avg_corrected.shape
    print("ntime", ntime, "nfreq", nfreq)
    Ag = -timing.get_grammian(nant)
    print(Ag)
    noise_matrix = np.median(ph_noises) * np.ones((ntime,nbl),dtype='float64')
    phases = np.unwrap(phases,axis=0)
    phase_diff = np.diff(phases,axis=0)
    # plt.plot(phase_diff)
    plt.plot(phases)
    plt.title("unwrapped along time, mutli chan")
    plt.show()
    data_matrix = phases.reshape(ntime,nfreq,1)
    print(data_matrix.shape)
    # plt.plot(data_matrix[50,:,0])
    # plt.plot(phases[50,:])
    # plt.show()
    # AtA,Atd = timing.get_AtA_Atd_independent(data_matrix,Ag,noise_matrix,nu,nant,nfreq,ntime)
    AtA,Atd = timing.get_AtA_Atd(data_matrix,Ag,noise_matrix,nu,nant,nfreq,ntime)
    s,V=np.linalg.eigh(AtA)
    print(s)
    plt.plot(s)
    plt.title("eigvals of AtA")
    plt.show()

    # plt.plot(V[:,0])
    # plt.title("near-singular vector of AtA")
    # plt.show()
    print("ratio of directions", V[-1,0]/V[0,0])
    AtA_inv = np.linalg.inv(AtA)
    # print("last row of AtAinv", AtA_inv[-1,-10,:])
    mfit = AtA_inv@Atd
    mfit=mfit.reshape(-1,nant-1)
    print(mfit.shape)
    bs = nant - 1
    tau_linear = mfit[:ntime * bs,0]
    phi_linear = mfit[ntime * bs:,0]
    print("constant phi0 is", phi_linear)
    pred_phase = 2*np.pi*tau_linear[:,None]*nu[None,:] + phi_linear[0]
    chisq1 = np.sum((phases-pred_phase)**2)
    pred_phase = 2*np.pi*(tau_linear[:,None]+20*4)*nu[None,:] + phi_linear[0]-2.81365632*80
    chisq2 = np.sum((phases-pred_phase)**2)
    print("chisq1 is", chisq1)
    print("chisq2 is", chisq2)
    sys.exit()
    # plt.plot(tau_linear-tau_linear.mean())
    # plt.plot(true_drift-true_drift.mean())
    # plt.show()
    # plt.plot(phi_linear)
    # plt.title("per time phi0")
    plt.title("linear estimate")
    plt.plot(tau_linear)
    plt.show()
    
    print(true_drift.shape,tau_linear.shape)
    # TRY DOING CONJUGATE GRADIENT HERE
    # adev = 1e-8
    # noise = np.median(ph_noises)
    # ddsigma = avglen * 4096 * adev * np.sqrt(2)
    # print("ddsigma", ddsigma, "phnoise estimate", noise)
    # Ninv = 1 / (
    #     np.ones(len(slopes)) * np.median(ph_noises) ** 2
    # )  # start with a constant noise for all time blocks
    # x = 2 * np.pi * new_channels / 4096 / osamp
    # params = np.zeros(2 * len(slopes), dtype="float64")
    # params[: len(slopes)] = 1
    # params[len(slopes) :] = 1
    # cg.cg_linear_model_adev(
    #     params, x, np.ravel(phases), ddsigma, noise, eps=1e-10, niter=100
    # )  # verified that without a prior the solution lines up with polyfit
    # plt.plot(slopes)
    # plt.plot(params[:len(slopes)])
    # plt.show()

    # plt.plot(np.abs(xc_avg[:,4]))
    # plt.plot(np.abs(xc_avg_uncorrected[:,0]),label='uncorrected')
    # plt.plot(np.abs(xc_avg_corrected[:,0]),label='corrected per-block')
    # # plt.axhline(np.mean(auto_avg1[:,0]),label='autos',c='red', lw=3, ls='dashed')
    # plt.legend()
    # # plt.plot(np.abs(average_rows(spec1*np.conj(spec2),nblock=avglen))[:,4])
    # plt.plot(np.abs(xc_avg_uncorrected[:,1]),label='uncorrected')
    # plt.plot(np.abs(xc_avg_corrected[:,1]),label='corrected per-block')
    # # # plt.axhline(np.mean(auto_avg1[:,0]),label='autos',c='red', lw=3, ls='dashed')
    # plt.legend()
    # # # ax=plt.gca()
    # # # ax2=plt.twinx()
    # # # ax2.plot(r_val,label='R value',c='red',alpha=0.5)
    # # # ax2.axhline(0.5,ls='dashed',c='black',lw=3,label='Drift ISNT too linear below')
    # plt.show()
    times = np.arange(len(slopes)) * 16e-6 * avglen
    fig = plt.gcf()
    fig.set_size_inches(10, 4)
    plt.clf()
    plt.title(
        rf"Int. time {avglen*16e-6:4.2f}s, channel $\Delta \nu = {61/osamp:4.2f}$ kHz"
    )
    plt.plot(times, slopes * 4, label="Fitted drift")
    plt.plot(times, tau_linear * 4, label="Fitted drift 2")
    # plt.plot(times, params[: len(slopes)] * 4, label="Fit w/ ADEV prior")
    maderr = mad(slopes + delays[:: blocksize * osamp][: len(slopes)]) * 4
    # maderr2 = (
    #     mad(params[: len(slopes)] + delays[:: blocksize * osamp][: len(slopes)]) * 4
    # )
    # stddev = np.std(slopes + delays[:: blocksize * osamp][: len(slopes)]) * 4
    # print(maderr, maderr2, stddev)
    err = maderr
    plt.plot(
        times,
        true_drift*4,
        label="True drift",
        ls="dashed",
        lw="2",
        c="purple",
    )
    plt.text(
        0.5,
        0.1,
        # f"MAD {int(err)} ns (no prior) vs {int(maderr2)} (w/ prior) ns",
        f"MAD {int(err)} ns (no prior)",
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="right",
    )
    # plt.ylim(-0.5,0.5)
    # plt.ylim(-1000, 2000)
    plt.xlabel("Time (s)")
    plt.ylabel("Drift (ns)")
    plt.legend(loc=4)
    plt.tight_layout()
    # plt.savefig(f"./images/drift_fitted{mult}_adev.png", dpi=300)
    plt.show()

    plt.plot(true_drift-tau_linear)
    plt.show()