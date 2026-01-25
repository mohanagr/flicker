import numpy as np
import numba as nb


@nb.njit()
def symmetrize(A):
    nr, nc = A.shape
    for i in range(nr):
        for j in range(i, nc):
            A[j, i] = A[i, j]


def get_grammian(nant):
    nbl = (nant - 1) * nant // 2
    Ag = np.zeros((nbl, nant - 1), dtype="float64")
    b = 0
    for i in range(nant):
        for j in range(i + 1, nant):
            # print(i,j)
            if i == 0:
                Ag[b, j - 1] = -1
            else:
                Ag[b, i - 1] = 1
                Ag[b, j - 1] = -1
            b += 1
    return Ag


def get_AtA_Atd(data, Ag, noise_var, nu, nant, nfreq, ntime):
    # data is ntime, nfreq, nbl, C-major
    nparam = (nant - 1) * ntime + nant - 1
    nbl = (nant - 1) * nant // 2
    npertime = nfreq * nbl
    myAtA = np.empty((nparam, nparam), dtype="float64")
    myAtd = np.empty((nparam,), dtype="float64")
    myAtA[:] = 0.0
    myAtd[:] = 0.0
    two_pi = 2 * np.pi
    Snu2 = np.sum(nu**2) * two_pi**2
    Snu = np.sum(nu) * two_pi
    bs = nant - 1  # block size
    # print("nant", nant, "nbl", nbl, "ntime", ntime, "nfreq", nfreq, "bs", bs)
    # print("d shape", d.shape)
    # print("n pertime", npertime)
    for i in range(ntime):
        Agw = Ag / noise_var[i, :][:, None]  # whitened A
        myAtA[i * bs : (i + 1) * bs, i * bs : (i + 1) * bs] = Ag.T @ Agw * Snu2
        myAtA[i * bs : (i + 1) * bs, ntime * bs :] = Ag.T @ Agw * Snu
        myAtA[ntime * bs :, ntime * bs :] += Ag.T @ Agw * nfreq
        for j in range(nfreq):
            idx = i * npertime + j * nbl
            # print("start idx", idx)
            # Aa = Agw*nu[j]
            # myAtd[i*bs :(i+1)*bs]+= nu[j]*Agw.T@d[idx:idx+nbl]
            vec = Agw.T @ data[i, j, :]
            myAtd[i * bs : (i + 1) * bs] += two_pi * nu[j] * vec
            myAtd[-bs:] += vec
    symmetrize(myAtA)
    return myAtA, myAtd


# @nb.njit(parallel=True)
# def chisq_lm(params,data,B,nu,weights,tau_ref,scale,nant=6):
#     # data is now for each baseline
#     # params is now 2-D (nant-1) x n_params_per_ant
#     # weights is 2-D ntime x nbl
#     nbl = data.shape[2]
#     ntime = data.shape[0]
#     nfreq = data.shape[1]
#     params = params.reshape(nant-1,-1)
#     n_eval = ntime*nfreq*nbl
#     nk = B.shape[1] #spline basis matrix (ntime , nk)
#     residuals = np.empty((2*ntime*nfreq*nbl,),dtype='float64')
#     for i in nb.prange(2*n_eval):
#         residuals[i]=0
#     tau_t = np.zeros((nant-1, ntime),dtype='float64')
#     for i in range(0,nant-1):
#         ci=params[i,:-1] # basis coeffs
#         tau_t[i,:]= tau_ref[i]  + scale[i]*B@ci
#     phi0 = params[:,-1].copy()
#     tau_t = tau_t.T.copy() #transpose for faster access since nbl << ntime
#     # fill pred_phase and Jacobian
#     for i in nb.prange(ntime):
#         for j in range(nfreq):
#             blnum=0
#             two_pi_nu = 2*np.pi*nu[j]
#             for ai in range(nant):
#                 for aj in range(ai+1,nant):
#                     rowidx = i*nfreq*nbl + j*nbl + blnum
#                     if ai==0:
#                         pred = -two_pi_nu*tau_t[i,aj-1] - phi0[aj-1] #ant 0 is refant
#                     else:
#                         pred = two_pi_nu*(tau_t[i,ai-1]-tau_t[i,aj-1]) + phi0[ai-1]-phi0[aj-1]
#                     res = np.exp(1j*data[i,j,blnum])-np.exp(1j*pred)
#                     w=weights[i, blnum]
#                     residuals[rowidx]=w*np.real(res)
#                     residuals[rowidx+n_eval]=w*np.imag(res)
#                     blnum+=1
#     return residuals


@nb.njit(parallel=True)
def chisq_lm(params, data, B, nu, weights, tau_ref, scale, lamda, nant=6):
    # data is now for each baseline
    # params is now 2-D (nant-1) x n_params_per_ant
    # weights is 2-D ntime x nbl
    nbl = data.shape[2]
    ntime = data.shape[0]
    nfreq = data.shape[1]
    params = params.reshape(nant - 1, -1)
    nparam_per_ant = params.shape[1]
    n_eval = ntime * nfreq * nbl
    nk = B.shape[1]  # spline basis matrix (ntime , nk)
    prior_length = (nant - 1) * (
        params.shape[1] - 1
    )  # number of total spline coefficients
    residuals = np.empty((2 * ntime * nfreq * nbl + prior_length,), dtype="float64")
    for i in nb.prange(2 * n_eval + prior_length):
        residuals[i] = 0
    tau_t = np.zeros((nant - 1, ntime), dtype="float64")
    for i in range(0, nant - 1):
        ci = params[i, :-1]  # basis coeffs
        tau_t[i, :] = tau_ref[i] + scale[i] * B @ ci
    phi0 = params[:, -1].copy()
    tau_t = tau_t.T.copy()  # transpose for faster access since nbl << ntime
    # fill pred_phase and Jacobian
    for i in nb.prange(ntime):
        for j in range(nfreq):
            blnum = 0
            two_pi_nu = 2 * np.pi * nu[j]
            for ai in range(nant):
                for aj in range(ai + 1, nant):
                    rowidx = i * nfreq * nbl + j * nbl + blnum
                    if ai == 0:
                        pred = (
                            -two_pi_nu * tau_t[i, aj - 1] - phi0[aj - 1]
                        )  # ant 0 is refant
                    else:
                        pred = (
                            two_pi_nu * (tau_t[i, ai - 1] - tau_t[i, aj - 1])
                            + phi0[ai - 1]
                            - phi0[aj - 1]
                        )
                    res = np.exp(1j * data[i, j, blnum]) - np.exp(1j * pred)
                    w = weights[i, blnum]
                    residuals[rowidx] = w * np.real(res)
                    residuals[rowidx + n_eval] = w * np.imag(res)
                    blnum += 1
    idx = 2 * n_eval
    for i in range(params.shape[0]):  # over nant-1
        for j in range(params.shape[1] - 1):  # over num of spline coeffs per antenna
            residuals[idx + i * (nparam_per_ant) + j] = (
                np.sqrt(lamda) * params[i, j]
            )  # skip the constant phi0, no prior on that
    return residuals


@nb.njit(parallel=True)
def jac_lm(params, data, B, nu, weights, tau_ref, scale, lamda, nant=6):
    # data is now for each baseline
    # params is now 2-D (nant-1) x n_params_per_ant
    # weights is 2-D ntime x nbl
    nbl = data.shape[2]
    ntime = data.shape[0]
    nfreq = data.shape[1]
    params = params.reshape(nant - 1, -1)
    nparam = np.size(params)  # coeffs and constant phi0 for nant-1
    nparam_per_ant = params.shape[1]
    n_eval = ntime * nfreq * nbl
    prior_length = (nant - 1) * (
        params.shape[1] - 1
    )  # number of total spline coefficients
    nk = B.shape[1]  # spline basis matrix (ntime , nk)
    J = np.empty((2 * n_eval + prior_length, nparam), dtype="float64")
    Jflat = np.ravel(J)
    nn = J.shape[0] * J.shape[1]
    for m in nb.prange(nn):
        Jflat[m] = 0
    tau_t = np.zeros((nant - 1, ntime), dtype="float64")
    phi0 = params[:, -1].copy()
    blnum = 0
    for i in range(0, nant - 1):
        ci = params[i, :-1]  # basis coeffs
        tau_t[i, :] = tau_ref[i] + scale[i] * B @ ci
    tau_t = tau_t.T.copy()  # transpose for faster access since nbl << ntime
    # fill pred_phase and Jacobian
    for i in nb.prange(ntime):
        for j in range(nfreq):
            blnum = 0
            two_pi_nu = 2 * np.pi * nu[j]
            for ai in range(nant):
                for aj in range(ai + 1, nant):
                    rowidx = i * nfreq * nbl + j * nbl + blnum
                    w = weights[i, blnum]
                    if ai == 0:
                        pred = (
                            -two_pi_nu * tau_t[i, aj - 1] - phi0[aj - 1]
                        )  # ant 0 is refant
                        ym = np.exp(1j * pred)
                        dtheta = -1j * ym * w
                        for k in range(nk):
                            temp = two_pi_nu * B[i, k]
                            colidx2 = (
                                aj - 1
                            ) * nparam_per_ant + k  # only gotta fill antenna i+1 and onwards for ant 0
                            re = temp * np.real(dtheta)
                            im = temp * np.imag(dtheta)
                            J[rowidx, colidx2] = -re
                            J[rowidx + n_eval, colidx2] = -im
                        re = np.real(dtheta)
                        im = np.imag(dtheta)
                        colidx2 = (aj - 1) * nparam_per_ant + nk
                        J[rowidx, colidx2] = -re
                        J[rowidx + n_eval, colidx2] = -im
                    else:
                        pred = (
                            two_pi_nu * (tau_t[i, ai - 1] - tau_t[i, aj - 1])
                            + phi0[ai - 1]
                            - phi0[aj - 1]
                        )
                        ym = np.exp(1j * pred)
                        dtheta = -1j * ym * w
                        for k in range(nk):
                            temp = two_pi_nu * B[i, k]
                            colidx1 = (ai - 1) * nparam_per_ant + k
                            colidx2 = (aj - 1) * nparam_per_ant + k
                            re = temp * np.real(dtheta)
                            im = temp * np.imag(dtheta)
                            J[rowidx, colidx1] = re
                            J[rowidx, colidx2] = -re
                            J[rowidx + n_eval, colidx1] = im
                            J[rowidx + n_eval, colidx2] = -im
                        re = np.real(dtheta)
                        im = np.imag(dtheta)
                        colidx1 = (ai - 1) * nparam_per_ant + nk
                        colidx2 = (aj - 1) * nparam_per_ant + nk
                        J[rowidx, colidx1] = re
                        J[rowidx, colidx2] = -re
                        J[rowidx + n_eval, colidx1] = im
                        J[rowidx + n_eval, colidx2] = -im
                    blnum += 1
    idx = 2 * n_eval
    for i in range(nant - 1):
        for j in range(nparam_per_ant - 1):
            k = i * nparam_per_ant + j
            J[idx + k, k] = np.sqrt(lamda)
    return J


@nb.njit(parallel=True)
def chisq_lm_full(params, data, nu, weights, lamda, nant=6):
    # data is now for each baseline
    # params is now 2-D (nant-1) x n_params_per_ant
    # weights is 2-D ntime x nbl
    nbl = data.shape[2]
    ntime = data.shape[0]
    nfreq = data.shape[1]

    n_eval = ntime * nfreq * nbl
    prior_length = ntime * (nant - 1)
    residuals = np.empty((2 * ntime * nfreq * nbl + prior_length,), dtype="float64")
    for i in nb.prange(2 * n_eval + prior_length):
        residuals[i] = 0

    tau_t = params[: ntime * (nant - 1)].reshape(-1, nant - 1)
    phi0 = params[ntime * (nant - 1) :]
    # fill pred_phase and Jacobian
    for i in nb.prange(ntime):
        for j in range(nfreq):
            blnum = 0
            two_pi_nu = 2 * np.pi * nu[j]
            for ai in range(nant):
                for aj in range(ai + 1, nant):
                    rowidx = i * nfreq * nbl + j * nbl + blnum
                    if ai == 0:
                        pred = (
                            -two_pi_nu * tau_t[i, aj - 1] - phi0[aj - 1]
                        )  # ant 0 is refant
                    else:
                        pred = (
                            two_pi_nu * (tau_t[i, ai - 1] - tau_t[i, aj - 1])
                            + phi0[ai - 1]
                            - phi0[aj - 1]
                        )
                    res = np.exp(1j * data[i, j, blnum]) - np.exp(1j * pred)
                    w = weights[i, blnum]
                    residuals[rowidx] = w * np.real(res)
                    residuals[rowidx + n_eval] = w * np.imag(res)
                    blnum += 1
    idx = 2 * n_eval
    residuals[idx : idx + (nant - 1)] = np.sqrt(lamda) * (
        -2 * tau_t[0, :] + 2 * tau_t[1, :]
    )
    for i in range(1, ntime - 1):  # over nant-1
        for j in range(nant - 1):  # over num of spline coeffs per antenna
            residuals[idx + i * (nant - 1) + j] = np.sqrt(lamda) * (
                tau_t[i - 1, j] - 2 * tau_t[i, j] + tau_t[i + 1, j]
            )
    i = ntime - 1
    residuals[idx + i * (nant - 1) : idx + i * (nant - 1) + (nant - 1)] = np.sqrt(
        lamda
    ) * (-2 * tau_t[i, :] + 2 * tau_t[i - 1, :])
    return residuals


@nb.njit(parallel=True)
def jac_lm_full(params, data, nu, weights, lamda, nant=6):
    # this function is to fit one delay per time sample.
    # data is now for each baseline
    # params is now 2-D (nant-1) x ntime + 1
    # weights is 2-D ntime x nbl
    nbl = data.shape[2]
    ntime = data.shape[0]
    nfreq = data.shape[1]

    n_eval = ntime * nfreq * nbl
    prior_length = (nant - 1) * ntime
    J = np.empty(
        (2 * n_eval + prior_length, (nant - 1) * ntime + nant - 1), dtype="float64"
    )
    Jflat = np.ravel(J)
    nn = J.shape[0] * J.shape[1]
    for m in nb.prange(nn):
        Jflat[m] = 0.0
    tau_t = params[: ntime * (nant - 1)].reshape(-1, nant - 1)
    phi0 = params[ntime * (nant - 1) :]
    blnum = 0
    # fill pred_phase and Jacobian
    for i in nb.prange(ntime):
        for j in range(nfreq):
            blnum = 0
            two_pi_nu = 2 * np.pi * nu[j]
            for ai in range(nant):
                for aj in range(ai + 1, nant):
                    rowidx = i * nfreq * nbl + j * nbl + blnum
                    w = weights[i, blnum]
                    if ai == 0:
                        pred = (
                            -two_pi_nu * tau_t[i, aj - 1] - phi0[aj - 1]
                        )  # ant 0 is refant
                        ym = np.exp(1j * pred)
                        dtheta = -1j * ym * w
                        colidx2 = i * (nant - 1) + aj - 1
                        re = two_pi_nu * np.real(dtheta)
                        im = two_pi_nu * np.imag(dtheta)
                        J[rowidx, colidx2] = -re
                        J[rowidx + n_eval, colidx2] = -im
                        re = np.real(dtheta)
                        im = np.imag(dtheta)
                        colidx2 = ntime * (nant - 1) + aj - 1  # constant phi0
                        J[rowidx, colidx2] = -re
                        J[rowidx + n_eval, colidx2] = -im
                    else:
                        pred = (
                            two_pi_nu * (tau_t[i, ai - 1] - tau_t[i, aj - 1])
                            + phi0[ai - 1]
                            - phi0[aj - 1]
                        )
                        ym = np.exp(1j * pred)
                        dtheta = -1j * ym * w
                        colidx1 = i * (nant - 1) + ai - 1
                        colidx2 = i * (nant - 1) + aj - 1
                        re = two_pi_nu * np.real(dtheta)
                        im = two_pi_nu * np.imag(dtheta)
                        J[rowidx, colidx1] = re
                        J[rowidx, colidx2] = -re
                        J[rowidx + n_eval, colidx1] = im
                        J[rowidx + n_eval, colidx2] = -im
                        re = np.real(dtheta)
                        im = np.imag(dtheta)
                        colidx1 = ntime * (nant - 1) + ai - 1
                        colidx2 = ntime * (nant - 1) + aj - 1
                        J[rowidx, colidx1] = re
                        J[rowidx, colidx2] = -re
                        J[rowidx + n_eval, colidx1] = im
                        J[rowidx + n_eval, colidx2] = -im
                    blnum += 1
    idx = 2 * n_eval
    rowidx = idx
    for j in range(nant - 1):
        colidx = 0
        J[rowidx + j, colidx + j] = -2 * np.sqrt(lamda)
        colidx = nant - 1
        J[rowidx + j, colidx + j] = 2 * np.sqrt(lamda)
    for i in range(1, ntime - 1):
        rowidx = idx + i * (nant - 1)
        for j in range(nant - 1):
            colidx = (i - 1) * (nant - 1)
            J[rowidx + j, colidx + j] = np.sqrt(lamda)
            colidx = i * (nant - 1)
            J[rowidx + j, colidx + j] = -2 * np.sqrt(lamda)
            colidx = (i + 1) * (nant - 1)
            J[rowidx + j, colidx + j] = np.sqrt(lamda)
    i = ntime - 1
    rowidx = idx + i * (nant - 1)
    for j in range(nant - 1):
        colidx = (i - 1) * (nant - 1)
        J[rowidx + j, colidx + j] = 2 * np.sqrt(lamda)
        colidx = i * (nant - 1)
        J[rowidx + j, colidx + j] = -2 * np.sqrt(lamda)
    return J


# @nb.njit(parallel=True)
# def jac_lm(params,data,B,nu,weights,tau_ref,scale,nant=6):
#     # data is now for each baseline
#     # params is now 2-D (nant-1) x n_params_per_ant
#     # weights is 2-D ntime x nbl
#     nbl = data.shape[2]
#     ntime = data.shape[0]
#     nfreq = data.shape[1]
#     params = params.reshape(nant-1,-1)
#     nparam = np.size(params) #coeffs and constant phi0 for nant-1
#     nparam_per_ant = params.shape[1]
#     n_eval = ntime*nfreq*nbl
#     nk = B.shape[1] #spline basis matrix (ntime , nk)
#     J = np.empty((2*ntime*nfreq*nbl,nparam),dtype='float64')
#     Jflat=np.ravel(J)
#     nn=2*ntime*nfreq*nbl*nparam
#     for i in nb.prange(nn):
#         Jflat[i]=0
#     tau_t = np.zeros((nant-1, ntime),dtype='float64')
#     phi0 = params[:,-1].copy()
#     blnum=0
#     for i in range(0,nant-1):
#         ci=params[i,:-1] # basis coeffs
#         tau_t[i,:]= tau_ref[i]  + scale[i]*B@ci
#     tau_t = tau_t.T.copy() #transpose for faster access since nbl << ntime
#     # fill pred_phase and Jacobian
#     for i in nb.prange(ntime):
#         for j in range(nfreq):
#             blnum=0
#             two_pi_nu = 2*np.pi*nu[j]
#             for ai in range(nant):
#                 for aj in range(ai+1,nant):
#                     rowidx = i*nfreq*nbl + j*nbl + blnum
#                     w = weights[i, blnum]
#                     if ai==0:
#                         pred = -two_pi_nu*tau_t[i,aj-1] - phi0[aj-1] #ant 0 is refant
#                         ym = np.exp(1j*pred)
#                         dtheta = -1j*ym*w
#                         for k in range(nk):
#                             temp = two_pi_nu*B[i,k]
#                             colidx2 = (aj-1)*nparam_per_ant + k #only gotta fill antenna i+1 and onwards for ant 0
#                             re = temp*np.real(dtheta)
#                             im = temp*np.imag(dtheta)
#                             J[rowidx, colidx2] = -re
#                             J[rowidx+n_eval,colidx2] = -im
#                         re = np.real(dtheta)
#                         im = np.imag(dtheta)
#                         colidx2 = (aj-1)*nparam_per_ant + nk
#                         J[rowidx, colidx2] = -re
#                         J[rowidx+n_eval,colidx2] = -im
#                     else:
#                         pred = two_pi_nu*(tau_t[i,ai-1]-tau_t[i,aj-1]) + phi0[ai-1]-phi0[aj-1]
#                         ym = np.exp(1j*pred)
#                         dtheta = -1j*ym*w
#                         for k in range(nk):
#                             temp = two_pi_nu*B[i,k]
#                             colidx1 = (ai-1)*nparam_per_ant + k
#                             colidx2 = (aj-1)*nparam_per_ant + k
#                             re = temp*np.real(dtheta)
#                             im = temp*np.imag(dtheta)
#                             J[rowidx, colidx1] = re
#                             J[rowidx, colidx2] = -re
#                             J[rowidx+n_eval,colidx1] = im
#                             J[rowidx+n_eval,colidx2] = -im
#                         re = np.real(dtheta)
#                         im = np.imag(dtheta)
#                         colidx1 = (ai-1)*nparam_per_ant + nk
#                         colidx2 = (aj-1)*nparam_per_ant + nk
#                         J[rowidx, colidx1] = re
#                         J[rowidx, colidx2] = -re
#                         J[rowidx+n_eval,colidx1] = im
#                         J[rowidx+n_eval,colidx2] = -im
#                     blnum+=1
#     return J

nb.njit(parallel=True)
def chisq_admm(tau_phi0, tau_smooth, u, z, rho, data, nu, weights, tau_ref, scale=1):
    ntime = data.shape[0]
    nfreq = data.shape[1]
    n_eval = data.shape[0] * data.shape[1]
    phi0 = tau_phi0[-1]
    tau = tau_phi0[:-1]
    res = np.zeros((2 * n_eval + ntime,), dtype="float64")
    for i in nb.prange(ntime):
        w = weights[i]
        for j in range(nfreq):
            two_pi_nu = 2 * np.pi * nu[j]
            pred = two_pi_nu * (tau_ref + scale * tau[i]) + phi0
            dtheta = np.exp(1j * data[i, j]) - np.exp(1j * pred)
            res[i * nfreq + j] = w * np.real(dtheta)
            res[i * nfreq + j + n_eval] = w * np.imag(dtheta)
        res[i + 2 * n_eval] = np.sqrt(rho) * (tau[i] - tau_smooth[i] - z[i] + u[i])
    return res


@nb.njit(parallel=True)
def jac_admm(tau_phi0, tau_smooth, u, z, rho, data, nu, weights, tau_ref, scale=1):
    n_eval = data.shape[0] * data.shape[1]
    ntime = data.shape[0]
    nfreq = data.shape[1]
    phi0 = tau_phi0[-1]
    tau = tau_phi0[:-1]
    J = np.zeros((2 * n_eval + ntime, ntime + 1), dtype="float64")
    # fill pred_phase and Jacobian
    for i in nb.prange(ntime):
        w = weights[i]
        for j in range(nfreq):
            two_pi_nu = 2 * np.pi * nu[j]
            pred = two_pi_nu * (tau_ref + scale * tau[i]) + phi0
            ym = np.exp(1j * pred)
            dtheta = -1j * two_pi_nu * ym
            J[i * nfreq + j, i] = w * np.real(dtheta)
            J[i * nfreq + j + n_eval, i] = w * np.imag(dtheta)
            dtheta = -1j * ym
            J[i * nfreq + j, -1] = w * np.real(dtheta)
            J[i * nfreq + j + n_eval, -1] = w * np.imag(dtheta)
        J[i + 2 * n_eval, i] = np.sqrt(rho)
    return J

