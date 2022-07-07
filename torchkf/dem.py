import torch
from .transformations import *
from typing import Dict, Optional, List, Callable, Tuple
import numpy as np
from itertools import chain
import logging
import scipy as sp
import scipy.linalg
from pprint import pformat
from tqdm.autonotebook import tqdm
import pprint
from .dem_de import *
from .dem_z  import *


logging.basicConfig()


from .dem_structs import *
from .dem_dx import *
from .dem_hgm import *

torch.set_default_dtype(torch.float64)



class DEMInversion: 
    def __init__(self, 
                 systems: HierarchicalGaussianModel, 
                 states_embedding_order: int = 4, 
                 causes_embedding_order: int = None): 
        # DEMInversion.check_systems(systems)

        self.M  : List = systems                             # systems, from upper-most to lower-most
        self.n  : int  = states_embedding_order + 1          # embedding order of states
        if causes_embedding_order is not None:
            self.d  : int  = causes_embedding_order + 1      # embedding order of causes
        else: 
            self.d  : int  = self.n                          # embedding order of causes

        self.nl : int  = len(systems)                        # number of levels
        self.nv : int  = sum(M.m for M in self.M)            # number of v (causal states)
        self.nx : int  = sum(M.n for M in self.M)            # number of x (hidden states)
        self.ny : int  = self.M[0].l                         # number of y (model output)
        self.nc : int  = self.M[-1].l                        # number of c (prior causes)
        self.nu : int  = self.d * self.nv + self.n * self.nx # number of generalized states
        self.logger    = logging.getLogger('[DEM]')

    @staticmethod
    def generalized_covariance(
        p   : int,            # derivative order  
        s   : float,          # s.d. of the noise process (1/sqrt(roughness))
        cov : bool = False    # whether to return the precision matrix 
        ):
        """ Mimics the behavior of spm_DEM_R.m
        s is the roughtness of the noise process. 
        """
        if s == 0:
            s = np.exp(-8)


        k = np.arange(p)
        x = np.sqrt(2) * s
        r = np.cumprod(1 - 2. * k) / (x**(2*k))
        S = np.zeros((p,p))
        for i in range(p): 
            j = 2 * k - i
            filt = np.logical_and(j >= 0, j < p)
            S[i,j[filt]] = (-1) ** (i) * r[filt]

        R = np.linalg.inv(S)#.contiguous()

        if cov: 
            return S, R
        else: return R

    @staticmethod
    def generalized_coordinates(
        x    : np.ndarray,    # timeseries
        p    : int,             # embedding order,
        dt   : int = 1          # sampling interval
        ):
        """ series: torch tensor (n_times, dim) 
            inspired from spm_DEM_embed.m
            series_generalized = T * [series[t-order/2], dots, series[t+order/2]]
        """
        n_times, dim = x.shape
        E = np.zeros((p, p))
        x = np.array(x)
        times = np.arange(1, n_times + 1)

        # Create E_ij(t) (note that indices start at 0) 
        for i in range(p): 
            for j in range(p): 
                E[i, j] = float((i + 1 - int(p / 2) * dt)**(j) / np.math.factorial(j))

        # Compute T
        T = np.linalg.inv(E)

        # Compute the slices
        slices = []
        for t in times: 
            start = t - int(p / 2)
            end = start + p

            if start < 0: 
                X = np.concatenate((- start) * [x[0, None]] + [x[0:end]])
            elif end > n_times:
                X = np.concatenate([x[start:n_times]] + (end - n_times) * [x[-1, None]])
            else: 
                X = x[start:end]

            X = T @ X 
            slices.append(X)


        slices = np.stack(slices, axis=0)

        # series_slices is (n_times, order + 1, dim)
        # T is ( order+1, order+1)
        # generalized_coordinates = slices.swapaxes(1, 2) @ T.T

        # return generalized_coordinates.swapaxes(1, 2)
        return slices


    def run(self, 
            y   : np.ndarray,                     # Observed timeseries with shape (time, dimension) 
            u   : Optional[np.ndarray] = None,    # Explanatory variables, inputs or prior expectation of causes
            x   : Optional[np.ndarray] = None,    # Confounds
            nD  : int = 1,                          # Number of D-steps 
            nE  : int = 8,                          # Number of E-steps
            nM  : int = 8,                          # Number of M-steps
            K   : int = 1,                          # Learning rate
            tol : float = np.exp(-4),               # Numerical tolerance
            td  : Optional[float] = None            # Integration time 
            ):
        log = self.logger

        # torch.set_grad_enabled(False)

        # miscellanous variables
        # ----------------------
        M  : HierarchicalGaussianModel  = self.M        # systems, from upper-most (index 0) to lower-most (index -1)
        n  : int                        = self.n        # embedding order of states
        d  : int                        = self.d        # embedding order of causes
        nl : int                        = self.nl       # number of levels
        nv : int                        = self.nv       # number of v (causal states)
        nx : int                        = self.nx       # number of x (hidden states)
        ny : int                        = self.ny       # number of y (model output)
        nc : int                        = self.nc       # number of c (prior causes)
        nu : int                        = self.nu       # number of generalized states
        nT : int                        = y.shape[0]    # number of timesteps
        dt : float                      = self.M.dt     # sampling interval

        if ny != y.shape[1]: raise ValueError('Output dimension mismatch.')

        # conditional moments
        # -------------------
        qU : List[dotdict]       = list()     # conditional moments of model states - q(u) for each time t
        qP : dotdict             = dotdict()  # conditional moments of model parameters - q(p)
        qH : dotdict             = dotdict()  # conditional moments of model hyperparameters - q(h)
        qu : dotdict             = dotdict()  # loop variable for qU
        qp : dotdict             = dotdict()  # loop variable for qP
        qh : dotdict             = dotdict()  # loop variable for qH

        # loop variables 
        # ------
        qE  : List[dotdict]      = list()               # errors
        B   : dotdict            = dotdict()            # saved states
        F   : np.ndarray       = np.zeros(nE)      # Free-energy
        A   : np.ndarray       = np.zeros(nE)      # Free-action

        # prior moments
        # -------------
        pu     = dotdict()  # prior moments of model states - p(u)
        pp     = dotdict()  # prior moments of model parameters - p(p)
        ph     = dotdict()  # prior moments of model hyperparameters - p(h)

        # embedded series
        # ---------------
        
        if y.shape[1] != ny: 
            raise ValueError(f'Last dimension of input y ({y.shape}) does not match that of deepest model cause ({M[0].l})')

        if u is None:
            u = np.zeros((nT, nc))
        elif u.shape[1] != nc: 
            raise ValueError(f'Last dimension of input u ({u.shape}) does not match that of deepest model cause ({M[-1].l})')

        if x is None:
            x = np.zeros((nT, 0))
        
        Y = DEMInversion.generalized_coordinates(y, n, dt) 
        U = np.zeros((nT, n, nc))
        X = np.zeros((nT, n, nc))
        if u.shape[-1] > 0: 
            U[:, :d] = DEMInversion.generalized_coordinates(u, d, dt) 
        if x.shape[-1] > 0:
            X[:, :d] = DEMInversion.generalized_coordinates(x, d, dt) 
        else: X = np.zeros((nT, n, 0))

        # setup integration times
        if td is None: 
            td = 1. / nD
        else: 
            td = td
        te = 0.
        tm = 4.

        # precision components Q requiring [Re]ML estimators (M-step)
        # -----------------------------------------------------------
        Q    = []
        v0, w0 = [], []
        for i in range(nl):
            v0.append(np.zeros((M[i].l, M[i].l)))
            w0.append(np.zeros((M[i].n, M[i].n)))
        V0 = kron(np.zeros((n,n)), sp.linalg.block_diag(*v0))
        W0 = kron(np.zeros((n,n)), sp.linalg.block_diag(*w0))
        Qp = sp.linalg.block_diag(V0, W0)

        # Qp is: 
        # ((ny*n,    0,    0)
        #  (   0, n*nv,    0)
        #  (   0,    0, n*nx) 


        for i in range(nl): 
            # Precision (R) and covariance of generalized errors
            # --------------------------------------------------
            iVv    = DEMInversion.generalized_covariance(n, M[i].sv)
            iVw    = DEMInversion.generalized_covariance(n, M[i].sw)

            # noise on causal states (Q)
            # --------------------------
            for j in range(len(M[i].Q)): 
                q    = list(*v0)
                q[i] = M[i].Q[j]
                Q.append(sp.linalg.block_diag(kron(iVv, sp.linalg.block_diag(*q)), W0))

            # and fixed components (V) 
            # ------------------------
            q    = list(v0)
            q[i] = M[i].V
            Qp  += sp.linalg.block_diag(kron(iVv, sp.linalg.block_diag(*q)), W0)


            # noise on hidden states (R)
            # --------------------------
            for j in range(len(M[i].R)): 
                q    = list(*w0)
                q[i] = M[i].R[j]
                Q.append(sp.linalg.block_diag(V0, kron(iVw, sp.linalg.block_diag(*q))))

            # and fixed components (W) 
            # ------------------------
            q    = list(w0)
            q[i] = M[i].W
            Qp  += sp.linalg.block_diag(V0, kron(iVw, sp.linalg.block_diag(*q)))

        # number of hyperparameters
        # -------------------------
        nh : int = len(Q)  

        # fixed priors on states (u) 
        # --------------------------
        xP              =   sp.linalg.block_diag(*(M[i].xP for i in range(nl)))
        Px              =   kron(DEMInversion.generalized_covariance(n, 0), xP)
        Pv              =   np.zeros((n*nv,n*nv))
        Pv[:d*nv,:d*nv] =   kron(DEMInversion.generalized_covariance(d, 0), np.zeros((nv, nv)))
        Pu              =   sp.linalg.block_diag(Px, Pv)
        Pu[:nu,:nu]     =   Pu[:nu,:nu] + np.eye(nu, nu) * nu * np.finfo(np.float32).eps

        iqu             =   dotdict()

        # hyperpriors 
        # -----------
        hgE   = list(chain((M[i].hE for i in range(nl)), (M[i].gE for i in range(nl))))
        hgC   = chain((M[i].hC for i in range(nl)), (M[i].gC for i in range(nl)))
        ph.h  = np.concatenate(hgE)           # prior expectation on h
        ph.c  = sp.linalg.block_diag(*hgC)    # prior covariance on h
        qh.h  = ph.h                     # conditional expecatation 
        qh.c  = ph.c                     # conditional covariance
        ph.ic = np.linalg.pinv(ph.c)  # prior precision      

        # priors on parameters (in reduced parameter space)
        # =================================================
        pp.c = list() 
        qp.p = list() 
        qp.u = list() 
        for i in range(nl - 1): 
            # eigenvector reduction: p <- pE + qp.u * qp.p 
            # --------------------------------------------
            Ui      = np.linalg.svd(M[i].pC, full_matrices=False)[0]
            M[i].p  = Ui.shape[1]               # number of qp.p

            qp.u.append(Ui)                     # basis for parameters (U from SVD's USV')
            qp.p.append(np.zeros((M[i].p, 1)))    # initial qp.p 
            pp.c.append(Ui.T @ M[i].pC @ Ui)    # prior covariance

        Up = sp.linalg.block_diag(*qp.u)

        # initialize and augment with confound parameters B; with flat priors
        # -------------------------------------------------------------------
        nP    = sum(M[i].p for i in range(nl - 1))  # number of model parameters
        nb    = x.shape[-1]                     # number of confounds
        nn    = nb * ny                         # number of nuisance parameters
        nf    = nP + nn                         # number of free parameters
        ip    = slice(0,nP)
        ib    = slice(nP,nP + nn)
        pp.c  = sp.linalg.block_diag(*pp.c)
        pp.ic = np.linalg.inv(pp.c)
        pp.p  = np.concatenate(qp.p)

        # initialize conditional density q(p) := qp.e (for D-step)
        # --------------------------------------------------------
        qp.e  = list()
        for i in range(nl - 1): 
            try: 
                qp.e.append(qp.p[i] + qp.u[i].T @ (M[i].P - M[i].pE))
            except KeyError: 
                qp.e.append(qp.p[i])
        qp.e  = np.concatenate(qp.e)
        qp.c  = np.zeros((nf, nf))
        qp.b  = np.zeros((ny, nb))

        # initialize dedb
        # ---------------
        # NotImplemented

        # initialize arrays for D-step
        # ============================
        qu.x = np.zeros((n, nx))
        qu.v = np.zeros((n, nv))
        qu.y = np.zeros((n, ny))
        qu.u = np.zeros((n, nc))

        # initialize arrays for hierarchical structure of x[0] and v[0]
        if nx > 0: 
            qu.x[0, :] = np.concatenate([M[i].x for i in range(0, nl-1)], axis=0).squeeze()

        if nv > 0: 
            qu.v[0, :] = np.concatenate([M[i].v for i in range(1,   nl)], axis=0).squeeze()

        # derivatives for Jacobian of D-step 
        # ----------------------------------
        Dx              = kron(np.diag(np.ones((n-1,)), 1), np.eye(nx))
        Dv              = kron(           np.zeros((n, n)), np.eye(nv))
        Dv[:nv*d,:nv*d] = kron(np.diag(np.ones((d-1,)), 1), np.eye(nv))
        Dy              = kron(np.diag(np.ones((n-1,)), 1), np.eye(ny))
        Dc              = kron(           np.zeros((n, n)), np.eye(nv))
        Dc[:nc*d,:nc*d] = kron(np.diag(np.ones((d-1,)), 1), np.eye(nc))
        D               = sp.linalg.block_diag(Dx, Dv, Dy, Dc)

        # and null blocks
        # ---------------
        dVdy  = np.zeros((n * ny, 1))
        dVdc  = np.zeros((n * nc, 1))
        dVdyy = np.zeros((n * ny, n * ny))
        dVdcc = np.zeros((n * nc, n * nc))

        # gradients and curvatures for conditional uncertainty
        # ----------------------------------------------------
        dWdu  = np.zeros((nu, 1))
        dWdp  = np.zeros((nf, 1))
        dWduu = np.zeros((nu, nu))
        dWdpp = np.zeros((nf, nf))

        # preclude unnecessary iterations
        # -------------------------------
        if nh == 0: nM = 1
        if nf == 0 and nh == 0: nE = 1

        # prepare progress bars 
        # ---------------------
        if nE > 1: Ebar = tqdm(desc='  E-step', total=nE)
        if nM > 1: Mbar = tqdm(desc='  M-step', total=nM)
        Tbar = tqdm(desc='timestep', total=nT)

        # preclude very precise states from entering free-energy/action
        # -------------------------------------------------------------
        ix = slice(ny*n + nv*n,ny*n + nv*n + nx*n)
        iv = slice(ny*n,ny*n + nv*d)
        je = np.diag(Qp) < np.exp(16)
        ju = np.concatenate([je[ix], je[iv]])


        # E-step: (with) embedded D- and M-steps) 
        # =======================================
        Fi = - np.inf
        if nE > 1:
            Ebar.reset()

        for iE in range(nE): 

            # [re-]set accumulators for E-step 
            # --------------------------------
            dFdh  = np.zeros((nh, 1))
            dFdhh = np.zeros((nh, nh))
            dFdp  = np.zeros((nf, 1))
            dFdpp = np.zeros((nf, nf))
            qp.ic = 0
            iqu.c = 0
            EE    = 0
            ECE   = 0

            # [re-]set precisions using ReML hyperparameter estimates
            # -------------------------------------------------------
            iS    = Qp + sum(Q[i] * np.exp(qh.h[i]) for i in range(nh))

            # [re-]adjust for confounds
            # -------------------------
            if nb > 0: 
                y     = y - qp.b * x

            # [re-]set states & their derivatives
            # -----------------------------------
            qu = qU[0] if iE > 0 else qu

            # D-step: (nD D-steps for each sample) 
            # ====================================
            Tbar.reset()
            for iT in range(nT): 
                # update progress bar
                # -------------------
                Tbar.update()
                Tbar.refresh()

                # [re-]set states for static systems
                # ----------------------------------
                if nx == 0:
                    qu = qU[iT] if len(qU) > iT else qu
                
                # D-step: until convergence for static systems
                # ============================================ 
                for iD in range(nD): 

                    # sampling time 
                    # not implemented (diff: sampling time does not change with iD)


                    # derivatives of responses and inputs
                    # -----------------------------------
                    qu.y  = Y[iT].copy()
                    qu.u  = U[iT].copy()

                    # compute dEdb (derivatives of confounds)
                    # NotImplemented 

                    # evaluatefunction: 
                    # E = v - g(x,v) and derivatives dE.dx
                    # ====================================
                    E, dE = dem_eval_err_diff(n, d, M, qu, qp)

                    # conditional covariance [of states u]
                    # ------------------------------------
                    qu.p         = dE.du.T @ iS @ dE.du + Pu
                    quc          = np.zeros(((nv+nx)*n, (nv+nx)*n))
                    quc[:nu,:nu] = np.diag(ju.astype(np.float64)) @ np.linalg.inv(qu.p[:nu,:nu]) @ np.diag(ju.astype(np.float64)) 
                    qu.c         = quc
                    # differs from spm_DEM: we use ju to select components of quc that are not very precise
                    # otherwise, some rows and cols of quc are set to 0, and therefore the det is 0
                    # In SPM, this is done internally by spm_logdet 
                    iqu.c        = iqu.c + np.linalg.slogdet(quc[ju][:, ju])[1] 

                    # and conditional covariance [of parameters P]
                    # --------------------------------------------
                    dE.dP = dE.dp # no dedb for now
                    ECEu  = dE.du @ qu.c @ dE.du.T
                    ECEp  = dE.dp @ qp.c @ dE.dp.T

                    if nx == 0: 
                        pass 

                    # save states at iT
                    if iD == 0: 
                        if iE == 0:
                            qE.append(E.squeeze(1))
                            qU.append(dotdict({k: v.copy() for k, v in qu.items()})) 
                        else: 
                            qE[iT] = E.squeeze(1)
                            qU[iT] = dotdict({k: v.copy() for k, v in qu.items()})

                    # uncertainty about parameters dWdv, ...
                    if nP > 0: 
                        CJp   = np.zeros((iS.shape[0] * nP, (nx + nv) * n))
                        dEdpu = np.zeros((iS.shape[0] * nP, (nx + nv) * n))
                        for i in range((nx + nv) * n): 
                            CJp[:, i]   = (qp.c[ip,ip] @ dE.dpu[i].T @ iS).reshape((-1,))
                            dEdpu[:, i] = (dE.dpu[i].T).reshape((-1,))

                        dWdu  = CJp.T @ (dE.dp.T).reshape((-1,1))
                        dWduu = CJp.T @ dEdpu


                    # D-step update: of causes v[i] and hidden states x[i]
                    # ====================================================

                    # conditional modes
                    # -----------------
                    u = np.concatenate([qu.x.reshape((-1,1)), qu.v.reshape((-1,1)), qu.y.reshape((-1,1)), qu.u.reshape((-1,1))])

                    # first-order derivatives
                    dVdu    = - dE.du.T @ iS @ E - dWdu/2 - Pu @ u[0:(nx+nv)*n]

                    # second-order derivatives
                    dVduu   = - dE.du.T @ iS @ dE.du - dWduu / 2 - Pu
                    dVduy   = - dE.du.T @ iS @ dE.dy 
                    dVduc   = - dE.du.T @ iS @ dE.dc
                    
                    # gradient
                    dFdu = np.concatenate([dVdu.reshape((-1,)), dVdy.reshape((-1,)), dVdc.reshape((-1,))], axis=0)

                    # Jacobian (variational flow)
                    dFduu = block_matrix([[dVduu, dVduy, dVduc],
                                        [   [], dVdyy,    []],
                                        [   [],    [], dVdcc]])

                    # update conditional modes of states
                    f     = K * dFdu[..., None]  + D @ u
                    dfdu  = K * dFduu + D
                    du    = compute_dx(f, dfdu, td)
                    q     = u + du

                    # ... and save them 
                    qu.x = q[:nx * n].reshape((n, nx))
                    qu.v = q[n * nx:n * (nx + nv)].reshape((n, nv))

                    # ommit part for static models

                # Gradients and curvatures for E-step 

                if nP > 0: 
                    CJu     = np.zeros(((nx + nv) * n * iS.shape[0], nP))
                    dEdup   = np.zeros(((nx + nv) * n * iS.shape[0], nP))
                    for i in range(nP): 
                        CJu[:, i]   = (qu.c @ dE.dup[i].T @ iS).reshape((-1,))
                        dEdup[:, i] = (dE.dup[i].T).reshape((-1,))
                    dWdp[ip]        = CJu.T @ (dE.du.T).reshape((-1,1))
                    dWdpp[ip,ip]    = CJu.T @ dEdup

                # Accumulate dF/dP = <dL/dp>, dF/dpp = ... 
                dFdp[:, :]  = dFdp  - dWdp / 2 - (dE.dP.T @ iS @ E)
                dFdpp[:,:]  = dFdpp - dWdpp /2 - dE.dP.T @ iS @ dE.dP
                qp.ic       = qp.ic + dE.dP.T @ iS @ dE.dP

                # and quantities for M-step 
                EE  = E @ E.T + EE
                ECE = ECE + ECEu + ECEp

            # M-step - optimize hyperparameters (mh = total update)
            mh = np.zeros((nh,))
            if nM > 1: 
                Mbar.reset()
            for iM in range(nM): 
                # [re-]set precisions using ReML hyperparameter estimates
                iS    = Qp + sum(Q[i] * np.exp(qh.h[i]) for i in range(nh))
                S     = np.linalg.inv(iS)
                dS    = ECE + EE - S * nT 

                # 1st order derivatives 
                for i in range(nh): 
                    dPdh[i] = Q[i] * np.exp(qh.h[i])
                    dFdh[i] = - np.trace(dPdh[i] * dS) / 2

                # 2nd order derivatives 
                for i in range(nh): 
                    for j in range(nh): 
                        dFdhh[i, j] = - np.trace(dPdh[i] * S * dPdh[j] * S * nT)/ 2

                # hyperpriors
                qh.e        = qh.h - ph.h
                if nh > 0:
                    dFdh[:, :]  = dFdh - ph.ic * qh.e
                    dFdhh[:,:]  = dFdhh - ph.ic

                    # update ReML extimate of parameters
                    dh = compute_dx(dFdh, dFdhh, tm, isreg=True)

                    dh   = np.clip(dh, -2, 2)
                    qh.h = qh.h + dh 
                    mh   = mh + dh

                # conditional covariance of hyperparameters 
                qh.c = np.linalg.inv(dFdhh)

                # convergence (M-step)
                if nh > 0 and (((dFdh.T @ dh).squeeze() < tol) or np.linalg.norm(dh, 1) < tol): 
                    break

                if nM > 1: 
                    # update progress bar
                    # -------------------
                    Mbar.update()
                    Mbar.refresh()

            # conditional precision of parameters
            # -----------------------------------
            qp.ic[ip, ip] = qp.ic[ip, ip] + pp.ic
            qp.c = np.linalg.inv(qp.ic)

            # evaluate objective function F
            # =============================

            # free-energy and action 
            # ----------------------
            Lu = - np.trace(iS[je][:, je] @ EE[je][:, je]) / 2 \
                 - n * ny * np.log(2 * np.pi) * nT / 2\
                 + np.linalg.slogdet(iS[je][:, je])[1] * nT / 2\
                 + iqu.c / (2*nD)

            Lp = - np.trace(qp.e.T @ pp.ic @ qp.e) / 2\
                 - np.trace(qh.e.T @ ph.ic @ qh.e) / 2\
                 + np.linalg.slogdet(qp.c[ip][:, ip] @ pp.ic)[1] / 2\
                 + np.linalg.slogdet(qh.c @ ph.ic)[1] / 2

            La = - np.trace(qp.e.T @ pp.ic @ qp.e) * nT / 2\
                 - np.trace(qh.e.T @ ph.ic @ qh.e) * nT / 2\
                 + np.linalg.slogdet(qp.c[ip][:, ip] @ pp.ic * nT)[1] * nT / 2\
                 + np.linalg.slogdet(qh.c @ ph.ic * nT)[1] * nT / 2


            # print(iqu.c)
            # print(np.linalg.slogdet(iS[je][:, je])[1])
            # print(np.trace(iS[je][:, je] @ EE[je][:, je]))
            Li = Lu + Lp
            Ai = Lu + La 
            # print('Fi: ', Fi)

            log.info(f'Li: {Li}')
            log.info(f'Ai: {Ai}')
            if Li == -np.inf: 
                print('Lu: ', Lu)
                print('... - np.trace(iS[je][:, je] @ EE[je][:, je]) / 2', (- np.trace(iS[je][:, je] @ EE[je][:, je]) / 2).item())
                print('... - n * ny * np.log(2 * np.pi) * nT / 2', (- n * ny * np.log(2 * np.pi) * nT / 2).item())
                print('... + np.linalg.slogdet(iS[je][:, je])[1] * nT / 2',  (np.linalg.slogdet(iS[je][:, je])[1] * nT / 2).item())
                print('... + iqu.c / (2*nD)', (iqu.c / (2*nD)).item())


                print('Lp: ', Lp)
                print('...  - np.trace(qp.e.T @ pp.ic @ qp.e) / 2',  - np.trace(qp.e.T @ pp.ic @ qp.e) / 2)
                print('...  - np.trace(qh.e.T @ ph.ic @ qh.e) / 2',  - np.trace(qh.e.T @ ph.ic @ qh.e) / 2)
                print('...  + np.linalg.slogdet(qp.c[ip][:, ip] @ pp.ic)[1] / 2', np.linalg.slogdet(qp.c[ip][:, ip] @ pp.ic)[1] / 2)
                print('...  + np.linalg.slogdet(qh.c @ ph.ic)[1] / 2',  np.linalg.slogdet(qh.c @ ph.ic)[1] / 2)

            # if F is increasng save expansion point and derivatives 
            if Li > Fi or iE < 1: 
                # Accept free-energy and save current parameter estimates
                #--------------------------------------------------------
                Fi      = Li
                te      = min(te + 1/2.,4.)
                tm      = min(tm + 1/2.,4.)
                B.qp    = dotdict(**qp)
                B.qh    = dotdict(**qh)
                B.pp    = dotdict(**pp)

                ## TODO : PB with loop ? 

                # E-step: update expectation (p)
                # ==============================
                
                # gradients and curvatures
                # ------------------------
                dFdp[ip]         = dFdp[ip]         - pp.ic @ (qp.e - pp.p)
                dFdpp[ip][:, ip] = dFdpp[ip][:, ip] - pp.ic
                
                # update conditional expectation
                # ------------------------------
                dp      = compute_dx(dFdp, dFdpp, te, isreg=True) 
                qp.e    = qp.e + dp[ip]
                qp.p    = list()
                npi     = 0
                for i in range(nl - 1): 
                    qp.p.append(qp.e[npi:npi + M[i].p])
                    npi += M[i].p
                qp.b    = dp[ib]

            else:
                
                # otherwise, return to previous expansion point
                # ---------------------------------------------
                nM      = 1;
                qp      = dotdict(**B.qp)
                pp      = dotdict(**B.pp)
                qh      = dotdict(**B.qh)
                te      = min(te - 2, -2)
                tm      = min(tm - 2, -2)
                dp      = np.zeros_like(dp)
                
            
            F[iE]  = Fi;
            A[iE]  = Ai;


            log.info(f'Li: {Li}')
            log.info(f'Ai: {Ai}')

            # Check convergence 
            if np.linalg.norm(dp.reshape((-1,)), 1) <= tol * np.linalg.norm(np.concatenate(qp.p).reshape((-1,)), 1) and np.linalg.norm(mh.reshape((-1,)), 1) <= tol: 
                break 
            if te < -8: 
                break

            # update progress bar
            # -------------------
            if nE > 1:
                Ebar.update()
                Ebar.refresh()

        results    = dotdict()
        results.F  = F
        results.A  = A

        qH.h       = qh.h
        qH.C       = qh.c

        results.qH = qH

        qP.P       = Up @ qp.e + np.concatenate([m.pE for m in M])
        qP.C       = Up @ qp.c[ip][:, ip] @ Up.T
        qP.dFdp    = Up @ dFdp[ip]
        qP.dFdpp   = Up @ dFdpp[ip][:, ip] @ Up.T

        results.qP = qP

        results.qU = dotdict({k: np.stack([qU[i][k] for i in range(len(qU))], axis=0) for k in qU[0].keys()})
        results.qE = qE
        

        return results

    def generate(self, nT, u=None): 

        # torch.set_grad_enabled(False)

        n  = self.n                 # Derivative order
        M  = self.M
        dt = self.M.dt
        nl = len(M)                 # Number of levels
        nx = sum(m.n for m in M)    # Number of states
        nv = sum(m.l for m in M)    # Number of outputs

        z, w  = dem_z(M, nT)
        # inputs are integrated as random innovations
        if u is not None: 
            z[-1] = u + z[-1]

        Z    = [DEMInversion.generalized_coordinates(zi, n, dt)[..., None] for zi in z]
        W    = [DEMInversion.generalized_coordinates(wi, n, dt)[..., None] for wi in w]
        X    = np.zeros((nT, n, nx, 1))
        V    = np.zeros((nT, n, nv, 1))

        # Setup initial conditions
        X[0, 0] = np.concatenate([m.x for m in M if m.n > 0], axis=0)
        V[0, 0] = np.concatenate([m.v for m in M if m.l > 0], axis=0)

        # Derivatives operators
        Dx = kron(np.diag(np.ones((n-1,)), 1), np.eye(nx));
        Dv = kron(np.diag(np.ones((n-1,)), 1), np.eye(nv));
        D  = sp.linalg.block_diag(Dv, Dx, Dv, Dx)
        dfdw  = kron(np.eye(n),np.eye(nx));

        xt = X[0]
        vt = V[0]
        for t in tqdm(range(0, nT)):     
            # Unpack state
            zi = [_[t] for _ in Z]
            wi = [_[t] for _ in W] 

            # Unvec states
            nxi = 0
            nvi = 0
            xi  = []
            vi  = []
            dfdx = cell(nl, nl)
            dfdv = cell(nl, nl)
            dgdx = cell(nl, nl)
            dgdv = cell(nl, nl)
            for i in range(nl): 
                xi.append(xt[:, nxi:nxi + M[i].n])
                vi.append(vt[:, nvi:nvi + M[i].l])
                
                nxi = nxi + M[i].n
                nvi = nvi + M[i].l
                
                # Fill cells 
                dfdx[i, i] = np.zeros((M[i].n, M[i].n))
                dfdv[i, i] = np.zeros((M[i].n, M[i].l))
                dgdx[i, i] = np.zeros((M[i].l, M[i].n))
                dgdv[i, i] = np.zeros((M[i].l, M[i].l))
                
            f   = []
            g   = []
            df  = []
            dg  = []
            
            # Run in descending order
            vi[-1][0] = zi[-1][0] 
            for i in range(nl - 1)[::-1]: 
                p = M[i].pE 
                
                # compute functions
                fi = M[i].f(xi[i][0], vi[i + 1][0], p)
                gi = M[i].g(xi[i][0], vi[i + 1][0], p)
                
                xv = tuple(_ if sum(_.shape) > 0 else np.empty(_.shape) for _ in  (xi[i][0], vi[i+1][0]))
                
                # compute derivatives
                if M[i].df is not None:
                    dfi = M[i].df(*xv, M[i].pE)
                else: 
                    dfi, _ = compute_df_d2f(lambda x, v: M[i].f(x, v, M[i].pE), xv, ['dx', 'dv'])

                if M[i].dg is not None:
                    dgi = M[i].dg(*xv, M[i].pE)
                else:
                    dgi, _ = compute_df_d2f(lambda x, v: M[i].g(x, v, M[i].pE), xv, ['dx', 'dv']) 

                # g(x, v) && f(x, v)
                vi[i][0] = gi + zi[i][0]
                f.append(fi)
                g.append(gi)
                
                # and partial derivatives
                dfdx[i,     i] = dfi.dx
                dfdv[i, i + 1] = dfi.dv
                dgdx[i,     i] = dgi.dx
                dgdv[i, i + 1] = dgi.dv
                
                df.append(dfi)
                dg.append(dgi)
            
            f = np.concatenate(f)
            g = np.concatenate(g)
            
            dfdx = block_matrix(dfdx)
            dfdv = block_matrix(dfdv)
            dgdx = block_matrix(dgdx)
            dgdv = block_matrix(dgdv)
            
            v = np.concatenate(vi, axis=1)
            x = np.concatenate(xi, axis=1) 

            z = np.concatenate(zi, axis=1)
            w = np.concatenate(wi, axis=1)

            # x[0, :] = 
            x[1, :] = f + w[0]

            # compute higher orders
            for i in range(1, n-1): 
                v[i]   = dgdx @ x[i] + dgdv @ v[i] + z[i]
                x[i+1] = dfdx @ x[i] + dfdv @ v[i] + w[i]
            
            dgdv = kron(np.diag(np.ones((n-1,)),1), dgdv)
            dgdx = kron(np.diag(np.ones((n-1,)),1), dgdx)
            dfdv = kron(np.eye(n), dfdv)
            dfdx = kron(np.eye(n), dfdx)

            # Save realization
            V[t] = v
            X[t] = x
            
            J    = block_matrix([
                [dgdv, dgdx,   Dv,   []] , 
                [dfdv, dfdx,   [], dfdw],  
                [[],     [],   Dv,   []],  
                [[],     [],   [],   Dx]   
            ])
            
            u  = np.concatenate([v.reshape((-1,)), x.reshape((-1,)), z.reshape((-1,)), w.reshape((-1,))])[..., None]
            du = compute_dx(D @ u, J, dt)

            u  = u + du
            
            vt   = u[:v.shape[0] * v.shape[1]].reshape(v.shape)
            xt   = u[ v.shape[0] * v.shape[1]:v.shape[0] * v.shape[1] + x.shape[0] * x.shape[1]].reshape(x.shape)
            
        results   = dotdict()
        results.v = V
        results.x = X
        results.z = np.concatenate(Z, axis=2) 
        results.w = np.concatenate(W, axis=2)

        return results