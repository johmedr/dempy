import torch
from .transformations import *
from typing import Dict, Optional, List, Callable, Tuple
import numpy as np
from itertools import chain
import logging
from pprint import pformat
from tqdm import tqdm
from .dem_de import *


logging.basicConfig()


from .dem_structs import *
from .dem_dx import *
from .dem_hgm import *

torch.set_default_dtype(torch.float64)



class DEMInversion: 
    def __init__(self, 
                 systems: HierarchicalGaussianModel, 
                 states_embedding_order: int = 4, 
                 causes_embedding_order: int = 4): 
        # DEMInversion.check_systems(systems)

        self.M  : List = systems                             # systems, from upper-most to lower-most
        self.n  : int  = states_embedding_order + 1          # embedding order of states
        self.d  : int  = causes_embedding_order + 1          # embedding order of causes
        self.nl : int  = len(systems)                        # number of levels
        self.nv : int  = sum(M.m for M in self.M)            # number of v (causal states)
        self.nx : int  = sum(M.n for M in self.M)            # number of x (hidden states)
        self.ny : int  = self.M[0].l                        # number of y (model output)
        self.nc : int  = self.M[-1].l                         # number of c (prior causes)
        self.nu : int  = self.d * self.nv + self.n * self.nx # number of generalized states
        self.logger    = logging.getLogger('[DEM]')

    @staticmethod
    def generalized_covariance(
        p    : int,           # derivative order  
        s    : float,         # roughness of the noise process
        cov : bool = False   # whether to return the precision matrix 
        ):
        """ Mimics the behavior of spm_DEM_R.m
        s is the roughtness of the noise process. 
        """
        if s == 0:
            s = np.exp(-8)

        k = torch.arange(p)
        x = np.sqrt(2) * s
        r = np.cumprod(1 - 2 * k) / (x**(2*k))
        S = torch.zeros((p,p))
        for i in range(p): 
            j = 2 * k - i
            filt = torch.logical_and(j >= 0, j < p)
            S[i,j[filt]] = (-1) ** (i) * r[filt]

        R = torch.linalg.inv(S).contiguous()

        if cov: 
            return S, R
        else: return R

    @staticmethod
    def generalized_coordinates(
        x    : torch.Tensor,    # timeseries
        p    : int,             # embedding order
        ):
        """ series: torch tensor (n_times, dim) 
            inspired from spm_DEM_embed.m
            series_generalized = T * [series[t-order/2], dots, series[t+order/2]]
        """
        n_times, dim = x.shape
        E = torch.zeros((p, p)).double() 
        x = torch.DoubleTensor(x)
        times = torch.arange(n_times)

        # Create E_ij(t) (note that indices start at 0) 
        for i in range(p): 
            for j in range(p): 
                E[i, j] = float((i + 1 - int((p) / 2))**(j) / np.math.factorial(j))

        # Compute T
        T = torch.linalg.inv(E)

        # Compute the slices
        slices = []
        for t in times: 
            start = int(t - (p) / 2)
            end = start + p
            if start < 0: 
                slices.append(slice(0, p))
            elif end > n_times:
                slices.append(slice(n_times - (p), n_times))
            else: 
                slices.append(slice(start, end))

        series_slices = torch.stack([x[_slice] for _slice in slices], dim=0)

        # series_slices is (n_times, order + 1, dim)
        # T is ( order+1, order+1)
        generalized_coordinates = series_slices.swapaxes(1, 2) @ T.T

        return generalized_coordinates.swapaxes(1, 2)


    def run(self, 
            y   : torch.Tensor,                     # Observed timeseries with shape (time, dimension) 
            u   : Optional[torch.Tensor] = None,    # Explanatory variables, inputs or prior expectation of causes
            x   : Optional[torch.Tensor] = None,    # Confounds
            nD  : int = 1,                          # Number of D-steps 
            nE  : int = 8,                          # Number of E-steps
            nM  : int = 8,                          # Number of M-steps
            K   : int = 1,                          # Learning rate
            tol : float = np.exp(-4),               # Numerical tolerance
            td  : Optional[float] = None            # Integration time 
            ):
        log = self.logger

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
        F   : torch.Tensor       = torch.zeros(nE)      # Free-energy
        A   : torch.Tensor       = torch.zeros(nE)      # Free-action

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
            u = torch.zeros(nT, M[-1].l)
        elif u.shape[1] != nc: 
            raise ValueError(f'Last dimension of input u ({u.shape}) does not match that of deepest model cause ({M[-1].l})')

        if x is None:
            x = torch.zeros(nT, 0)
        
        Y = DEMInversion.generalized_coordinates(y, n) 
        U = DEMInversion.generalized_coordinates(u, d) if u.shape[-1] > 0 else torch.zeros((nT, d, 0))
        X = DEMInversion.generalized_coordinates(x, d) if x.shape[-1] > 0 else torch.zeros((nT, d, 0))

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
            v0.append(torch.zeros(M[i].l, M[i].l))
            w0.append(torch.zeros(M[i].n, M[i].n))
        V0 = torch.kron(torch.zeros(n,n), torch.block_diag(*v0))
        W0 = torch.kron(torch.zeros(n,n), torch.block_diag(*w0))
        Qp = torch.block_diag(V0, W0)

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
                Q.append(torch.block_diag(torch.kron(iVv, torch.block_diag(*q)), W0))

            # and fixed components (V) 
            # ------------------------
            q    = list(v0)
            q[i] = M[i].V
            Qp  += torch.block_diag(torch.kron(iVv, torch.block_diag(*q)), W0)


            # noise on hidden states (R)
            # --------------------------
            for j in range(len(M[i].R)): 
                q    = list(*w0)
                q[i] = M[i].R[j]
                Q.append(torch.block_diag(V0, torch.kron(iVw, torch.block_diag(*q))))

            # and fixed components (W) 
            # ------------------------
            q    = list(w0)
            q[i] = M[i].W
            Qp  += torch.block_diag(V0, torch.kron(iVw, torch.block_diag(*q)))

        # number of hyperparameters
        # -------------------------
        nh : int = len(Q)  

        # fixed priors on states (u) 
        # --------------------------
        xP              =   torch.block_diag(*(M[i].xP for i in range(nl)))
        Px              =   torch.kron(DEMInversion.generalized_covariance(n, 0), xP)
        Pv              =   torch.zeros(n*nv,n*nv)
        Pv[:d*nv,:d*nv] =   torch.kron(DEMInversion.generalized_covariance(d, 0), torch.zeros(nv, nv))
        Pu              =   torch.block_diag(Px, Pv)
        Pu[:nu,:nu]    +=   torch.eye(nu, nu) * nu * np.finfo(np.float64).eps

        iqu             =   dotdict()

        # hyperpriors 
        # -----------
        hgE   = list(chain((M[i].hE for i in range(nl)), (M[i].gE for i in range(nl))))
        hgC   = chain((M[i].hC for i in range(nl)), (M[i].gC for i in range(nl)))
        ph.h  = torch.cat(hgE)           # prior expectation on h
        ph.c  = torch.block_diag(*hgC)    # prior covariance on h
        qh.h  = ph.h                     # conditional expecatation 
        qh.c  = ph.c                     # conditional covariance
        ph.ic = torch.linalg.pinv(ph.c)  # prior precision      

        # priors on parameters (in reduced parameter space)
        # =================================================
        pp.c = list() 
        qp.p = list() 
        qp.u = list() 
        for i in range(nl - 1): 
            # eigenvector reduction: p <- pE + qp.u * qp.p 
            # --------------------------------------------
            Ui      = torch.linalg.svd(M[i].pC, full_matrices=False)[0]
            M[i].p  = Ui.shape[1]               # number of qp.p

            qp.u.append(Ui)                     # basis for parameters (U from SVD's USV')
            qp.p.append(torch.zeros((M[i].p, 1)))    # initial qp.p 
            pp.c.append(Ui.T @ M[i].pC @ Ui)    # prior covariance

        Up = torch.block_diag(*qp.u)

        # initialize and augment with confound parameters B; with flat priors
        # -------------------------------------------------------------------
        nP    = sum(M[i].p for i in range(nl - 1))  # number of model parameters
        nb    = x.shape[-1]                     # number of confounds
        nn    = nb * ny                         # number of nuisance parameters
        nf    = nP + nn                         # number of free parameters
        ip    = slice(0,nP)
        ib    = slice(nP,nP + nn)
        pp.c  = torch.block_diag(*pp.c)
        pp.ic = torch.linalg.inv(pp.c)
        pp.p  = torch.cat(qp.p)

        # initialize conditional density q(p) := qp.e (for D-step)
        # --------------------------------------------------------
        qp.e  = list()
        for i in range(nl - 1): 
            try: 
                qp.e.append(qp.p[i] + qp.u[i].T @ (M[i].P - M[i].pE))
            except KeyError: 
                qp.e.append(qp.p[i])
        qp.e  = torch.cat(qp.e)
        qp.c  = torch.zeros(nf, nf)
        qp.b  = torch.zeros(ny, nb)

        # initialize dedb
        # ---------------
        # NotImplemented

        # initialize arrays for D-step
        # ============================
        qu.x = torch.zeros(n, nx)
        qu.v = torch.zeros(n, nv)
        qu.y = torch.zeros(n, ny)
        qu.u = torch.zeros(n, nc)

        # initialize arrays for hierarchical structure of x[0] and v[0]
        qu.x[0, :] = torch.cat([M[i].x for i in range(0, nl-1)], axis=0).squeeze()
        qu.v[0, :] = torch.cat([M[i].v for i in range(1,   nl)], axis=0).squeeze()

        print('first:', qu.x)

        # derivatives for Jacobian of D-step 
        # ----------------------------------
        Dx              = torch.kron(torch.diag(torch.ones(n-1), 1), torch.eye(nx))
        Dv              = torch.kron(             torch.zeros(n, n), torch.eye(nv))
        Dv[:nv*d,:nv*d] = torch.kron(torch.diag(torch.ones(d-1), 1), torch.eye(nv))
        Dy              = torch.kron(torch.diag(torch.ones(n-1), 1), torch.eye(ny))
        Dc              = torch.kron(             torch.zeros(n, n), torch.eye(nv))
        Dc[:nc*d,:nc*d] = torch.kron(torch.diag(torch.ones(d-1), 1), torch.eye(nc))
        D               = torch.block_diag(Dx, Dv, Dy, Dc)

        # and null blocks
        # ---------------
        dVdy  = torch.zeros(n * ny, 1)
        dVdc  = torch.zeros(n * nc, 1)
        dVdyy = torch.zeros(n * ny, n * ny)
        dVdcc = torch.zeros(n * nc, n * nc)

        # gradients and curvatures for conditional uncertainty
        # ----------------------------------------------------
        dWdu  = torch.zeros(nu, 1)
        dWdp  = torch.zeros(nf, 1)
        dWduu = torch.zeros(nu, nu)
        dWdpp = torch.zeros(nf, nf)

        # preclude unnecessary iterations
        # -------------------------------
        if nh == 0: nM = 1
        if nf == 0 and nh == 0: nE = 1

        # preclude very precise states from entering free-energy/action
        # -------------------------------------------------------------
        ix = slice(ny*n + nv*n,ny*n + nv*n + nx*n)
        iv = slice(ny*n,ny*n + nv*d)
        je = torch.diag(Qp) < np.exp(16)
        ju = torch.cat([je[ix], je[iv]])


        # E-step: (with) embedded D- and M-steps) 
        # =======================================
        Fi = - np.inf
        for iE in range(nE): 
            # [re-]set accumulators for E-step 
            # --------------------------------
            dFdh  = torch.zeros(nh, 1)
            dFdhh = torch.zeros(nh, nh)
            dFdp  = torch.zeros(nf, 1)
            dFdpp = torch.zeros(nf, nf)
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
            for iT in range(nT): 

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
                    qu.y  = Y[iT].clone()
                    qu.u  = U[iT].clone()

                    # compute dEdb (derivatives of confounds)
                    # NotImplemented 

                    # evaluatefunction: 
                    # E = v - g(x,v) and derivatives dE.dx
                    # ====================================
                    E, dE = dem_eval_err_diff(n, d, M, qu, qp)
                    print('dEdu: ', dE.du)

                    log.debug(f'E: {pformat(E)}')

                    # conditional covariance [of states u]
                    # ------------------------------------
                    qu.p         = dE.du.T @ iS @ dE.du + Pu
                    quc          = torch.zeros((nv+nx)*n, (nv+nx)*n)
                    quc[:nu,:nu] = torch.diag(ju.double()) @ torch.linalg.inv(qu.p[:nu,:nu]) @ torch.diag(ju.double()) 
                    qu.c         = quc
                    iqu.c        = iqu.c + torch.logdet(qu.c[:nu,:nu])


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
                            qU.append(dotdict(**qu)) 
                        else: 
                            qE[iT] = E.squeeze(1)
                            qU[iT] = dotdict(**qu) 

                    # uncertainty about parameters dWdv, ...
                    if nP > 0: 
                        CJp   = torch.zeros(iS.shape[0] * nP, (nx + nv) * n)
                        dEdpu = torch.zeros(iS.shape[0] * nP, (nx + nv) * n)
                        for i in range((nx + nv) * n): 
                            CJp[:, i]   = (qp.c[ip,ip] @ dE.dpu[i].T @ iS).reshape((-1,))
                            dEdpu[:, i] = (dE.dpu[i].T).reshape((-1,))

                        dWdu  = CJp.T @ (dE.dp.T).reshape((-1,1))
                        dWduu = CJp.T @ dEdpu


                    # D-step update: of causes v[i] and hidden states x[i]
                    # ====================================================

                    # conditional modes
                    # -----------------
                    u = torch.cat([qu.x, qu.v, qu.y, qu.u], dim=1).reshape((-1,1))

                    # first-order derivatives
                    dVdu    = - dE.du.T @ iS @ E - dWdu/2 - Pu @ u[0:(nx+nv)*n]
                    
                    # second-order derivatives
                    dVduu   = - dE.du.T @ iS @ dE.du - dWduu / 2 - Pu
                    dVduy   = - dE.du.T @ iS @ dE.dy 
                    dVduc   = - dE.du.T @ iS @ dE.dc
                    
                    # gradient
                    dFdu = torch.cat([dVdu.reshape((-1,)), dVdy.reshape((-1,)), dVdc.reshape((-1,))], dim=0)

                    # Jacobian (variational flow)
                    dFduu = torch.Tensor(block_matrix([[dVduu, dVduy, dVduc],
                                                       [   [], dVdyy,    []],
                                                       [   [],    [], dVdcc]]))

                    # update conditional modes of states
                    f     = K * dFdu.unsqueeze(-1)  + D @ u
                    dfdu  = K * dFduu + D
                    du    = compute_dx(f, dfdu, td)
                    q     = u + du

                    qu.x = q[:n * nx].reshape((n, nx))
                    qu.v = q[n * nx:n * (nx + nv)].reshape((n, nv))

                    # ... 

                # Gradients and curvatures for E-step 

                if nP > 0: 
                    CJu     = torch.zeros((nx + nv) * n * iS.shape[0], nP)
                    dEdup   = torch.zeros((nx + nv) * n * iS.shape[0], nP)
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
            mh = torch.zeros(nh)
            for iM in range(nM): 
                # [re-]set precisions using ReML hyperparameter estimates
                iS    = Qp + sum(Q[i] * np.exp(qh.h[i]) for i in range(nh))
                S     = torch.linalg.inv(iS)
                dS    = ECE + EE - S * nT 

                # 1st order derivatives 
                for i in range(nh): 
                    dPdh[i] = Q[i] * np.exp(qh.h[i])
                    dFdh[i] = - torch.trace(dPdh[i] * dS) / 2

                # 2nd order derivatives 
                for i in range(nh): 
                    for j in range(nh): 
                        dFdhh[i, j] = - torch.trace(dPdh[i] * S * dPdh[j] * S * nT)/ 2

                # hyperpriors
                qh.e        = qh.h - ph.h
                if nh > 0:
                    dFdh[:, :]  = dFdh - ph.ic * qh.e
                    dFdhh[:,:]  = dFdhh - ph.ic

                    # update ReML extimate of parameters
                    dh = compute_dx(dFdh, dFdhh, tm, isreg=True)

                    dh   = torch.clamp(dh, -2, 2)
                    qh.h = qh.h + dh 
                    mh   = mh + dh

                # conditional covariance of hyperparameters 
                qh.c = torch.linalg.inv(dFdhh)

                # convergence (M-step)
                if nh > 0 and (((dFdh.T @ dh).squeeze() < tol) or torch.linalg.norm(dh, 1) < tol): 
                    break

            # conditional precision of parameters
            # -----------------------------------
            qp.ic[ip, ip] = qp.ic[ip, ip] + pp.ic
            qp.c = torch.linalg.inv(qp.ic)

            # evaluate objective function F
            # =============================

            # free-energy and action 
            # ----------------------
            Lu = - torch.trace(iS[je][:, je] @ EE[je][:, je]) / 2 \
                 - n * ny * np.log(2 * np.pi) * nT / 2\
                 + torch.logdet(iS[je][:, je]) * nT / 2\
                 + iqu.c / (2*nD)

            # Lu = - torch.trace(iS[je, je] * EE[je, je]) / 2 \
            #      - n * ny * np.log(2 * np.pi) * nT / 2\
            #      + torch.logdet(iS[je, je]) * nT / 2\
            #      + iqu.c / (2*nD)

            Lp = - torch.trace(qp.e.T @ pp.ic @ qp.e) / 2\
                 - torch.trace(qh.e.T @ ph.ic @ qh.e) / 2\
                 + torch.logdet(qp.c[ip][:, ip] @ pp.ic) / 2\
                 + torch.logdet(qh.c @ ph.ic) / 2

            La = - torch.trace(qp.e.T @ pp.ic @ qp.e) * nT / 2\
                 - torch.trace(qh.e.T @ ph.ic @ qh.e) * nT / 2\
                 + torch.logdet(qp.c[ip][:, ip] @ pp.ic * nT) * nT / 2\
                 + torch.logdet(qh.c @ ph.ic * nT) * nT / 2


            print(iqu.c)
            print(torch.logdet(iS[je][:, je]))
            print(torch.trace(iS[je][:, je] @ EE[je][:, je]))
            Li = Lu + Lp
            Ai = Lu + La 
            print('Fi: ', Fi)
            print('Li: ', Li)
            print('Ai: ', Ai)

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
                
                # E-step: update expectation (p)
                # ==============================
                
                # gradients and curvatures
                # ------------------------
                dFdp[ip]         = dFdp[ip]         - pp.ic @ (qp.e - pp.p)
                dFdpp[ip][:, ip] = dFdpp[ip][:, ip] - pp.ic
                
                # update conditional expectation
                # ------------------------------
                dp      = compute_dx(dFdp,dFdpp,te, isreg=True) 
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
                dp      = torch.zeros_like(dp)
                
            
            F[iE]  = Fi;
            A[iE]  = Ai;


            print('mh: ', mh)
            print('dp: ', torch.linalg.norm(dp.reshape((-1,)), 1))
            print('qp: ', torch.linalg.norm(torch.cat(qp.p).reshape((-1,)), 1) )

            # Check convergence 
            if torch.linalg.norm(dp.reshape((-1,)), 1) <= tol * torch.linalg.norm(torch.cat(qp.p).reshape((-1,)), 1) and torch.linalg.norm(mh.reshape((-1,)), 1) <= tol: 
                break 
            if te < -8: 
                break

        results    = dotdict()
        results.F  = F
        results.A  = A

        qH.h       = qh.h
        qH.C       = qh.c

        results.qH = qH

        qP.P       = Up @ qp.e + torch.cat([m.pE for m in M])
        qP.C       = Up @ qp.c[ip][:, ip] @ Up.T
        qP.dFdp    = Up @ dFdp[ip]
        qP.dFdpp   = Up @ dFdpp[ip][:, ip] @ Up.T

        results.qP = qP

        results.qU = dotdict({k: torch.stack([qU[i][k] for i in range(len(qU))], dim=0) for k in qU[0].keys()})
        results.qE = qE

        return results