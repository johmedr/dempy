import torch
from .transformations import *
from typing import Dict, Optional, List, Callable, Tuple
import abc
import numpy as np
from itertools import chain

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def block_matrix(nested_lists): 
    # a is a list of list [[[], [], tensor], [[], tensor, []]]
    # row and column size must be similar
    sizes = np.zeros((len(nested_lists), len(nested_lists[0]), 2))
    for i, row in enumerate(nested_lists): 
        for j, e in enumerate(row): 
            sizes[i, j, :] = e.shape if len(e) > 0 else (0, 0)
    row_sizes = sizes.max(1)[:, 0].astype(int)
    col_sizes = sizes.max(0)[:, 1].astype(int)
    
    arr = []
    for i, row in enumerate(nested_lists): 
        arr_row = []
        for j, e in enumerate(row): 
            if len(e) > 0: 
                arr_row.append(e)
            else: 
                arr_row.append(np.zeros((row_sizes[i], col_sizes[j])))
        print([_.shape for _ in arr_row])
        arr.append(np.concatenate(arr_row, axis=1))
    return np.concatenate(arr, axis=0)


class GaussianModel(dotdict): 
    def __init__(self, f, g, m, n, l, pE, pC, hE, hC, gE, gC, Q, R, V, W, xP): 
        self.f  : Callable           = f  # forward function
        self.g  : Callable           = g  # observation function

        self.m  : int                = m  # number of inputs
        self.n  : int                = n  # number of states
        self.l  : int                = l  # number of outputs

        self.pE : torch.Tensor       = pE # prior expectation of parameters p
        self.pC : torch.Tensor       = pC # prior covariance of parameters p
        self.hE : torch.Tensor       = hE # prior expectation of hyperparameters h (log-precision of cause noise)
        self.hC : torch.Tensor       = hC # prior covariance of hyperparameters h (log-precision of cause noise)
        self.gE : torch.Tensor       = gE # prior expectation of hyperparameters g (log-precision of state noise)
        self.gC : torch.Tensor       = gC # prior covariance of hyperparameters g (log-precision of state noise)

        self.Q  : List[torch.Tensor] = Q  # precision components (input noise)
        self.R  : List[torch.Tensor] = R  # precision components (state noise)
        self.V  : torch.Tensor       = V  # fixed precision (input noise)
        self.W  : torch.Tensor       = W  # fixed precision (state noise)
        self.xP : torch.Tensor       = xP # precision (states)

        self.sv : torch.Tensor       = sv # smoothness (input noise)
        self.sw : torch.Tensor       = sw # smoothness (state noise)




class DEMInversion: 
    def __init__(self, 
                 systems: List[GaussianModel], 
                 states_embedding_order: int, 
                 causes_embedding_order: int): 
        DEMInversion.check_systems(systems)

        self.M  : List = systems                             # systems, from upper-most to lower-most
        self.n  : int  = states_embedding_order              # embedding order of states
        self.d  : int  = causes_embedding_order              # embedding order of causes
        self.nl : int  = len(systems)                        # number of levels
        self.nv : int  = sum(M.m for M in self.M)            # number of v (causal states)
        self.nx : int  = sum(M.n for M in self.M)            # number of x (hidden states)
        self.ny : int  = self.M[-1].l                        # number of y (model output)
        self.nc : int  = self.M[0].l                         # number of c (prior causes)
        self.nu : int  = self.d * self.nv + self.n * self.nx # number of generalized states

    @staticmethod
    def check_systems(systems):
        """ Checks for output/input mismatch """ 
        if len(systems) ==  1:
            return
        elif len(systems) == 0:
            raise ValueError("Empty list of systems")

        for k in range(1, len(systems)):
            if not systems[k-1].l == systems[k].m:
                raise ValueError(f"Dimension mismatch between index {k}.obs_dim "
                                 f"({systems[k-1].l}) and index {k+1}.input_dim ({systems[k-1].m}) "
                                 f"(indexing from 1)")

    @staticmethod
    def generalized_covariance(
        p    : int,           # derivative order  
        s    : float,         # roughness of the noise process
        prec : bool = False   # whether to return the precision matrix 
        ):
        """ Mimics the behavior of spm_DEM_R.m
        s is the roughtness of the noise process. 
        """
        k = torch.arange(p + 1)
        x = np.sqrt(2) * s
        r = np.cumprod(1 - 2 * k) / (x**(2*k))
        S = torch.zeros((p+1,p+1))
        for i in range(p + 1): 
            j = 2 * k - i
            filt = torch.logical_and(j >= 0, j < p + 1)
            S[i,j[filt]] = (-1) ** (i) * r[filt]

        if prec:
            R = torch.linalg.inv(S)
            return S, R
        else: return S

    @staticmethod
    def generalized_coordinates(
        x    : torch.Tensor,    # timeseries
        p    : int,             # embedding order
        ):
        """ series: torch tensor (n_batchs, n_times, dim) 
            inspired from spm_DEM_embed.m
            series_generalized = T * [series[t-order/2], dots, series[t+order/2]]
        """
        n_batchs, n_times, dim = series.shape
        E = torch.zeros((p + 1, p + 1))
        times = torch.arange(n_times)

        # Create E_ij(t) (note that indices start at 0) 
        for i in range(p + 1): 
            for j in range(p + 1): 
                E[i, j] = (i + 1 - int((p + 1) / 2))**(j) / np.math.factorial(j) 

        # Compute T
        T = torch.linalg.inv(E)

        # Compute the slices
        slices = []
        for t in times: 
            start = int(t - (p + 1) / 2)
            end = start + p + 1
            if start < 0: 
                slices.append(slice(0, p + 1))
            elif end > n_times:
                slices.append(slice(n_times - (p + 1), n_times))
            else: 
                slices.append(slice(start, end))

        series_slices = torch.stack([series[:, _slice] for _slice in slices], dim=0)

        # series_slices is (n_times, order + 1, dim)
        # T is ( order+1, order+1)
        generalized_coordinates = torch.einsum('ilj,kl->ijk', series_slices, T)

        return generalized_coordinates


    def run(self, 
            y   : torch.Tensor,         # Observed timeseries with shape (time, dimension) 
            u   : torch.Tensor,         # Explanatory variables, inputs or prior expectation of causes
            x   : torch.Tensor,         # Confounds
            nD  : int = 1,              # Number of D-steps 
            nE  : int = 8,              # Number of E-steps
            nM  : int = 8,              # Number of M-steps
            K   : int = 1,              # Learning rate
            tol : float = np.exp(-4)    # Numerical tolerance
            ):

        # miscellanous variables
        # ----------------------
        M  : List[GaussianModel] = self.M        # systems, from upper-most (index 0) to lower-most (index -1)
        n  : int                 = self.n        # embedding order of states
        d  : int                 = self.d        # embedding order of causes
        nl : int                 = self.nl       # number of levels
        nv : int                 = self.nv       # number of v (causal states)
        nx : int                 = self.nx       # number of x (hidden states)
        ny : int                 = self.nc       # number of y (model output)
        nc : int                 = self.nc       # number of c (prior causes)
        nu : int                 = self.nu       # number of generalized states
        nT : int                 = y.shape[1]    # number of timesteps

        if ny != y.shape[2]: raise ValueError('Output dimension mismatch.')

        # conditional moments
        # -------------------
        qU : List[dotdict]       = list()     # conditional moments of model states - q(u) for each time t
        qP : dotdict             = dotdict()  # conditional moments of model parameters - q(p)
        qH : dotdict             = dotdict()  # conditional moments of model hyperparameters - q(h)
        qu : dotdict             = dotdict()  # loop variable for qU
        qp : dotdict             = dotdict()  # loop variable for qP
        qh : dotdict             = dotdict()  # loop variable for qH

        # prior moments
        # -------------
        pu     = dotdict()  # prior moments of model states - p(u)
        pp     = dotdict()  # prior moments of model parameters - p(p)
        ph     = dotdict()  # prior moments of model hyperparameters - p(h)

        # embedded series
        # ---------------
        U = DEMInversion.generalized_coordinates(u, d)
        Y = DEMInversion.generalized_coordinates(y, n) 
        X = DEMInversion.generalized_coordinates(x, n)

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
                Q.append(torch.block_diag(torch.kron(iVv, torch.block_diag(q)), W0))

            # and fixed components (V) 
            # ------------------------
            q    = [*v0]
            q[i] = M[i].V
            Qp  += torch.block_diag(torch.kron(iVv, torch.block_diag(q)), W0)


            # noise on hidden states (R)
            # --------------------------
            for j in range(len(M[i].R)): 
                q    = list(*w0)
                q[i] = M[i].R[j]
                Q.append(torch.block_diag(V0, torch.kron(iVw, torch.block_diag(q))))

            # and fixed components (W) 
            # ------------------------
            q    = [*w0]
            q[i] = M[i].W
            Qp  += torch.block_diag(V0, torch.kron(iVw, torch.block_diag(q)))

        # number of hyperparameters
        # -------------------------
        nh : int = len(Q)  

        # fixed priors on states (u) 
        # --------------------------
        xP  =   torch.block_diag(M[i].xP for i in range(nl))
        Px  =   torch.kron(DEMInversion.generalized_covariance(n, 0), xP)
        Pv  =   torch.kron(DEMInversion.generalized_covariance(d, 0), torch.zeros(nv, nv))
        Pu  =   torch.block_diag(Px, Pv)
        Pu +=   torch.eye(nu, nu) * nu * np.finfo(np.float64).eps

        # hyperpriors 
        # -----------
        hgE   = list(chain(M[i].hE for i in range(nl), M[i].gE for i in range(nl)))
        hgC   = chain(M[i].hC for i in range(nl), M[i].gC for i in range(nl))
        ph.h  = torch.cat(hgE)           # prior expectation on h
        ph.c  = torch.block_diag(hgC)    # prior covariance on h
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
            qp.p.append(torch.zeros(M[i].p))    # initial qp.p 
            pp.c.append(Ui.T @ M[i].pC @ Ui)    # prior covariance

        Up = torch.block_diag(qp.u)

        # initialize and augment with confound parameters B; with flat priors
        # -------------------------------------------------------------------
        nP    = sum(M[i].p for i in range(nl))  # number of model parameters
        nb    = x.shape[-1]                     # number of confounds
        nn    = nb * ny                         # number of nuisance parameters
        nf    = np + nn                         # number of free parameters
        pp.c  = torch.block_diag(pp.c)
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

        # initialize arrys for hierarchical structure of x[0] and v[0]


        # derivatives for Jacobian of D-step 
        # ----------------------------------
        Dx  = torch.kron(torch.diag(torch.ones(n-1), 1), torch.eye(nx))
        Dv  = torch.kron(torch.diag(torch.ones(d-1), 1), torch.eye(nv))
        Dy  = torch.kron(torch.diag(torch.ones(n-1), 1), torch.eye(ny))
        Dc  = torch.kron(torch.diag(torch.ones(d-1), 1), torch.eye(nc))
        D   = torch.block_diag(Dx, Dv, Dy, Dc)

        # and null blocks
        # ---------------
        dVdy  = torch.zeros(n * ny, 1)
        dVdc  = torch.zeros(d * nc, 1)
        dVdyy = torch.zeros(n * ny, n * ny)
        dVdcc = torch.zeros(d * nc, d * nc)

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
        # NotImplemented

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
            y     = y - qp.b * x

            # [re-]set states & their derivatives
            # -----------------------------------
            qu = qU[0] if len(qU) > 0 else qu

            # D-step: (nD D-steps for each sample) 
            # ====================================
            for iT in range(nT): 

                # [re-]set states for static systems
                # ----------------------------------
                if not nx:
                    qu = qU[iT] if len(qU) > iT else qu
                
                # D-step: until convergence for static systems
                # ============================================ 
                Fd =  -np.exp(64)
                for iD in range(nD): 

                    # sampling time 
                    # not implemented (diff: sampling time does not change with iD)

                    # derivatives of responses and inputs
                    # -----------------------------------
                    qu.y = Y[iT]
                    qu.u = U[iT]

                    # compute dEdb (derivatives of confounds)
                    # NotImplemented 

                    # evaluatefunction: 
                    # E = v - g(x,v) and derivatives dE.dx
                    # ====================================
                    E,dE : Tuple[torch.Tensor, dotdict] = eval(M, qu, qp)

                    # conditional covariance [of states u]
                    # ------------------------------------
                    qu.p  = dE.du.T @ iS @ dE.du + Pu
                    qu.c  = torch.diag(ju) @ torch.linalg.inv(qu.p) @ torch.diag(ju)
                    iqu.c = iqu.c + torch.logdet(qu.c)

                    # and conditional covariance [of parameters P]
                    # --------------------------------------------
                    dE.dP = dE.dp # no dedb for now
                    ECEu  = dE.du.T @ qu.c @ dE.du
                    ECEp  = dE.dp.T @ qp.c @ dE.dp

                    if nx == 0: 
                        pass 

                    # save states at iT
                    if iD == 0: 
                        qE.append(E)
                        qU.append(qu) 

                    # uncertainty about parameters dWdv, ...
                    if nP > 0: 
                        for i in range(nu): 
                            CJp[:, i]   = qp.c @ dE.dpu[i].T @ iS
                            dedpu[:, i] = dE.dpu[i]
                        dWdu  = CJp.T @ dE.dp.T
                        dWduu = CJp.T @ dE.dpu


                    # D-step update: of causes v[i] and hidden states x[i]
                    # ====================================================

                    # conditional modes
                    # -----------------
                    u = torch.cat([qu.x, qu.v, qu.y, qu.u])

                    # first-order derivatives
                    dVdu    = - dE.du.T @ iS @ E - dWdu/2 - Pu * u[0:nu]
                    
                    # second-order derivatives
                    dVduu   = - dE.du.T @ iS @ dE.du - dWduu / 2 - Pu
                    dVduy   = - dE.du.T @ iS @ dE.dy 
                    dVduc   = - dE.du.T @ iS @ dE.dc

                    # gradient
                    dFdy = torch.cat(dVdu, dVdy, dVdc)

                    # Jacobian (variational flow)
                    dFduu = torch.Tensor(block_matrix([[dVduu, dVduy, dVduc],
                                                       [   [], dVdyy,    []],
                                                       [   [],    [], dVdcc]]))

                    # update conditional modes of states 
                    f     = K * dFdu  + D @ u
                    dfdu  = K * dFduu + D

                    du = ... 
                    q  = u + du

                    qu.x = q[:n]
                    qu.v = q[n:n+d]

                    # ... 

                # Gradients and curvatures for E-step 

                if nP > 0: 
                    for i in range(np): 
                        CJu[:, i]   = qu.c @ dE.dup[i].T @ iS
                        dedup[:, i] = dE.dup[i]
                    dWdp  = CJp.T @ dE.du.T
                    dWdpp = CJp.T @ dE.dpu

                # Accumulate dF/dP = <dL/dp>, dF/dpp = ... 
                dFdp  = dFdp  - dWdp / 2 - dE.dP.T @ iS @ E
                dFdpp = dFdpp - dWdpp /2 - dE.dP.T @ iS @ dE.dP
                qp.ic = qp.ic + dE.dP.T @ iS @ dE.dP

                # and quantities for M-step 
                EE   = E @ E.T + EE
                ECE = ECE + ECEu + ECEp


            # M-step - optimize hyperparameters (mh = total update)
            mh = 0
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
                qh.e = qh.h - ph.h
                dFdh = dFdh - ph.ic * qh.e
                dFdhh = dFdhh - ph.ic

                # update ReML extimate of parameters
                dh = ... 
                dh = max(min(dh, 2), -2)
                qh.h = qh.h + dh 
                mh = mh + dh

                # conditional covariance of hyperparameters 
                qh.c = torch.linalg.inv(dFdhh)

                # convergence (M-step)
                # if (dFdh.T @ dh < TOL) || 

            # conditional precision of parameters
            # -----------------------------------
            qp.ic = qp.ic + pp.ic
            qp.c = torch.linalg.inv(qp.ic)

            # evaluate objective function F
            # =============================

            # free-energy and action 
            # ----------------------
            Lu = - torch.trace(iS[je, je] * EE[je, je]) / 2 \
                 - n * ny * np.log(2 * np.pi) * nT / 2\
                 + torch.logdet(iS[je, je]) * nT / 2\
                 + iqu.c / (2*nD)

            Lp = - torch.trace(qp.e.T @ pp.ic @ qp.e) / 2\
                 - torch.trace(qh.e.T @ ph.ic @ qh.e) / 2\
                 + torch.logdet(qp.c @ pp.ic) / 2\
                 + torch.logdet(qh.c @ ph.ic) / 2

            La = - torch.trace(qp.e.T @ pp.ic @ qp.e) * nT / 2\
                 - torch.trace(qh.e.T @ ph.ic @ qh.e) * nT / 2\
                 + torch.logdet(qp.c @ pp.ic * nT) * nT / 2\
                 + torch.logdet(qh.c @ ph.ic * nT) * nT / 2

            Li = Lu + Lp
            Ai = Lu + La 

            # if F is increasng save expansion point and derivatives 
            if Li > Fi or iE < 2: 
                # Accept free-energy and save current parameter estimates
                #--------------------------------------------------------
                Fi      = Li;
                te      = min(te + 1/2,4);
                tm      = min(tm + 1/2,4);
                B.qp    = qp;
                B.qh    = qh;
                B.pp    = pp;
                
                # E-step: update expectation (p)
                # ==============================
                
                # gradients and curvatures
                # ------------------------
                dFdp(ip)     = dFdp(ip)     - pp.ic*(qp.e - pp.p);
                dFdpp(ip,ip) = dFdpp(ip,ip) - pp.ic;
                
                # update conditional expectation
                # ------------------------------
                dp      = spm_dx(dFdpp,dFdp,{te}); ... 
                qp.e    = qp.e + dp(ip);
                qp.p    = spm_unvec(qp.e,qp.p);
                qp.b    = spm_unvec(dp(ib),qp.b);

            else:
                
                # otherwise, return to previous expansion point
                # ---------------------------------------------
                nM      = 1;
                qp      = B.qp;
                pp      = B.pp;
                qh      = B.qh;
                te      = min(te - 2, -2);
                tm      = min(tm - 2, -2); 
                
            
            F[iE]  = Fi;
            A[iE]   = Ai;



        # # E step 
        # for iE in range(nE): 
        #     # D step
        #     for iT in range(nT): 
        #         update_x()
        #     # M step 
        #     for iM in range(nM): 
        #         update_h()
            
        #     fa[iT] = compute_fa()
        #     if fa[iT] > fa[iT - 1]: 
        #         update_p()

 


# class DEMHierarchicalDynamicalModel:
#     def __init__(self, systems: List[GaussianModel], order=4):
#         DEMHierarchicalDynamicalModel.check_systems(systems)
#         self._systems = systems
#         self._n_systems = len(systems)
#         self._input_dim = systems[0]['input_dim']
#         self._obs_dim = systems[-1]['obs_dim']
#         self._input_mean = input_mean
#         self._input_cov = input_cov
#         self._order = order

#     @staticmethod
#     def check_systems(systems):
#         if len(systems) < 2:
#             return

#         for k in range(1, len(systems)):
#             if not systems[k-1]['obs_dim'] == systems[k]['input_dim']:
#                 raise ValueError(f"Dimension mismatch between index {k}.obs_dim "
#                                  f"({systems[k-1]['obs_dim']}) and index {k+1}.input_dim ({systems[k-1]['obs_dim']}) "
#                                  f"(indexing from 1)")

#     def prepare_data(self, y):
#         """ y: torch tensor ([n_batchs,] n_times, obs_dim)"""
#         if not y.shape[-1] == self._obs_dim:
#             raise ValueError(
#                 f"Dimension of measurements y ({y.shape[-1]} does not match declared obs_dim ({self._obs_dim}).")
#         if len(y.shape) == 2:
#             y = y.unsqueeze(0)
#         elif len(y.shape) != 3:
#             raise ValueError(
#                 f"Expected measurements y to have 2 or 3 dimensions, but got ({len(y.shape)}.")
#         return ydef predef prepare_data(self, y):
#         """ y: torch tensor ([n_batchs,] n_times, obs_dim)"""
#         if not y.shape[-1] == self._obs_dim:
#             raise ValueError(
#                 f"Dimension of measurements y ({y.shape[-1]} does not match declared obs_dim ({self._obs_dim}).")
#         if len(y.shape) == 2:
#             y = y.unsqueeze(0)
#         elif len(y.shape) != 3:
#             raise ValueError(
#                 f"Expected measurements y to have 2 or 3 dimensions, but got ({len(y.shape)}.")
#         return ypare_data(self, y):
#         """ y: torch tensor ([n_batchs,] n_times, obs_dim)"""
#         if not y.shape[-1] == self._obs_dim:
#             raise ValueError(
#                 f"Dimension of measurements y ({y.shape[-1]} does not match declared obs_dim ({self._obs_dim}).")
#         if len(y.shape) == 2:
#             y = y.unsqueeze(0)
#         elif len(y.shape) != 3:
#             raise ValueError(
#                 f"Expected measurements y to have 2 or 3 dimensions, but got ({len(y.shape)}.")
#         return y

#     def to_generalized_coordinates(self, series): 
#         """ series: torch tensor (n_batchs, n_times, dim) 
#             inspired from spm_DEM_embed.m
#             series_generalized = T * [series[t-order/2], dots, series[t+order/2]]
#         """
#         n_batchs, n_times, dim = series.shape
#         p = self._order

#         E = torch.zeros((p + 1, p + 1))
#         times = torch.arange(n_times)

#         # Create E_ij(t) (note that indices start at 0) 
#         for i in range(p + 1): 
#             for j in range(p + 1): 
#                 E[i, j] = (i + 1 - int((p + 1) / 2))**(j) / np.math.factorial(j) 

#         # Compute T
#         T = torch.linalg.inv(E)

#         # Compute the slices
#         slices = []
#         for t in times: 
#             start = int(t - (p + 1) / 2)
#             end = start + p + 1
#             if start < 0: 
#                 slices.append(slice(0, p + 1))
#             elif end > n_times:
#                 slices.append(slice(n_times - (p + 1), n_times))
#             else: 
#                 slices.append(slice(start, end))

#         series_slices = torch.stack([series[:, _slice] for _slice in slices], dim=1)

#         # series_slices is (n_batchs, n_times, order + 1, dim)
#         # T is ( order+1, order+1)
#         generalized_coordinates = torch.einsum('ijkl,mk->ijlm', series_slices, T)


#         return generalized_coordinates

#     def generalized_covariance(self, s):
#         """ s is the roughtness of the noise process 
#         """
#         p = self._order
#         k = torch.arange(p + 1)
#         x = np.sqrt(2) * s
#         r = np.cumprod(1 - 2 * k) / (x**(2*k))
#         S = torch.zeros((p+1,p+1))
#         for i in range(p + 1): 
#             j = 2 * k - i
#             filt = torch.logical_and(j >= 0, j < p + 1)
#             S[i,j[filt]] = (-1) ** (i) * r[filt]
#         R = torch.linalg.inv(S)
#         return S, R

#     def filter(self, y):
#         y = self.prepare_data(y)
#         y = self.generalized_coordinates(y)
#         D = torch.diagonal(torch.ones(self._order), k=1)
#         