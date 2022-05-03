import torch

from .dem_structs import *

class GaussianModel(dotdict): 
    def __init__(self, 
        f=None, g=None, m=None, n=None, l=None, x=None, v=None, 
        pE=None, pC=None, hE=None, hC=None, gE=None, gC=None, 
        Q=None, R=None, V=None, W=None, xP=None, vP=None, sv=None, sw=None): 
        self.f  : Callable           = f  # forward function
        self.g  : Callable           = g  # observation function

        self.m  : int                = m  # number of inputs
        self.n  : int                = n  # number of states
        self.l  : int                = l  # number of outputs

        self.x  : torch.Tensor       = x  # explicitly specified states
        self.v  : torch.Tensor       = v  # explicitly specified inputs

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
        self.vP : torch.Tensor       = vP # precision (inputs)

        self.sv : torch.Tensor       = sv # smoothness (input noise)
        self.sw : torch.Tensor       = sw # smoothness (state noise)

class HierarchicalGaussianModel(list): 
    def __init__(self, *models: GaussianModel): 
        models = HierarchicalGaussianModel.prepare_models(*models)
        super().__init__(models)

    @staticmethod
    def prepare_models(*models):
        M = list(models)

        # order 
        g   = len(M)

        # check supra-ordinate level and add one (with flat priors) if necessary
        if callable(M[-1].g): 
            M.append(GaussianModel(l=M[-1].m))
            g = len(M)
        M[-1].n = 0
        M[-1].m = 0

        for i in range(g): 
            # check for hidden states
            if M[i].f is not None and M[i].n is None and M[i].x is None:
                raise ValueError('please specify hidden states or their number')

            # default fields for static models (hidden states)
            if not callable(M[i].f): 
                M[i].f = lambda *x: torch.zeros((0,1))
                M[i].x = torch.zeros((0,1))
                M[i].n = 0

            # consistency and format check on states, parameters and functions
            # ================================================================

            # prior expectation of parameters pE
            # ----------------------------------
            if   M[i].pE is None: 
                 M[i].pE = torch.zeros((0,1))
            elif M[i].pE.shape == 1: 
                 M[i].pE.unsqueeze(1)

            p = M[i].pE.shape[0]

            # and prior covariances pC
            if  M[i].pC is None:
                M[i].pC = torch.zeros((p, p))

            # convert variance to covariance
            elif not hasattr(M[i].pC, 'shape') or M[i].pC.shape == 0:  
                M[i].pC = torch.eye(p) * M[i].pC

            # convert variances to covariance
            elif M[i].pC.shape == 1: 
                M[i].pC = torch.diag(M[i].pC)

            # check size
            if M[i].pC.shape[0] != p or M[i].pC.shape[1] != p: 
                raise ValueError(f'Wrong shape for model[{i}].pC: expected ({p},{p}) but got {M[i].pC.shape}.')

        # get inputs
        v = torch.zeros((0,0)) if M[-1].v is None else M[-1].v
        if sum(v.shape) == 0:
            if M[-2].m is not None: 
                v = torch.zeros((M[-2].m, 1))
            elif M[-1].l is not None: 
                v = torch.zeros((M[-1].l, 1))
        M[-1].l  = v.shape[0]
        M[-1].v  = v

        # check functions
        for i in reversed(range(g - 1)):
            x   = torch.zeros((M[i].n, 1)) if M[i].x is None else M[i].x
            if sum(x.shape) == 0 and M[i].n > 0:
                x = torch.zeros(M[i].n, 1)

            # check function f(x, v, P)
            if not callable(M[i].f): 
                raise ValueError(f"Not callable function: model[{i}].f!")
            try: 
                f = M[i].f(x, v, M[i].pE)
            except: 
                raise ValueError(f"Error while calling function: model[{i}].f")
            if f.shape != x.shape:
                raise ValueError(f"Wrong shape for output of model[{i}].f (expected {x.shape}, got {f.shape}).")

            # check function g(x, v, P)
            if not callable(M[i].g): 
                raise ValueError(f"Not callable function for model[{i}].g!")
            if M[i].m is not None and M[i].m != v.shape[0]:
                warnings.warn(f'Declared input shape of model {i} ({M[i].m}) '
                    f'does not match output shape of model[{i+1}].g ({v.shape[0]})!')
            M[i].m = v.shape[0]
            try: 
                v      = M[i].g(x, v, M[i].pE)
            except: 
                raise ValueError(f"Error while calling function: model[{i}].f")
            if M[i].l is not None and M[i].l != v.shape[0]:
                warnings.warn(f'Declared output shape of model {i} ({M[i].l}) '
                    f'does not match output of model[{i}].g ({v.shape[0]})!')

            M[i].l = v.shape[0]
            M[i].n = x.shape[0]

            M[i].v = v
            M[i].x = x

        # full priors on states
        for i in range(g): 

            # hidden states
            M[i].xP = torch.Tensor() if M[i].xP is None else M[i].xP
            if sum(M[i].xP.shape) == 1: 
                M[i].xP = torch.eye(M[i].n, M[i].n) * xP.squeeze()
            elif len(M[i].xP.shape) == 1 and M[i].xP.shape[0] == M[i].n: 
                M[i].xP = torch.diag(M[i].xP)
            else: 
                M[i].xP = torch.zeros((M[i].n, M[i].n))

            # hidden causes
            M[i].vP = torch.Tensor() if M[i].vP is None else M[i].vP
            if sum(M[i].vP.shape) == 1: 
                M[i].vP = torch.eye(M[i].n, M[i].n) * vP.squeeze()
            elif len(M[i].vP.shape) == 1 and M[i].vP.shape[0] == M[i].n: 
                M[i].vP = torch.diag(M[i].vP)
            else: 
                M[i].vP = torch.zeros((M[i].n, M[i].n))

        nx = sum(M[i].n for i in range(g))

        # Hyperparameters and components (causes: Q V and hidden states R, W)
        # ===================================================================

        # check hyperpriors hE - [log]hyper-parameters and components
        # -----------------------------------------------------------
        pP = 1
        for i in range(g):
            M[i].Q = []
            M[i].R = []
            M[i].V = torch.Tensor()
            M[i].W = torch.Tensor()

            # check hyperpriors (expectation)
            M[i].hE = torch.zeros((len(M[i].Q), 1)) if M[i].hE is None or sum(M[i].hE.shape) == 0 else M[i].hE
            M[i].gE = torch.zeros((len(M[i].R), 1)) if M[i].gE is None or sum(M[i].gE.shape) == 0 else M[i].gE

            #  check hyperpriors (covariances)
            try:
                M[i].hC * M[i].hE
            except: 
                M[i].hC = torch.eye(len(M[i].hE)) / pP 
            try:
                M[i].gC * M[i].gE
            except: 
                M[i].gC = torch.eye(len(M[i].gE)) / pP 

            # check Q and R (precision components)

            # check components and assume iid if not specified
            if len(M[i].Q) > len(M[i].hE): 
                M[i].hE = torch.zeros((M[i].Q), 1) + M[i].hE[1]
            elif len(M[i].Q) < len(M[i].hE): 
                M[i].Q  = [torch.eye(M[i].l)]
                M[i].hE = M[i].hE[1]

            if len(M[i].hE) > len(M[i].hC): 
                M[i].hC = torch.eye(len(M[i].Q)) * M[i].hC[1]
            
            if len(M[i].R) > len(M[i].gE): 
                M[i].gE = torch.zeros(len(M[i].R), 1)
            elif len(M[i].R) < len(M[i].gE): 
                M[i].R = [torch.eye(M[i].n)]
                M[i].gE = M[i].gE[1]
            
            if len(M[i].gE) > len(M[i].gC): 
                M[i].gC = torch.eye(len(M[i].R)) * M[i].gC[1]

            # check consistency and sizes (Q)
            # -------------------------------
            for j in range(len(M[i].Q)):
                if len(M[i].Q[j]) != M[i].l: 
                    raise ValueError(f"Wrong shape for model[{i}].Q[{i}]"
                                     f"(expected ({M[i].l},{M[i].l}), got {M[i].Q[j].shape})")
            
            # check consistency and sizes (R)
            # -------------------------------
            for j in range(len(M[i].R)):
                if len(M[i].R[j]) != M[i].n: 
                    raise ValueError(f"Wrong shape for model[{i}].R[{i}]"
                                     f"(expected ({M[i].n},{M[i].n}), got {M[i].R[j].shape})")
            
            # check V and W (lower bound on precisions)
            # -----------------------------------------
            if len(M[i].V.shape) == 1 and len(M[i].V) == M[i].l: 
                M[i].V = torch.diag(M[i].V)
            elif len(M[i].V) != M[i].l:
                try: 
                    M[i].V = torch.eye(M[i].l) * M[i].V[0]   
                except:
                    if len(M[i].hE) == 0:
                        M[i].V = torch.eye(M[i].l)
                    else: 
                        M[i].V = torch.zeros((M[i].l, M[i].l))

            if len(M[i].W.shape) == 1 and len(M[i].W) == M[i].n: 
                M[i].W = torch.diag(M[i].W)
            elif len(M[i].W) != M[i].n:
                try: 
                    M[i].W = torch.eye(M[i].n) * M[i].W[0]
                except:
                    if len(M[i].gE) == 0:
                        M[i].W = torch.eye(M[i].n)
                    else: 
                        M[i].W = torch.zeros((M[i].n,M[i].n))

            # check smoothness parameter
            s = 0 if nx == 0 else 1/2.
            M[i].sv = s if M[i].sv is None else M[i].sv
            M[i].sw = s if M[i].sw is None else M[i].sw

        return M