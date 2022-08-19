import numpy as np

from .dem_structs import *
from .dem_hgm import *
from .dem_dx import *

def as_matrix_it(*args):
    for arg in args: 
        if arg.size > 0: 
            yield arg.reshape((arg.shape[0], -1)) 
        else: 
            yield arg.reshape((0, 1)) 

def dem_eval_err_diff(n: int, d: int, M: HierarchicalGaussianModel, qu: dotdict, qp: dotdict): 
    # inspired by spm_DEM_eval_diff and other deps., by Karl Friston 

    # Get dimensions
    # ==============
    nl = len(M)
    ne = sum(m.l for m in M) 
    nv = sum(m.m for m in M)
    nx = sum(m.n for m in M)
    nP = sum(M[i].p for i in range(nl - 1))
    ny = M[0].l
    nc = M[-1].l

    # Evaluate functions at each level
    # ================================
    f = []
    g = []
    x = []
    v = []
    nxi = 0
    nvi = 0


    # Evaluate derivatives at each level
    df  = list()
    d2f = list()
    dg  = list()
    d2g = list()
    for i in range(nl - 1):
        xi  = qu.x[0, nxi:nxi + M[i].n]
        vi  = qu.v[0, nvi:nvi + M[i].m]

        nxi = nxi + M[i].n
        nvi = nvi + M[i].m

        x.append(xi)
        v.append(vi)
        
        xi,vi,q,u,p = (_ if sum(_.shape) > 0 else np.empty(0) for _ in  (x[i], v[i], qp.p[i], qp.u[i], M[i].pE))
        puq = p + u @ q

        if M[i].constraints is not None:
            # evaluate at mode 
            puq[M[i].cpos] =   np.exp(M[i].cpE[M[i].cpos] + np.sqrt(M[i].cpC[M[i].cpos]) * puq[M[i].cpos])
            puq[M[i].cneg] = - np.exp(M[i].cpE[M[i].cneg] + np.sqrt(M[i].cpC[M[i].cneg]) * puq[M[i].cneg])

            u[M[i].csel, :] *= puq[M[i].csel] * np.sqrt(M[i].cpC[M[i].csel])[:, None]
         

        xvp = (xi, vi, puq)
        xvp = tuple(as_matrix_it(*xvp))

        try: 
            res = M[i].f(*xvp)
        except: 
            raise RuntimeError(f"Error while evaluating model[{i}].f!")
        f.append(res)

        try: 
            res = M[i].g(*xvp)
        except: 
            raise RuntimeError(f"Error while evaluating model[{i}].g!")
        g.append(res)

        dfi  = M[i].df(*xvp)
        d2fi = M[i].d2f(*xvp)
        dgi  = M[i].dg(*xvp)
        d2gi = M[i].d2g(*xvp) 

        dfi.dp = dfi.dp @ u
        dgi.dp = dgi.dp @ u

        d2fi.dx.dp = d2fi.dx.dp @ u
        d2fi.dv.dp = d2fi.dv.dp @ u
        d2fi.dv.dx = d2fi.dx.dv.swapaxes(1, 2)
        d2fi.dp.dx = d2fi.dx.dp.swapaxes(1, 2)
        d2fi.dp.dv = d2fi.dv.dp.swapaxes(1, 2)

        d2fi.dp.dp = np.einsum('ik,aij,jl->akl', u, d2fi.dp.dp, u)

        d2gi.dx.dp = d2gi.dx.dp @ u
        d2gi.dv.dp = d2gi.dv.dp @ u
        d2gi.dv.dx = d2gi.dx.dv.swapaxes(1, 2)
        d2gi.dp.dx = d2gi.dx.dp.swapaxes(1, 2)
        d2gi.dp.dv = d2gi.dv.dp.swapaxes(1, 2)

        d2gi.dp.dp = np.einsum('ik,aij,jl->akl', u, d2gi.dp.dp, u)

        dg.append(dgi)
        df.append(dfi)

        d2f.append(d2fi)
        d2g.append(d2gi)

    # Stack f's  and g's
    f = np.concatenate(f).reshape((nx,))
    g = np.concatenate(g).reshape((ne - nc,))

    # Setup df
    df = dotdict({k: block_diag(*(dfi[k] for dfi in df)) for k in ['dx', 'dv', 'dp']}) 

    # Setup dgdv manually 
    dgdv = cell(nl, nl-1)
    for i in range(nl - 1): 
        # causes (level i) appear at level i in g(x[i],v[i]) and at level i+1 as -I
        # nb: dg = [dydv[:], dv[0]dv[:], ...]
        dgdv[  i, i] = dg[i].dv 
        dgdv[i+1, i] = -np.eye(M[i].m)

    dgdv = block_matrix(dgdv)

    # Setup dg
    dg = dotdict({k: block_diag(*(dgi[k] for dgi in dg)) for k in ['dx', 'dp']}) 
    # add an extra row to accomodate the highest hierarchical level
    for k in dg.keys():
        dg[k] = np.concatenate([dg[k], np.zeros((nc, dg[k].shape[1]))], axis=0)
    dg.dv = dgdv

    # Reshape df and dg to avoid errors laters
    df.dx = df.dx.reshape((nx, nx))
    df.dv = df.dv.reshape((nx, nv))
    df.dp = df.dp.reshape((nx, nP))
    dg.dx = dg.dx.reshape((ne, nx))
    dg.dv = dg.dv.reshape((ne, nv))
    dg.dp = dg.dp.reshape((ne, nP))

    # Process d2f, d2g
    d2f = dotdict({i: dotdict({j: [d2fk[i][j] for d2fk in d2f] for j in ['dx', 'dv', 'dp']}) for i in ['dx', 'dv', 'dp']})
    d2g = dotdict({i: dotdict({j: [d2gk[i][j] for d2gk in d2g] for j in ['dx', 'dv', 'dp']}) for i in ['dx', 'dv', 'dp']}) 

    # Create 2nd order derivative matrices (hierarchical)
    dfdxp = np.zeros((nP, nx, nx))
    dfdvp = np.zeros((nP, nx, nv))
    dgdxp = np.zeros((nP, ne, nx))
    dgdvp = np.zeros((nP, ne, nv))

    # we use ndarray views to change both dfdxp and dfdpx at the same time
    dfdpx = np.swapaxes(dfdxp, 0, 2)
    dfdpv = np.swapaxes(dfdvp, 0, 2)
    dgdpx = np.swapaxes(dgdxp, 0, 2)
    dgdpv = np.swapaxes(dgdvp, 0, 2)

    if nP > 0:
        ix0, iv0, ie0, ip0 = 0,0,0,0

        for i in range(nl - 1): 
            ix = ix0 + M[i].n
            iv = iv0 + M[i].m
            ie = ie0 + M[i].l
            ip = ip0 + M[i].p 

            dfdxp[ip0:ip, ix0:ix, ix0:ix] = d2f.dp.dx[i].swapaxes(0, 1)
            dfdvp[ip0:ip, ix0:ix, iv0:iv] = d2f.dp.dv[i].swapaxes(0, 1)
            dgdxp[ip0:ip, ie0:ie, ix0:ix] = d2g.dp.dx[i].swapaxes(0, 1)
            dgdvp[ip0:ip, ie0:ie, iv0:iv] = d2g.dp.dv[i].swapaxes(0, 1)

            ix0, iv0, ie0, ip0 = ix, iv, ie, ip

    dfdpu = np.concatenate([dfdpx, dfdpv], axis=0)
    dgdpu = np.concatenate([dgdpx, dgdpv], axis=0)

    de    = dotdict(
      dy  = np.eye(ne, ny), 
      dc  = np.diag(-np.ones(max(ne, nc) - (nc - ne)), nc - ne)[:ne, :nc]
    )

    # Prediction error (E) - causes        
    Ev = [np.concatenate([qu.y[0], qu.v[0]]) -  np.concatenate([g, qu.u[0]])]
    for i in range(1, n):
        Evi = de.dy @ qu.y[i] + de.dc @ qu.u[i] - dg.dx @ qu.x[i] - dg.dv @ qu.v[i]
        Ev.append(Evi)

    # Prediction error (E) - states
    Ex = [qu.x[1] - f]
    for i in range(1, n-1):
        Exi = qu.x[i + 1] - df.dx @ qu.x[i] - df.dv @ qu.v[i]
        Ex.append(Exi)
    Ex.append(np.zeros_like(Exi))

    Ev = np.concatenate(Ev)
    Ex = np.concatenate(Ex)
    E  = np.concatenate([Ev, Ex])[:, None]

    # generalised derivatives
    dgdp = [dg.dp]
    dfdp = [df.dp]

    qux = qu.x[..., None]
    quv = qu.v[..., None]
    for i in range(1, n):
        dgdpi = dg.dp.copy()
        dfdpi = df.dp.copy()

        for ip in range(nP): 
            dgdpi[:, ip] = (dgdxp[ip] @ qux[i] + dgdvp[ip] @ quv[i]).squeeze(1)
            dfdpi[:, ip] = (dfdxp[ip] @ qux[i] + dfdvp[ip] @ quv[i]).squeeze(1)
        
        dfdp.append(dfdpi)
        dgdp.append(dgdpi)

    df.dp = np.concatenate(dfdp)
    dg.dp = np.concatenate(dgdp)

    de.dy               = kron_eye(de.dy, n)
    df.dy               = np.zeros((n*nx, n*ny))# kron(np.eye(n, n), np.zeros((nx, ny)))
    df.dc               = np.zeros((n*nx, n*nc))
    dg.dx               = kron_eye(dg.dx, n)
    # df.dx = (I * df.dx) - D, Eq. 45
    df.dx               = kron_eye(df.dx, n) - kron(np.diag(np.ones(n - 1), 1), np.eye(nx, nx)) 

    # embed to n >= d
    dedc                = np.zeros((n*ne, n*nc)) 
    dedc[:n*ne,:d*nc]   = kron_eye(de.dc, n, d) # kron(np.eye(n, d), 
    de.dc               = dedc

    dgdv                = np.zeros((n*ne, n*nv))
    dgdv[:n*ne,:d*nv]   = kron_eye(dg.dv, n, d)# kron(np.eye(n, d), dg.dv)
    dg.dv               = dgdv

    dfdv                = np.zeros((n*nx, n*nv))
    dfdv[:n*nx,:d*nv]   = kron_eye(df.dv, n, d)#kron(np.eye(n, d), df.dv)
    df.dv               = dfdv

    dE    = dotdict()
    dE.dy = np.concatenate([de.dy, df.dy])
    dE.dc = np.concatenate([de.dc, df.dc])
    dE.dp = - np.concatenate([dg.dp, df.dp])
    dE.du = - block_matrix([
            [dg.dx, dg.dv], 
            [df.dx, df.dv]])

    dE.dup = []
    for ip in range(nP): 
        dfdxpi              = kron_eye(dfdxp[ip], n) # kron(np.eye(n,n), dfdxp[ip])
        dgdxpi              = kron_eye(dgdxp[ip], n) # kron(np.eye(n,n), dgdxp[ip])
        dfdvpi              = np.zeros((n*nx, n*nv))
        dfdvpi[:,:d*nv]     = kron_eye(dfdvp[ip], n, d) # kron(np.eye(n,d), dfdvp[ip])
        dgdvpi              = np.zeros((n*ne, n*nv))
        dgdvpi[:,:d*nv]     = kron_eye(dgdvp[ip], n, d) # kron(np.eye(n,d), dgdvp[ip])

        dEdupi = -block_matrix([[dgdxpi, dgdvpi], [dfdxpi, dfdvpi]])
        dE.dup.append(dEdupi)

    dE.dpu = [] 
    for i in range(n): 
        for iu in range(nx + nv):
            dfdpui = kron_eye(dfdpu[iu], n, 1) # kron(np.eye(n,1), dfdpu[iu])
            dgdpui = kron_eye(dgdpu[iu], n, 1) # kron(np.eye(n,1), dgdpu[iu])
            dEdpui = np.concatenate([dgdpui, dfdpui], axis=0)

            dE.dpu.append(dEdpui)

    return E, dE