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
            puq = puq
            puq[M[i].constraints == 'positive'] = np.maximum(0, np.exp(puq[M[i].constraints == 'positive']) - 1)
            puq[M[i].constraints == 'negative'] = np.minimum(-np.exp(puq[M[i].constraints == 'negative']) + 1,0)

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

    f = np.concatenate(f).reshape((nx,))
    g = np.concatenate(g).reshape((ne - nc,))

    # Evaluate derivatives at each level
    df  = list()
    d2f = list()
    dg  = list()
    d2g = list()
    for i in range(nl - 1): 
        x,v,q,u,p = (_ if sum(_.shape) > 0 else np.empty(0) for _ in  (x[i], v[i], qp.p[i], qp.u[i], M[i].pE))
        puq = p + u @ q

        if M[i].constraints is not None:
            puq = puq
            puq[M[i].constraints == 'positive'] =  np.exp(puq[M[i].constraints == 'positive'])
            puq[M[i].constraints == 'negative'] = -np.exp(puq[M[i].constraints == 'negative'])
            cu  = puq * u 
            puq[M[i].constraints == 'positive'] = np.maximum(0, puq[M[i].constraints == 'positive'] - 1)
            puq[M[i].constraints == 'negative'] = np.minimum(puq[M[i].constraints == 'negative'] + 1,0)
            # puq[M[i].constraints == 'positive'] -= 1
            # puq[M[i].constraints == 'negative'] += 1
        else: 
            cu = u

        xvp = (xi, vi, puq)
        xvp = tuple(as_matrix_it(*xvp))

        if M[i].df is not None:
            dfi  = M[i].df(*xvp)
            d2fi = M[i].d2f(*xvp)
        else: 
            dfi, d2fi = compute_df_d2f(M[i].f, xvp, ['dx', 'dv', 'dp'])

        if M[i].dg is not None:
            dgi  = M[i].dg(*xvp)
            d2gi = M[i].d2g(*xvp) 
        else: 
            dgi, d2gi = compute_df_d2f(M[i].g, xvp, ['dx', 'dv', 'dp']) 



        dfi.dp = dfi.dp @ cu
        dgi.dp = dgi.dp @ cu


        d2fi.dx.dp = d2fi.dx.dp @ cu
        d2fi.dv.dp = d2fi.dv.dp @ cu
        # d2fi.dx.dp = np.einsum('ijk,kl->ijl', d2fi.dx.dp, cu)
        # d2fi.dv.dp = np.einsum('ijk,kl->ijl', d2fi.dv.dp, cu)
        d2fi.dv.dx = d2fi.dx.dv.swapaxes(1, 2)
        d2fi.dp.dx = d2fi.dx.dp.swapaxes(1, 2)
        d2fi.dp.dv = d2fi.dv.dp.swapaxes(1, 2)

        d2fi.dp.dp = np.einsum('ik,aij,jl->akl', cu, d2fi.dp.dp, cu)

        d2gi.dx.dp = d2gi.dx.dp @ cu
        d2gi.dv.dp = d2gi.dv.dp @ cu
        # d2gi.dx.dp = np.einsum('ijk,kl->ijl', d2gi.dx.dp, cu)
        # d2gi.dv.dp = np.einsum('ijk,kl->ijl', d2gi.dv.dp, cu)
        d2gi.dv.dx = d2gi.dx.dv.swapaxes(1, 2)
        d2gi.dp.dx = d2gi.dx.dp.swapaxes(1, 2)
        d2gi.dp.dv = d2gi.dv.dp.swapaxes(1, 2)

        d2gi.dp.dp = np.einsum('ik,aij,jl->akl', cu, d2gi.dp.dp, cu)

        dg.append(dgi)
        df.append(dfi)

        d2f.append(d2fi)
        d2g.append(d2gi)

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


    if nP > 0:
        dfdxp = np.stack([block_diag(*(_[:, ip] for _ in d2f.dp.dx if _.size > 0)) for ip in range(nP)], axis=0)
        dfdvp = np.stack([block_diag(*(_[:, ip] for _ in d2f.dp.dv if _.size > 0)) for ip in range(nP)], axis=0)

        # dgdvp = np.stack([block_diag(*(_[:, ip] for _ in d2g.dp.dv if _.size > 0)) for ip in range(nP)], axis=0)
        # dgdxp = np.stack([block_diag(*(_[:, ip] for _ in d2g.dp.dx if _.size > 0)) for ip in range(nP)], axis=0)

        dgdxp = [block_diag(*(_[:, ip] for _ in d2g.dp.dx if _.size > 0)) for ip in range(nP)]
        dgdvp = [block_diag(*(_[:, ip] for _ in d2g.dp.dv if _.size > 0)) for ip in range(nP)]

        # Add a component with nc rows to accomodate the highest hierarchical level
        dgdxp = np.stack([np.concatenate([dgdxpi, np.zeros((nc, dgdxpi.shape[1]))]) for dgdxpi in dgdxp], axis=0)
        dgdvp = np.stack([np.concatenate([dgdvpi, np.zeros((nc, dgdvpi.shape[1]))]) for dgdvpi in dgdvp], axis=0)

        dfdxp = dfdxp.reshape((nP, nx, nx))
        dfdvp = dfdvp.reshape((nP, nx, nv))
        dgdxp = dgdxp.reshape((nP, ne, nx))
        dgdvp = dgdvp.reshape((nP, ne, nv))
    else: 
        dfdxp = np.empty((nP, nx, nx))
        dfdvp = np.empty((nP, nx, nv))
        dgdxp = np.empty((nP, ne, nx))
        dgdvp = np.empty((nP, ne, nv))

    if nx > 0: 
        dfdpx = np.stack([block_diag(*(_[:, ix] for _ in d2f.dx.dp if _.size > 0)) for ix in range(nx)], axis=0)
        dgdpx = [block_diag(*(_[:, ix] for _ in d2g.dx.dp if _.size > 0)) for ix in range(nx)]

        # Add a component with nc rows to accomodate the highest hierarchical level
        dgdpx = np.stack([np.concatenate([dgdpxi, np.zeros((nc, dgdpxi.shape[1]))]) for dgdpxi in dgdpx], axis=0)

        dfdpx = dfdpx.reshape((nx, nx, nP))
        dgdpx = dgdpx.reshape((nx, ne, nP))
    else: 
        dfdpx = np.empty((nx, nx, nP))
        dgdpx = np.empty((nx, ne, nP))

    if nv > 0: 
        dfdpv = np.stack([block_diag(*(_[:, iv] for _ in d2f.dv.dp if _.size > 0)) for iv in range(nv)], axis=0)
        dgdpv = [block_diag(*(_[:, iv] for _ in d2g.dv.dp if _.size > 0)) for iv in range(nv)]

        # Add a component with nc rows to accomodate the highest hierarchical level
        dgdpv = np.stack([np.concatenate([dgdpvi, np.zeros((nc, dgdpvi.shape[1]))]) for dgdpvi in dgdpv], axis=0)

        dfdpv = dfdpv.reshape((nv, nx, nP))
        dgdpv = dgdpv.reshape((nv, ne, nP))
    else: 
        dfdpv = np.empty((nv, nx, nP))
        dgdpv = np.empty((nv, ne, nP))
    

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