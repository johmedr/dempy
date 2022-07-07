import torch 
from .dem_structs import *
from .dem_hgm import *
from .dem_dx import *
from typing import List
from pprint import pprint


def dem_eval_err_diff(n: int, d: int, M: HierarchicalGaussianModel, qu: dotdict, qp: dotdict): 
        # Get dimensions
        # ==============
        nl = len(M)
        ne = sum(m.l for m in M) 
        nv = sum(m.m for m in M)
        nx = sum(m.n for m in M)
        np = sum(M[i].p for i in range(nl - 1))
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

            p = M[i].pE + qp.u[i] @ qp.p[i]
            try: 
                res = M[i]._f(xi, vi, p)
            except: 
                raise RuntimeError(f"Error while evaluating model[{i}].f!")
            f.append(res)

            try: 
                res = M[i]._g(xi, vi, p)
            except: 
                raise RuntimeError(f"Error while evaluating model[{i}].g!")
            g.append(res)

        f = torch.cat(f).reshape((nx,))
        g = torch.cat(g).reshape((ne - nc,))

        # Evaluate derivatives at each level
        df  = list()
        d2f = list()
        dg  = list()
        d2g = list()
        for i in range(nl - 1): 
            xvp = tuple(_ if sum(_.shape) > 0 else torch.Tensor([]) for _ in  (x[i], v[i], qp.p[i], qp.u[i], M[i].pE))

            if M[i].df_qup is not None:
                dfi  = M[i].df_qup(*xvp)
                d2fi = M[i].d2f_qup(*xvp) 
            else: 
                dfi, d2fi = compute_df_d2f(lambda x, v, q, u, p: M[i].f(x, v, p + u @ q), xvp, ['dx', 'dv', 'dp', 'du', 'dq'])

            if M[i].dg_qup is not None:
                dgi  = M[i].dg_qup(*xvp)
                d2gi = M[i].d2g_qup(*xvp) 
            else: 
                dgi, d2gi = compute_df_d2f(lambda x, v, q, u, p: M[i].g(x, v, p + u @ q), xvp, ['dx', 'dv', 'dp', 'du', 'dq']) 

            dg.append(dgi)
            df.append(dfi)

            d2f.append(d2fi)
            d2g.append(d2gi)

        # Setup df
        df = dotdict({k: torch.block_diag(*(dfi[k] for dfi in df)) for k in ['dx', 'dv', 'dp']}) 

        # Setup dgdv manually 
        dgdv = cell(nl, nl-1)
        for i in range(nl - 1): 
            # causes (level i) appear at level i in g(x[i],v[i]) and at level i+1 as -I
            # nb: dg = [dydv[:], dv[0]dv[:], ...]
            dgdv[  i, i] = dg[i].dv 
            dgdv[i+1, i] = -torch.eye(M[i].m)

        dgdv = torch.tensor(block_matrix(dgdv))

        # Setup dg
        dg = dotdict({k: torch.block_diag(*(dgi[k] for dgi in dg)) for k in ['dx', 'dp']}) 
        # add an extra row to accomodate the highest hierarchical level
        for k in dg.keys():
            dg[k] = torch.cat([dg[k], torch.zeros(nc, dg[k].shape[1])], dim=0)
        dg.dv = dgdv

        # Reshape df and dg to avoid errors laters
        df.dx = df.dx.reshape((nx, nx))
        df.dv = df.dv.reshape((nx, nv))
        df.dp = df.dp.reshape((nx, np))
        dg.dx = dg.dx.reshape((ne, nx))
        dg.dv = dg.dv.reshape((ne, nv))
        dg.dp = dg.dp.reshape((ne, np))


        # Process d2f, d2g
        d2f = dotdict({i: dotdict({j: [d2fk[i][j] for d2fk in d2f] for j in ['dx', 'dv', 'dp']}) for i in ['dx', 'dv', 'dp']})
        d2g = dotdict({i: dotdict({j: [d2gk[i][j] for d2gk in d2g] for j in ['dx', 'dv', 'dp']}) for i in ['dx', 'dv', 'dp']}) 


        if np > 0:
            dfdxp = torch.stack([torch.block_diag(*(_[:, ip] for _ in d2f.dp.dx if torch.numel(_) > 0)) for ip in range(np)], dim=0)
            dfdvp = torch.stack([torch.block_diag(*(_[:, ip] for _ in d2f.dp.dv if torch.numel(_) > 0)) for ip in range(np)], dim=0)

            dgdvp = torch.stack([torch.block_diag(*(_[:, ip] for _ in d2g.dp.dv if torch.numel(_) > 0)) for ip in range(np)], dim=0)
            dgdxp = torch.stack([torch.block_diag(*(_[:, ip] for _ in d2g.dp.dx if torch.numel(_) > 0)) for ip in range(np)], dim=0)

            dgdxp = [torch.block_diag(*(_[:, ip] for _ in d2g.dp.dx if torch.numel(_) > 0)) for ip in range(np)]
            dgdvp = [torch.block_diag(*(_[:, ip] for _ in d2g.dp.dv if torch.numel(_) > 0)) for ip in range(np)]

            # Add a component with nc rows to accomodate the highest hierarchical level
            dgdxp = torch.stack([torch.cat([dgdxpi, torch.zeros(nc, dgdxpi.shape[1])]) for dgdxpi in dgdxp], dim=0)
            dgdvp = torch.stack([torch.cat([dgdvpi, torch.zeros(nc, dgdvpi.shape[1])]) for dgdvpi in dgdvp], dim=0)

            dfdxp = dfdxp.reshape((np, nx, nx))
            dfdvp = dfdvp.reshape((np, nx, nv))
            dgdxp = dgdxp.reshape((np, ne, nx))
            dgdvp = dgdvp.reshape((np, ne, nv))
        else: 
            dfdxp = torch.empty((np, nx, nx))
            dfdvp = torch.empty((np, nx, nv))
            dgdxp = torch.empty((np, ne, nx))
            dgdvp = torch.empty((np, ne, nv))

        if nx > 0: 
            dfdpx = torch.stack([torch.block_diag(*(_[:, ix] for _ in d2f.dx.dp if torch.numel(_) > 0)) for ix in range(nx)], dim=0)
            dgdpx = [torch.block_diag(*(_[:, ix] for _ in d2g.dx.dp if torch.numel(_) > 0)) for ix in range(nx)]

            # Add a component with nc rows to accomodate the highest hierarchical level
            dgdpx = torch.stack([torch.cat([dgdpxi, torch.zeros(nc, dgdpxi.shape[1])]) for dgdpxi in dgdpx], dim=0)

            dfdpx = dfdpx.reshape((nx, nx, np))
            dgdpx = dgdpx.reshape((nx, ne, np))
        else: 
            dfdpx = torch.empty((nx, nx, np))
            dgdpx = torch.empty((nx, ne, np))

        if nv > 0: 
            dfdpv = torch.stack([torch.block_diag(*(_[:, iv] for _ in d2f.dv.dp if torch.numel(_) > 0)) for iv in range(nv)], dim=0)
            dgdpv = [torch.block_diag(*(_[:, iv] for _ in d2g.dv.dp if torch.numel(_) > 0)) for iv in range(nv)]

            # Add a component with nc rows to accomodate the highest hierarchical level
            dgdpv = torch.stack([torch.cat([dgdpvi, torch.zeros(1, dgdpvi.shape[1])]) for dgdpvi in dgdpv], dim=0)

            dfdpv = dfdpv.reshape((nv, nx, np))
            dgdpv = dgdpv.reshape((nv, ne, np))
        else: 
            dfdpv = torch.empty((nv, nx, np))
            dgdpv = torch.empty((nv, ne, np))
        

        dfdpu = torch.cat([dfdpx, dfdpv], dim=0)
        dgdpu = torch.cat([dgdpx, dgdpv], dim=0)

        de    = dotdict(
          dy  = torch.eye(ne, ny), 
          dc  = torch.diag(-torch.ones(max(ne, nc) - (nc - ne)), nc - ne)[:ne, :nc]
        )

        # Prediction error (E) - causes        
        Ev = [torch.cat([qu.y[0], qu.v[0]]) -  torch.cat([g, qu.u[0]])]
        for i in range(1, n):
            Evi = de.dy @ qu.y[i] + de.dc @ qu.u[i] - dg.dx @ qu.x[i] - dg.dv @ qu.v[i]
            Ev.append(Evi)

        # Prediction error (E) - states
        Ex = [qu.x[1] - f]
        for i in range(1, n-1):
            Exi = qu.x[i + 1] - df.dx @ qu.x[i] - df.dv @ qu.v[i]
            Ex.append(Exi)
        Ex.append(torch.zeros_like(Exi))

        Ev = torch.cat(Ev)
        Ex = torch.cat(Ex)
        E  = torch.cat([Ev, Ex]).unsqueeze(1) 

        # generalised derivatives
        dgdp = [dg.dp]
        dfdp = [df.dp]

        qux = qu.x.unsqueeze(-1)
        quv = qu.v.unsqueeze(-1)
        for i in range(1, n):
            dgdpi = dg.dp.clone()
            dfdpi = df.dp.clone()

            for ip in range(np): 
                dgdpi[:, ip] = (dgdxp[ip] @ qux[i] + dgdvp[ip] @ quv[i]).squeeze(1)
                dfdpi[:, ip] = (dfdxp[ip] @ qux[i] + dfdvp[ip] @ quv[i]).squeeze(1)
            
            dfdp.append(dfdpi)
            dgdp.append(dgdpi)

        df.dp = torch.cat(dfdp)
        dg.dp = torch.cat(dgdp)

        de.dy               = torch.kron(torch.eye(n, n), de.dy)
        df.dy               = torch.kron(torch.eye(n, n), torch.zeros(nx, ny))
        df.dc               = torch.zeros(n*nx, n*nc)
        dg.dx               = torch.kron(torch.eye(n, n), dg.dx)
        # df.dx = (I * df.dx) - D, Eq. 45
        df.dx               = torch.kron(torch.eye(n, n), df.dx) - torch.kron(torch.diag(torch.ones(n - 1), 1), torch.eye(nx, nx)) 

        # embed to n >= d
        dedc                = torch.zeros(n*ne, n*nc) 
        dedc[:n*ne,:d*nc]   = torch.kron(torch.eye(n, d), de.dc)
        de.dc               = dedc

        dgdv                = torch.zeros(n*ne, n*nv)
        dgdv[:n*ne,:d*nv]   = torch.kron(torch.eye(n, d), dg.dv)
        dg.dv               = dgdv

        dfdv                = torch.zeros(n*nx, n*nv)
        dfdv[:n*nx,:d*nv]   = torch.kron(torch.eye(n, d), df.dv)
        df.dv               = dfdv

        dE    = dotdict()
        dE.dy = torch.cat([de.dy, df.dy])
        dE.dc = torch.cat([de.dc, df.dc])
        dE.dp = - torch.cat([dg.dp, df.dp])
        dE.du = - torch.Tensor(block_matrix([
                [dg.dx, dg.dv], 
                [df.dx, df.dv]]))

        dE.dup = []
        for ip in range(np): 
            dfdxpi              = torch.kron(torch.eye(n,n), dfdxp[ip])
            dgdxpi              = torch.kron(torch.eye(n,n), dgdxp[ip])
            dfdvpi              = torch.zeros(n*nx, n*nv)
            dfdvpi[:,:d*nv]     = torch.kron(torch.eye(n,d), dfdvp[ip])
            dgdvpi              = torch.zeros(n*ne, n*nv)
            dgdvpi[:,:d*nv]     = torch.kron(torch.eye(n,d), dgdvp[ip])

            dEdupi = -torch.Tensor(block_matrix([[dgdxpi, dgdvpi], 
                                                 [dfdxpi, dfdvpi]]))
            dE.dup.append(dEdupi)

        dE.dpu = [] 
        for i in range(n): 
            for iu in range(nx + nv):
                dfdpui = torch.kron(torch.eye(n,1), dfdpu[iu])
                dgdpui = torch.kron(torch.eye(n,1), dgdpu[iu])
                dEdpui = torch.cat([dgdpui, dfdpui], dim=0)

                dE.dpu.append(dEdpui)

        return E, dE