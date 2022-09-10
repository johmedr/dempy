import numpy as np
import plotly.express as px
import plotly.graph_objs as go    

from plotly.subplots import make_subplots

def Colorbar(mean=None, std=None, var=None, **kwargs):
    if var is not None: 
        std = np.sqrt(var)
    
    x = np.arange(std.size)
    x = np.concatenate([x, np.flip(x)])
    std = np.concatenate([std, -np.flip(std)])
    if mean is not None: 
        std = std + np.concatenate([mean, np.flip(mean)])

    if 'line_width' not in kwargs.keys():
        kwargs['line_width'] = 0
        
    return go.Scatter(x=x, y=std, fill='toself', mode='lines', **kwargs)

def plot_dem_generate(hgm, gen, time=None, show=True, tmin=0, tmax=None, names=None, subplots=True):     
    ix, iv = 0, 0

    if time is not None:
        tmin = np.argmin(np.abs(time - tmin)).astype('i')
        if tmax is not None:
            tmax = np.argmin(np.abs(time - tmax)).astype('i')
        time = time[tmin:tmax]

    tslice = slice(tmin, tmax)

    if names is None: 
        names = dict()

    figs = []
    for i, m in enumerate(hgm):
        xr = gen.x[tslice][:, 0, ix:ix+m.n]
        vr = gen.v[tslice][:, 0, iv:iv+m.l]

        
        iv += m.l
        ix += m.n   
        
        if i == len(hgm) - 1 and m.l == 0:
            break

        if i == 0:
            t = 'y'
        elif i == len(hgm) - 1:
            t = 'u'
        else: 
            t = 'v'

        if subplots: 
            fig = make_subplots(rows=1, cols=2)
            kw = dict(row=1, col=1) 
        else: 
            fig = go.Figure()
            kw = dict()
            _figs = [fig]

        for j in range(vr.shape[1]):  
            color = px.colors.qualitative.T10[j % len(px.colors.qualitative.T10)]
            name  = names.pop(f'{t}[{i},{j}]',f'{t}[{i},{j}]') 

            fig.add_scatter(x=time, y=vr[:, j], line_color=color, name=name, 
                                line_width=1, **kw)    


        fig.update_xaxes(mirror='allticks', ticks='outside', linewidth=1, linecolor='black', **kw)
        fig.update_yaxes(mirror='allticks', ticks='outside', linewidth=1, linecolor='black', **kw)
        fig.update_layout(template='plotly_white', height=500, width=900)

        if subplots: 
            kw = dict(row=1, col=2) 
        else: 
            if show: 
                fig.show()
            
            fig = go.Figure()
            kw = dict()
            
        for j in range(xr.shape[1]): 
            color = px.colors.qualitative.T10[j % len(px.colors.qualitative.T10)]
            name  = names.pop(f'x[{i},{j}]',f'x[{i},{j}]') 
            
            fig.add_scatter(x=time, y=xr[:, j], line_color=color, name=name, 
                                line_width=1, **kw)

        fig.update_xaxes(mirror='allticks', ticks='outside', linewidth=1, linecolor='black', **kw)
        fig.update_yaxes(mirror='allticks', ticks='outside', linewidth=1, linecolor='black', **kw)
        fig.update_layout(template='plotly_white', height=500, width=900)

        if subplots:
            figs.append(fig)
        else: 
            _figs.append(fig)
            figs.append(_figs)

        
        if show: 
            fig.show()
        
    return figs


def plot_dem_states(hgm, results, gen=None, time=None, tmin=0, tmax=None, overlay=None, show=True, names=None, subplots=True): 

    if time is not None:
        tmin = np.argmin(np.abs(time - tmin)).astype('i')
        if tmax is not None:
            tmax = np.argmin(np.abs(time - tmax)).astype('i')
        time = time[tmin:tmax]

    tslice = slice(tmin, tmax)
    
    if names is None:
        names = dict()

    qU = results.qU
    try:
    	C = np.linalg.pinv(results.qU.p)
    except np.linalg.LinAlgError: 
    	C  = results.qU.c
    	
    n = qU.x.shape[1]
    
    ix, iv, iu = 0, 0, 0
    nx = sum(m.n for m in hgm) * n
    
    figs = []
    for i, m in enumerate(hgm):
        x  = qU.x[:, 0, ix:ix+m.n]
        
        if i == 0: 
            v  = qU.y[:, 0]
            t  = 'y'
        else:
            v  = qU.v[:, 0, iu:iu+m.l]
            t  = 'v'
        
        if gen is not None:
            xr = gen.x[:, 0, ix:ix+m.n]
            vr = gen.v[:, 0, iv:iv+m.l]
            
        if i < len(hgm) - 1:
            Cx = C[:, n*ix:n*(ix+m.n), n*ix:n*(ix+m.n)]
            Cx = Cx.reshape((C.shape[0], n, m.n, n, m.n))[:, 0, :, 0]
            
        if i > 0:
            Cv = C[:, nx + n*iu:nx + n*(iu+m.l), nx + n*iu:nx + n*(iu+m.l)]
            Cv = Cv.reshape((C.shape[0], n, m.l, n, m.l))[:, 0, :, 0]
            iu += m.l
        iv += m.l
        ix += m.n   
        
        if i == len(hgm) - 1 and m.l == 0:
        	break

        if subplots: 
            fig = make_subplots(rows=1, cols=2)
            kw = dict(row=1, col=1) 
        else: 
            fig = go.Figure()
            kw = dict()

            _figs = [fig]
        
        for j in range(v.shape[1]):  
            color = px.colors.qualitative.T10[j % len(px.colors.qualitative.T10)]
            name = names.pop(f'{t}[{i},{j}]', f'{t}[{i},{j}]')

            if i > 0:
                fig.add_trace(Colorbar(mean=v[tslice, j], var=Cv[tslice, j, j], fillcolor=color, legendgroup=f'{t}[{i},{j}]', 
                                       opacity=0.3, showlegend=False), **kw)
            if gen is not None: 
                fig.add_scatter(y=vr[tslice, j], line_color=color, showlegend=False, legendgroup=f'{t}[{i},{j}]', 
                                line_dash='dash', line_width=1, **kw)    
            fig.add_scatter(y=v[tslice, j], line_color=color, name=name, legendgroup=f'{t}[{i},{j}]', line_width=1, **kw)

        fig.update_xaxes(mirror='allticks', ticks='outside', linewidth=1, linecolor='black', **kw)
        fig.update_yaxes(mirror='allticks', ticks='outside', linewidth=1, linecolor='black', **kw)
        fig.update_layout(template='plotly_white', height=500, width=900)

        if subplots: 
            kw = dict(row=1, col=2) 
        else: 
            if show: 
                fig.show()
            
            fig = go.Figure()
            kw = dict()

        for j in range(x.shape[1]): 
            color = px.colors.qualitative.T10[j % len(px.colors.qualitative.T10)]
            name = names.pop(f'x[{i},{j}]', f'x[{i},{j}]')

            if i < len(hgm) - 1:
                fig.add_trace(Colorbar(mean=x[tslice, j], var=Cx[tslice, j, j], fillcolor=color, legendgroup=f'x[{i},{j}]', 
                                       opacity=0.3, showlegend=False), **kw)
            if gen is not None: 
                fig.add_scatter(y=xr[tslice, j], line_color=color, showlegend=False, legendgroup=f'x[{i},{j}]', 
                                line_dash='dash', line_width=1, **kw)    
            fig.add_scatter(y=x[tslice, j], line_color=color, name=name, legendgroup=f'x[{i},{j}]', line_width=1, **kw)

        fig.update_xaxes(mirror='allticks', ticks='outside', linewidth=1, linecolor='black', **kw)
        fig.update_yaxes(mirror='allticks', ticks='outside', linewidth=1, linecolor='black', **kw)
        fig.update_layout(template='plotly_white', height=500, width=900)


        if subplots: 
            figs.append(fig)
        else: 
            _figs.append(fig)
            figs.append(_figs)

        if show: 
            fig.show()
        
    return figs