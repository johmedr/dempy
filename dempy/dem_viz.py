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

def plot_dem_generate(hgm, gen, time=None, show=True, tmin=0, tmax=None):     
    ix, iv = 0, 0

    if time is not None:
        tmin = np.argmin(np.abs(time - tmin)).astype('i')
        if tmax is not None:
            tmax = np.argmin(np.abs(time - tmax)).astype('i')
        time = time[tmin:tmax]

    tslice = slice(tmin, tmax)

    figs = []
    for i, m in enumerate(hgm):
        xr = gen.x[tslice][:, 0, ix:ix+m.n]
        vr = gen.v[tslice][:, 0, iv:iv+m.l]

        
        iv += m.l
        ix += m.n   
        
        if i == len(hgm) - 1 and m.l == 0:
            break
        
        fig = make_subplots(rows=1, cols=2)
        
        if i == 0:
            t = 'y'
        elif i == len(hgm) - 1:
            t = 'u'
        else: 
            t = 'v'

        for j in range(vr.shape[1]):  
            fig.add_scatter(x=time, y=vr[:, j], line_color=px.colors.qualitative.T10[j % len(px.colors.qualitative.T10)], name=f'{t}[{i},{j}]', 
                                line_width=1, row=1, col=1)    
            
        for j in range(xr.shape[1]): 
            fig.add_scatter(x=time, y=xr[:, j], line_color=px.colors.qualitative.T10[j % len(px.colors.qualitative.T10)], name=f'x[{i},{j}]', 
                                line_width=1, row=1, col=2)

        fig.update_xaxes(mirror='allticks', ticks='outside', linewidth=1, linecolor='black')
        fig.update_yaxes(mirror='allticks', ticks='outside', linewidth=1, linecolor='black')
        fig.update_layout(template='plotly_white', height=500, width=900)
        figs.append(fig)
        
        if show: 
            fig.show()
        
    return figs


def plot_dem_states(hgm, results, gen=None, time=None, tmin=0, tmax=None, overlay=None, show=True): 

    if time is not None:
        tmin = np.argmin(np.abs(time - tmin)).astype('i')
        if tmax is not None:
            tmax = np.argmin(np.abs(time - tmax)).astype('i')
        time = time[tmin:tmax]

    tslice = slice(tmin, tmax)

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
        fig = make_subplots(rows=1, cols=2)
        
        for j in range(v.shape[1]):  
            color = px.colors.qualitative.T10[j % len(px.colors.qualitative.T10)]
            if i > 0:
                fig.add_trace(Colorbar(mean=v[tslice, j], var=Cv[tslice, j, j], fillcolor=color, legendgroup=f'{t}[{i},{j}]', 
                                       opacity=0.3, showlegend=False), row=1, col=1)
            if gen is not None: 
                fig.add_scatter(y=vr[tslice, j], line_color=color, showlegend=False, legendgroup=f'{t}[{i},{j}]', 
                                line_dash='dash', line_width=1, row=1, col=1)    
            fig.add_scatter(y=v[tslice, j], line_color=color, name=f'{t}[{i},{j}]', legendgroup=f'{t}[{i},{j}]', line_width=1, row=1, col=1)
            
        for j in range(x.shape[1]): 
            color = px.colors.qualitative.T10[j % len(px.colors.qualitative.T10)]

            if i < len(hgm) - 1:
                fig.add_trace(Colorbar(mean=x[tslice, j], var=Cx[tslice, j, j], fillcolor=color, legendgroup=f'x[{i},{j}]', 
                                       opacity=0.3, showlegend=False), row=1, col=2)
            if gen is not None: 
                fig.add_scatter(y=xr[tslice, j], line_color=color, showlegend=False, legendgroup=f'x[{i},{j}]', 
                                line_dash='dash', line_width=1, row=1, col=2)    
            fig.add_scatter(y=x[tslice, j], line_color=color, name=f'x[{i},{j}]', legendgroup=f'x[{i},{j}]', line_width=1, row=1, col=2)
        fig.update_xaxes(mirror='allticks', ticks='outside', linewidth=1, linecolor='black')
        fig.update_yaxes(mirror='allticks', ticks='outside', linewidth=1, linecolor='black')
        fig.update_layout(template='plotly_white', height=500, width=900)
        figs.append(fig)
        
        if show: 
            fig.show()
        
    return figs