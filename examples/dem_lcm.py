import torch 
import numpy as np
from torchkf import *
from pprint import pprint
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

theta1 = torch.tensor([[0.125,  0.1633], 
                       [0.125,  0.0676], 
                       [0.125, -0.0676], 
                       [0.125, -0.1633]])
theta2 = torch.tensor([[-0.25,  1.00],
                       [-0.50, -0.25]])
theta3 = torch.tensor([[1.], 
                       [0.]])
pE = torch.cat([theta1.reshape((-1,)), theta2.reshape((-1,)), theta3.reshape((-1,))])

nps = (torch.numel(theta1),torch.numel(theta2),torch.numel(theta3))
models = [
    GaussianModel(
        g=lambda x, v, P: P[:nps[0]].reshape(theta1.shape) @ x, 
        f=lambda x, v, P: P[nps[0]:nps[0] + nps[1]].reshape(theta2.shape) @ x \
                        + P[-nps[2]:].reshape(theta3.shape) @ v,
        n=2, sv=1./2,sw=1./2,
        V=torch.tensor([np.exp(8.)]), 
        W=torch.tensor([np.exp(16.)]), 
        pE=pE, pC=torch.ones_like(pE) * np.exp(-32)
    ), 
    GaussianModel(l=1, V=torch.tensor([np.exp(32.)]))
]
genmodel = HierarchicalGaussianModel(*models)

nT  = 2**9
t   = np.arange(1, nT+1)  
u   = torch.tensor(np.exp(-(t - 12)**2/4)).unsqueeze(-1)
gen = DEMInversion(genmodel).generate(nT, u)
y   = gen.v[:,0,:4,0]

nps = (torch.numel(theta1),torch.numel(theta2),torch.numel(theta3))
models = [
    GaussianModel(
        g=lambda x, v, P: P[:nps[0]].reshape(theta1.shape) @ x, 
        f=lambda x, v, P: P[nps[0]:nps[0] + nps[1]].reshape(theta2.shape) @ x \
                        + P[-nps[2]:].reshape(theta3.shape) @ v,
        n=2, sv=1./2,sw=1./2,
        V=torch.tensor([np.exp(8.)]), 
        W=torch.tensor([np.exp(16.)]), 
        pE=pE, pC=torch.ones_like(pE) * np.exp(-32)
    ), 
    GaussianModel(l=1, V=torch.tensor([np.exp(0.)]))
]
decmodel = HierarchicalGaussianModel(*models)

deminv  = DEMInversion(decmodel)
results = deminv.run(y, nD=1, nE=1, nM=1, K=1, td=1)

fig = make_subplots(rows=2, cols=2) 

fig.add_scatter(y=results.qU.v[:, 0, 0], row=2, col=1, showlegend=True, legendgroup='estimated', name='Estimated', line_color=px.colors.qualitative.T10[0])
fig.add_scatter(y=gen.v[:, 0, -1, 0], row=2, col=1, showlegend=True, legendgroup='realized', name='Realized', line_dash='dash',line_color=px.colors.qualitative.T10[0])

for i in range(4): 
    fig.add_scatter(y=results.qU.y[:, 0, i], row=1, col=1,legendgroup='estimated', showlegend=False, line_color=px.colors.qualitative.T10[i])
    fig.add_scatter(y=y[:, i], row=1, col=1, legendgroup='realized',showlegend=False, line_dash='dash', line_color=px.colors.qualitative.T10[i])

for i in range(2): 
    fig.add_scatter(y=results.qU.x[:, 0, i], row=1, col=2, legendgroup='estimated',showlegend=False, line_color=px.colors.qualitative.T10[i])
    fig.add_scatter(y=gen.x[:, 0, i, 0], row=1, col=2, legendgroup='realized',showlegend=False, line_color=px.colors.qualitative.T10[i], line_dash='dash')
    
fig.update_layout(height=800, width=800, template='simple_white')
fig.show()