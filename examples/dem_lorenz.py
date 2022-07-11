from torchkf import *
import numpy as np

pE = np.array([18., 18., 46.92, 2., 1., 2., 4., 1., 1., 1.])

def lorenz(x, v, P):
    x0 = P[0] * x[1] - P[1] * x[0]
    x1 = P[2] * x[0] - P[3] * x[2] * x[0] - P[4] * x[1]
    x2 = P[5] * x[0] * x[1] - P[6] * x[2]
    
    return np.array([x0, x1, x2]) / 128

def obs(x, v, P): 
    return np.array(x.T @ P[-3:])

models = [GaussianModel(
    f=lorenz, 
    g=obs, 
    x=np.array([0.9,0.8,30])[:, None], pE=pE, sv=1/8., sw=1/8., m=1, 
    n=3, W=np.array([[np.exp(16)] * 3]), V=np.array([[np.exp(0)]]), 
)]

nT = 2**12
hdm = HierarchicalGaussianModel(*models)
gen = DEMInversion(hdm, states_embedding_order=3).generate(nT)
y   = gen.v[:, 0, :1, 0]


hdm[0]['x']  = np.array([[12,13,16]])
hdm[0]['pC'] = np.diag(np.ones(pE.shape)) * np.exp(-128)
hdm[0]['V'] *= np.exp(-4)

deminv = DEMInversion(hdm, states_embedding_order=12)
dec    = deminv.run(y, nD=1, nE=1, nM=1)