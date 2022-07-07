import numpy as np
from torchkf import *

theta1 = np.array([[0.125,  0.1633], 
                       [0.125,  0.0676], 
                       [0.125, -0.0676], 
                       [0.125, -0.1633]])
theta2 = np.array([[-0.25,  1.00],
                       [-0.50, -0.25]])
theta3 = np.array([[1.], 
                       [0.]])
pE = np.concatenate([theta1.reshape((-1,)), theta2.reshape((-1,)), theta3.reshape((-1,))])

nps = (theta1.size,theta2.size,theta3.size)
models = [
    GaussianModel(
        g=lambda x, v, P: P[:nps[0]].reshape(theta1.shape) @ x, 
        f=lambda x, v, P: P[nps[0]:nps[0] + nps[1]].reshape(theta2.shape) @ x \
                        + P[-nps[2]:].reshape(theta3.shape) @ v,
        n=2, sv=1./2,sw=1./2,
        V=np.array([np.exp(8.)]), 
        W=np.array([np.exp(16.)]), 
        pE=pE, pC=np.ones_like(pE) * np.exp(-32)
    ), 
    GaussianModel(l=1, V=np.array([np.exp(32.)]))
]
genmodel = HierarchicalGaussianModel(*models)

nT = 2**12
t  = np.arange(1, nT+1)  
u  = (np.exp(-(t - 12)**2/4))[:, None]
gen = DEMInversion(genmodel, states_embedding_order=4).generate(nT, u)
y   = gen.v[:,0,:4,0]

decmodel = genmodel
decmodel[1].V = np.ones((1,1))

deminv  = DEMInversion(decmodel, states_embedding_order=4)
results = deminv.run(y, nD=1, nE=1, nM=1, K=1, td=1)