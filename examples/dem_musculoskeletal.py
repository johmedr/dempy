from torchkf import * 
import numpy as np
import scipy as sp
import sympy

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

""" format: value[, cov[, constaint]]
If constraint == 'positive', then the cov must be in log-space
""" 
parameters = dotdict(
    skeletal=dotdict(
        weight=dotdict(
            body=(70, np.exp(0), 'positive'), 
            relative=dotdict(
                forearm=(0.00762, np.exp(-4), 'positive'),
                upper_arm=(0.01373, np.exp(-4), 'positive'), 
                hand=(0.00361, np.exp(-4), 'positive'),
            ), 
        ), 
        size=dotdict(
            body=(1.75, np.exp(0), 'positive'), 
            relative=dotdict(
                forearm=(0.16480, np.exp(-4), 'positive'),
                upper_arm=(0.19590, np.exp(-4), 'positive'), 
                hand=(0.11465, np.exp(-4), 'positive'),
                humerus_factor=(0.8, np.exp(-4), 'positive'),
            ), 
        ),
        constraints=dotdict(
            elbow=dotdict(
                max_flexion=np.pi / 6,
                max_extension=9 * np.pi / 10.,
            )
        ),
#         mechanics=dotdict(
#             viscosity=(5, np.exp(0), 'positive'), 
#             stiffness=(5, np.exp(0), 'positive'), 
#         ),
        g=9.81,
    ),
    muscles=dotdict(
        triceps=dotdict(
            insertion=(0.02, np.exp(-8), 'positive'),
            tendon_length=(0.12, np.exp(-2), 'positive'),
            hill_muscle=dotdict(
                Fmax = (2397.12, np.exp(2), 'positive'), # N
                Lm0  = (0.15, np.exp(-2), 'positive'), # m
                phi0 = (0.2094, np.exp(-2)), # rad
                W    = (0.56, np.exp(-2), 'positive'),
                vmax = (1.3, np.exp(-2), 'positive'), # m/s
                A    = 0.25, 
                gmax = 1.5, 
                Lslack = (0.18, np.exp(0), 'positive'), # m
                Kmee1 = 10, 
                Kmee2 = 1000,
                Tact  = 0.1,
            ),
        ), 
        biceps=dotdict(
            insertion=(0.02, np.exp(-8), 'positive'),
            tendon_length=(0.12, np.exp(-2), 'positive'),
            hill_muscle=dotdict(
                Fmax = (2874.67, np.exp(2), 'positive'), # N
                Lm0  = (0.13, np.exp(-2), 'positive'), # m
                phi0 = (0, np.exp(-2)), # rad
                W    = (0.56, np.exp(-2), 'positive'),
                vmax = (1.3, np.exp(-2), 'positive'), # m/s
                A    = 0.25, 
                gmax = 1.5, 
                Lslack = (0.21, np.exp(0), 'positive'), # m
                Kmee1 = 10, 
                Kmee2 = 1000,
                Tact  = 0.1,
            ), 
        ),
    ), 
    afferents=dotdict(
        Ia=dotdict(
            kv=2.5,
            lmf=0.2,
            kdl=0.2,
            knl=0.5, 
            c1= 0.01,
        ), 
        Ib=dotdict(
            kf=1,
            fF=0.1,
        ),
    ),
)

def process_parameters(parameters): 
    proc_params = type(parameters)()
    for key, val in parameters.items(): 
        if isinstance(val, dict): 
            _proc_params = process_parameters(val)
            proc_params[key] = _proc_params
        else: 
            try: n = len(val)
            except TypeError: n = 0
            if n > 3: raise AttributeError(f'Parameter {key} has too many values (expected length 3)')
            param = [0, 0, None]
            
            # parameter value (prior expectation)
            if n == 0: param[0] = val
            else: param[0] = val[0]
                
            # parameter covariance
            if n > 1: param[1] = val[1]
                
            # parameter constraint
            if n == 3: param[2] = val[2]
            
            proc_params[key] = tuple(param)
    return proc_params
            
def flatten_parameters(parameters, namespace=''):
    flat_params = type(parameters)()
    for key, val in parameters.items(): 
        nkey = f'{namespace}.{key}' if len(namespace) > 0 else key
        if isinstance(val, dict): 
            _flat_params = flatten_parameters(val, namespace=nkey)
            flat_params.update(_flat_params)
        else: 
            flat_params[nkey] = val
    return flat_params            

import pandas as pd
df = pd.DataFrame(flatten_parameters(process_parameters(parameters))).transpose()
df = df.set_axis(['expectation', 'covariance', 'constraint'], axis=1)
df['constraint'] = df['constraint'].apply(lambda x: x if isinstance(x, str) else 'none')
df['covariance'] = df['covariance'].apply(lambda x: f'exp({int(np.log(x))})' if x > 0 else 0)

from abc import ABC, abstractmethod
from itertools import repeat
from functools import lru_cache

class NamedParamModel(ABC): 
    def __init__(self, 
                 state_names, input_names, param_names, 
                 state_map=None, input_map=None, param_map=None,
                 state_keys=None, input_keys=None, param_keys=None
                ):
        
        self.state_names = state_names
        self.input_names = input_names
        self.param_names = param_names
                
        if state_map is None: 
            self.state_map = {k:k for k in self.state_names}
        else: 
            self.state_map = {
                k:state_map[k] if k in state_map.keys() else k 
                for k in state_names}
        
        if input_map is None: 
            self.input_map = {k:k for k in self.input_names}
        else: 
            self.input_map = {
                k:input_map[k] if k in input_map.keys() else k 
                for k in input_names}
            
        if param_map is None: 
            self.param_map = {k:k for k in self.param_names}
        else: 
            self.param_map = {
                k:param_map[k] if k in param_map.keys() else k 
                for k in param_names}
            
        if None not in (state_keys, input_keys, param_keys): 
            self.compute_indices(state_keys, input_keys, param_keys)
            
            
    def compute_indices(self, state_keys, input_keys, param_keys):
        self.state_keys = state_keys
        self.input_keys = input_keys
        self.param_keys = param_keys
        
        self.ix = [state_keys.index(self.state_map[k]) for k in self.state_names]
        self.iv = [input_keys.index(self.input_map[k]) for k in self.input_names]
        self.ip = [param_keys.index(self.param_map[k]) for k in self.param_names]
        
    @staticmethod
    def unvec_args(func): 
        from functools import wraps
        
        @wraps(func)
        def _wrap(self, x, v, p, **kwargs):
            try: 
                _x = x[:, 0][self.ix]
                _v = v[:, 0][self.iv]
                _p = p[:, 0][self.ip]
            except TypeError: 
                _x, _v, _p = x, v, p
            
            return func(self, *_x, *_v, *_p, **kwargs)
        return _wrap

import symengine as si

class HillMuscle(NamedParamModel):  
    state_names = 'L', 'vel'
    input_names = 'aMN', 'gMN' 
    param_names = 'Kmee1', 'Kmee2', 'Lslack', 'Fmax', 'Lm0', 'W', 'vmax', 'A', 'gmax', 'phi0', 'kv', 'kdl', 'knl', 'c1', 'kf'
        
    def __init__(self, *args, **kwargs): 
        super().__init__(HillMuscle.state_names, HillMuscle.input_names, HillMuscle.param_names, *args, **kwargs)
    
    @staticmethod
    @lru_cache
    def force_velocity(vel, vmax, A, gmax): 
        cd    = vmax * A * (gmax - 1.) / (A + 1.)
        expr1 = (vmax + vel) / (vmax - vel/A)
        expr2 = (gmax * vel + cd) / (vel + cd)
#         h     = sheavyside(vel)
        
        return si.Piecewise((expr1, (vel <= 0)), (expr2, True))
#         return expr1
    
    @staticmethod
    @lru_cache
    def force_length(L, Lm0, W): 
        return sympy.exp(- (L - Lm0)**2/(W * Lm0)**2)
    
    @staticmethod
    @lru_cache
    def Fmee(L, Kmee1, Kmee2, Lslack):
        expr1 = Kmee1 * (L - Lslack)
        expr2 = Kmee2 * (L - Lslack)**2
#         h     = sheavyside(L - Lslack)
#         return expr1 + h * expr2
        return expr1 + si.Piecewise((0, (L < Lslack)), (expr2, True))
    
    @staticmethod
    @lru_cache
    def Fmce(aMN, Fmax, L, vel, Lm0, W, vmax, A, gmax): 
        fl = HillMuscle.force_length(L, Lm0, W)
        fv = HillMuscle.force_velocity(vel, vmax, A, gmax)
        
        return aMN * Fmax * fl * fv
    
    @staticmethod
    @lru_cache
    def Frte(L, vel, aMN, Kmee1, Kmee2, Lslack, Fmax, Lm0, W, vmax, A, gmax, phi0):     
        Fmee = HillMuscle.Fmee(L, Kmee1, Kmee2, Lslack)
        Fmce = HillMuscle.Fmce(aMN, Fmax, L, vel, Lm0, W, vmax, A, gmax)
        
        return (Fmee + Fmce) * si.cos(Lm0 * si.sin(phi0) / L)

    @staticmethod
    @lru_cache
    def rateIa(L, vel, gMN, Lm0, vmax, kv, kdl, knl, c1):
        dnorm = (L - 0.2 * Lm0) / Lm0
        vnorm = vel / vmax
        Ia    = kv * vnorm + kdl * dnorm + knl * gMN + c1 
#         Ia    = sympy.tanh(si.Abs(Ia))
        Ia    = si.Piecewise((0, Ia < 0), (sympy.tanh(Ia), True))
        
        return Ia
    
    @staticmethod
    @lru_cache
    def rateIb(Frte, Fmax, kf): 
        Fnorm = (Frte - 0.1 * Fmax) / Fmax
        Ib    = kf * Fnorm
        Ib    = si.Piecewise((0, Ib < 0), (sympy.tanh(Ib), True))
        
        return Ib
    
    @lru_cache
    @NamedParamModel.unvec_args
    def compute_output(self, L, vel, aMN, gMN, Kmee1, Kmee2, Lslack, Fmax, Lm0, W, 
                       vmax, A, gmax, phi0, kv, kdl, knl, c1, kf, return_force=True, return_rates=True):      
        y = []
        
        Frte   = HillMuscle.Frte(L, vel, aMN, Kmee1, Kmee2, Lslack, Fmax, W, vmax, A, gmax, Lm0, phi0)
        if return_force: 
            y.append(Frte)
            
        if return_rates: 
            rateIa = HillMuscle.rateIa(L, vel, gMN, Lm0, vmax, kv, kdl, knl, c1)
            rateIb = HillMuscle.rateIb(Frte, Fmax, kf)       
            
            y.extend([rateIa, rateIb])
        if len(y) == 1:
            y, = y
        
        return y

state_keys = 'L', 'vel'
input_keys = 'aMN', 'gMN'
params     = flatten_parameters(process_parameters(parameters))
param_keys = [*params.keys()]
param_map  = {k: f'muscles.triceps.hill_muscle.{k}' 
              for k in HillMuscle.param_names if f'muscles.triceps.hill_muscle.{k}' in param_keys}
param_map.update(
    {k: f'afferents.Ia.{k}' for k in HillMuscle.param_names if f'afferents.Ia.{k}' in param_keys})
param_map.update(
    {k: f'afferents.Ib.{k}' for k in HillMuscle.param_names if f'afferents.Ib.{k}' in param_keys})

triceps_muscle = HillMuscle(param_map=param_map ,state_keys=state_keys, input_keys=input_keys, param_keys=param_keys)

from collections import OrderedDict

class ElbowModel(NamedParamModel):
    state_names = 'q', 'dq'
    input_names = 'q2', 'aMNb', 'aMNt', 'gMNb', 'gMNt'
    output_names = 'q', 'aMNb', 'aMNt', 'gMNb', 'gMNt', 'rateIa_b', 'rateIb_b', 'rateIa_t', 'rateIb_t'
    output_keys = output_names
    param_map   = OrderedDict({
        'uarm_l_rel':    'skeletal.size.relative.upper_arm', 
        'farm_l_rel':    'skeletal.size.relative.forearm', 
        'hand_l_rel':    'skeletal.size.relative.hand', 
        'uarm_w_rel':    'skeletal.weight.relative.upper_arm',
        'farm_w_rel':    'skeletal.weight.relative.forearm', 
        'hand_w_rel':    'skeletal.weight.relative.hand', 
        'body_w':        'skeletal.weight.body', 
        'body_h':        'skeletal.size.body',
        'humerus_factor':'skeletal.size.relative.humerus_factor', 
        'lrte_b':        'muscles.biceps.tendon_length', 
        'lrte_t':        'muscles.triceps.tendon_length', 
        'g':             'skeletal.g', 
        'db':            'muscles.biceps.insertion', 
        'dt':            'muscles.triceps.insertion', 
    })
    
    def __init__(self, param_keys, **kwargs): 
        pmap = OrderedDict(ElbowModel.param_map)
        
        param_maps = dict()
        for muscle in ('biceps', 'triceps'): 
            param_maps[muscle] = {k: f'muscles.{muscle}.hill_muscle.{k}' 
                  for k in HillMuscle.param_names if f'muscles.{muscle}.hill_muscle.{k}' in param_keys}
            param_maps[muscle].update(
                {k: f'afferents.Ia.{k}' for k in HillMuscle.param_names if f'afferents.Ia.{k}' in param_keys})
            param_maps[muscle].update(
                {k: f'afferents.Ib.{k}' for k in HillMuscle.param_names if f'afferents.Ib.{k}' in param_keys})
        
        self.biceps_model  = HillMuscle(param_map=param_maps['biceps'], state_keys=HillMuscle.state_names, input_keys=HillMuscle.input_names, param_keys=param_keys)
        self.triceps_model = HillMuscle(param_map=param_maps['triceps'], state_keys=HillMuscle.state_names, input_keys=HillMuscle.input_names, param_keys=param_keys)
        
        pmap.update({f'biceps.{k}': v for k, v in self.biceps_model.param_map.items()})
        pmap.update({f'triceps.{k}': v for k, v in self.triceps_model.param_map.items()})
        
        pnames = (*pmap.keys(),)
        
        super().__init__(ElbowModel.state_names, ElbowModel.input_names, pnames, param_map=pmap, 
                        state_keys=ElbowModel.state_names, input_keys=ElbowModel.input_names, param_keys=param_keys,
                        **kwargs)
        
        self.n  = len(self.state_names)
        self.m  = len(self.input_names)
        self.nP = len(self.param_names)
    
    @staticmethod
    @lru_cache
    def body_mensurations(uarm_l_rel, farm_l_rel, hand_l_rel, uarm_w_rel, farm_w_rel, hand_w_rel, 
                          body_w, body_h, humerus_factor):
        uarm_l, farm_l, hand_l = (rel * body_h for rel in (uarm_l_rel, farm_l_rel, hand_l_rel))
        uarm_w, farm_w, hand_w = (rel * body_w for rel in (uarm_w_rel, farm_w_rel, hand_w_rel))
        hl = uarm_l * humerus_factor
        return uarm_l, farm_l, hand_l, uarm_w, farm_w, hand_w, hl
    
    @staticmethod
    @lru_cache
    def muscle_length(q, hl, db, dt, lrte_b, lrte_t): 
        a  = si.sin(q)
        b  = si.cos(q)
        lb = si.sqrt((db * a)**2 + (hl - db * b)**2) - lrte_b
        lt = si.sqrt((dt * a)**2 + (hl + dt * b)**2) - lrte_t
        return lb, lt
    
    @staticmethod
    @lru_cache
    def moment_arm(q, lb, lt, hl, db, dt, lrte_b, lrte_t):
        rb = hl * db * si.sin(q) / (lb + lrte_b)
#         rb = si.Piecewise((rb, rb > 1e-2),(1e-2, True))  
        rt = - hl * dt * si.sin(q) / (lt + lrte_t)
#         rt = si.Piecewise((rt, rt < -1e-2),(-1e-2, True))  
#         rt = sympy.Min(rt, -1e-2) 
        return rb, rt
    
    @staticmethod
    @lru_cache
    def elbow_dynamic_properties(farm_w, farm_l, hand_w, hand_l):
        farm_com = farm_l / 2.
        hand_com = (farm_l + hand_l/2)
        resultant_mass = farm_w * farm_com + hand_w * hand_com
        inertia        = farm_w * farm_com**2 + hand_w * hand_com**2
        return resultant_mass, inertia
    
    @lru_cache
    def eval_muscle_models(self, dq, aMNb, aMNt, gMNb, gMNt, lb, lt, rb, rt, args_b, args_t, **kwargs): 
        dlb = rb * dq
        dlt = rt * dq
        
        bic = self.biceps_model.compute_output((lb, dlb), (aMNb, gMNb), args_b, **kwargs)
        tri = self.triceps_model.compute_output((lt, dlt), (aMNt, gMNt), args_t, **kwargs)
        
        return bic, tri
    
    @staticmethod
    @lru_cache
    def joint_acceleration(q, q2, Fbiceps, Ftriceps, rb, rt, farm_w, farm_l, hand_w, hand_l, g): 
        resultant_mass, inertia = ElbowModel.elbow_dynamic_properties(farm_w, farm_l, hand_w, hand_l)
        Mmuscles = - rb * Fbiceps - rt * Ftriceps
        Mgravity = - resultant_mass * g * sympy.sin(q2 - q)
        
        Mjoint   = Mmuscles + Mgravity
        ddq = Mjoint / inertia
        return ddq
    
    @NamedParamModel.unvec_args
    @lru_cache
    def compute_dynamics(self, q, dq, q2, aMNb, aMNt, gMNb, gMNt, uarm_l_rel, farm_l_rel, hand_l_rel, 
                         uarm_w_rel, farm_w_rel, hand_w_rel, body_w, body_h, humerus_factor, lrte_b, 
                         lrte_t, g, db, dt, *args): 
        args_b = args[:len(args)//2]
        args_t = args[len(args)//2:]
        
#         aMNb = sheavyside(aMNb, k=1)
#         aMNt = sheavyside(aMNt, k=1)
        
        uarm_l, farm_l, hand_l, uarm_w, farm_w, hand_w, hl =\
            ElbowModel.body_mensurations(uarm_l_rel, farm_l_rel, hand_l_rel, uarm_w_rel, farm_w_rel, 
                                         hand_w_rel, body_w, body_h, humerus_factor)
        lb, lt = ElbowModel.muscle_length(q, hl, db, dt, lrte_b, lrte_t)
        rb, rt = ElbowModel.moment_arm(q, lb, lt, hl, db, dt, lrte_b, lrte_t)
        
        Fbiceps, Ftriceps = self.eval_muscle_models(
            dq * 1e-2, aMNb, aMNt, gMNb, gMNt, lb, lt, rb, rt, args_b, args_t, return_rates=False)
        
        ddq    = ElbowModel.joint_acceleration(q, q2, Fbiceps, Ftriceps, rb, rt, farm_w, farm_l, hand_w, hand_l, g)
        
        y = si.Matrix([[dq], [ddq* 1e-2]]) * 1e-2
        return y
    
    
    @NamedParamModel.unvec_args
    @lru_cache
    def compute_output(self,  q, dq, q2, aMNb, aMNt, gMNb, gMNt, uarm_l_rel, farm_l_rel, hand_l_rel, 
                         uarm_w_rel, farm_w_rel, hand_w_rel, body_w, body_h, humerus_factor, lrte_b, lrte_t, g, db, dt, *args): 
        args_b = args[:len(args)//2]
        args_t = args[len(args)//2:]
        
        uarm_l, farm_l, hand_l, uarm_w, farm_w, hand_w, hl =\
            ElbowModel.body_mensurations(uarm_l_rel, farm_l_rel, hand_l_rel, uarm_w_rel, farm_w_rel, 
                                         hand_w_rel, body_w, body_h, humerus_factor)
        
        lb, lt = ElbowModel.muscle_length(q, hl, db, dt, lrte_b, lrte_t)
        rb, rt = ElbowModel.moment_arm(q, lb, lt, hl, db, dt, lrte_b, lrte_t)
        
        (rateIa_b, rateIb_b), (rateIa_t, rateIb_t) = \
            self.eval_muscle_models(dq * 1e-2, aMNb, aMNt, gMNb, gMNt, lb, lt, rb, rt, args_b, args_t, return_force=False)
        
#         y = 
        y = si.Matrix([[q], [aMNb], [aMNt], [gMNb], [gMNt], [rateIa_b], [rateIb_b], [rateIa_t], [rateIb_t]])
        return y

pE, pC, constraints = [[params[k][i] for k in param_keys] for i in range(3)] 
pE = np.asarray(pE)
pC = np.asarray(pC)
cC = np.asarray(constraints)

elbow = ElbowModel([*params.keys()])
elbow.ix, elbow.iv, elbow.ip; 

models = [
    GaussianModel(
        fsymb=elbow.compute_dynamics, 
        gsymb=elbow.compute_output,
        n=elbow.n, m=elbow.m, 
        pE=pE, pC=pC, V=np.array([np.exp(16)] * 5), W=np.array([np.exp(16)] * elbow.n), 
        x=np.array([[np.pi/2.],[0]]), constraints=cC, 
    ), 
    GaussianModel(l=elbow.m, v=np.zeros((elbow.m, 1)), V=np.array([np.exp(32)] * elbow.m))
]

hgm = HierarchicalGaussianModel(*models)

N  = 1024
dt = 1e-2
t  = np.linspace(0, N * dt, N)
aMNb = 0.2 + 0.2 * np.cos(2 * 0.2 * np.pi * t)
aMNt = 0.2 + 0.2 * np.cos(2 * 0.2 * np.pi * t + np.pi)
u  = np.zeros((N, elbow.m))
u[:, 1] = aMNb
u[:, 2] = aMNt

results = DEMInversion(hgm, states_embedding_order=4).generate(N, u=u)

in_vE = np.array([0, 0.1, 0.1, 0, 0]).reshape((5, 1))
in_vP = np.array([np.exp(32), np.exp(-8), np.exp(-8), np.exp(16), np.exp(16)])

models = [
    GaussianModel(
        fsymb=elbow.compute_dynamics, 
        gsymb=elbow.compute_output,
        n=elbow.n, m=elbow.m, 
        pE=pE, pC=pC * 1e-32, V=np.array([np.exp(4)] * 5), 
        W=np.array([np.exp(4)] * elbow.n), x=np.array([[np.pi/2.],[0]])
    ), 
    GaussianModel(v=in_vE, V=in_vP)
]
hgmd = HierarchicalGaussianModel(*models)

N  = 1024
dt = 1e-2
t  = np.linspace(0, N * dt, N)
q  = np.pi/2 + np.pi/4. * np.cos(2 * 0.5 * np.pi * t)
y  = np.zeros((N, hgmd[0].l))
y[:, 0] = q
dec = DEMInversion(hgmd).run(y, nE=1)