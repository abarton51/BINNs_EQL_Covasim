
#this is me trying to recreate the BINNCovasimEvaluation_dynamic notebook to see if I can figure out why it isn't working

import numpy as np
import pandas as pd

import sys
sys.path.append('../')

#from Modules.Utils.Imports import *
#from Modules.Models.BuildBINNs import BINNCovasim
#from Modules.Utils.ModelWrapper import ModelWrapper

#import PDESolver.py as PDESolver
#import Modules.Loaders.DataFormatter as DF
from utils import get_case_name #, AIC_OLS, RSS

# helper functions
def to_torch(x):
    return torch.from_numpy(x).float().to(device)
def to_numpy(x):
    return x.detach().cpu().numpy()

device = torch.device(GetLowestGPU(pick_from=[0,1,2,3]))

# instantiate BINN model parameters and path
path = '../Data/covasim_data/drums_data/'
# path = '../Data/covasim_data/xin_data/'

population = 50000
test_prob = 0.1
trace_prob = 0.3
keep_d = True
retrain = False
dynamic = True
masking = False
multiple = True
n_runs = 1000
chi_type = 'piecewise'

case_name = get_case_name(population, test_prob, trace_prob, keep_d, dynamic=dynamic, chi_type=chi_type)
# yita_lb, yita_ub = 0.2, 0.4


if masking:
    case_name = case_name + '_masking'
if multiple:
    params = DF.load_covasim_data(path, population, test_prob, trace_prob, keep_d, case_name + '_' + str(n_runs), plot=False)
else:
    params = DF.load_covasim_data(path, population, test_prob, trace_prob, keep_d, case_name, plot=False)

# split into train/val and convert to torch
if multiple:
    data = np.mean(params['data'], axis=0)
    data = (data / params['population'])
else:
    data = params['data']
    data = (data / params['population']).to_numpy()

N = len(data)
t_max = N - 1
t = np.arange(N)[:,None]

params.pop('data')

tracing_array = params['tracing_array']

# mydir = '../models/covasim/2023-06-21_00-12-29' # piecewise h(t) function
# mydir = '../models/covasim/2023-06-25_23-26-00' # constant h(t) function
# mydir = '../models/covasim/2023-06-26_23-48-27' # constant h(t) function
mydir = '../models/covasim/2023-07-06_23-47-16' # piecewise h(t) function, no masking

# instantiate BINN model
binn = BINNCovasim(params, t_max, tracing_array, keep_d=keep_d).to(device)

parameters = binn.parameters()
model = ModelWrapper(binn, None, None, save_name=os.path.join(mydir, case_name))

# load model weights. if retrain==True then load the retrained model
if retrain:
    model.save_name += '_retrain'
model.save_name += '_best_val'
model.load(model.save_name + '_model', device=device)

# grab initial condition
u0 = data[0, :].copy()

# learned surface fitter
def surface_fitter(t):
    res = binn.surface_fitter(t)
    return res

# learned contact_rate function
def contact_rate(u):
    res = binn.eta_func(to_torch(u)) # [:,[0,3,4]]
    return to_numpy(res)

# learned effective tracing rate function
def beta(u):
    res = binn.beta_func(to_torch(u))
    return to_numpy(res)

# learned diagnosis of quarantined rate function
def tau(u):
    res = binn.tau_func(to_torch(u))
    return to_numpy(res)

# do regression to figure out contact rate
def contact_rate_regression(u):
    s, a, y = u[:, 0][:, None], u[:, 1][:, None], u[:, 2][:, None]
    features = [np.ones_like(a), s, s**2, a, y] #
    features = np.concatenate(features, axis=1)
    res = features @ regression_coefs_cr
    # res *= 1.4
    return res

# do regression to figure out tracing rate
def beta_regression(u):
    a, b = u[:, 0][:, None], u[:, 1][:, None]
    features = [np.ones_like(a), a, b] #
    features = np.concatenate(features, axis=1)
    res = features @ regression_coefs_qt
    return res

# do regression to figure out diagnoses rate (on quarantined folks)
def tau_regression(u):
    a, b = u[:, 0][:, None], u[:, 1][:, None]
    features = [np.ones_like(a), a, b] #
    features = np.concatenate(features, axis=1)
    res = features @ regression_coefs_tau
    return res

t_torch = to_torch(t)
solutions = surface_fitter(t_torch).detach().numpy()
eta_values = contact_rate(data[:,[0, 3, 4]])
# beta_values = beta(np.sum(data[:,[0, 3, 4]], axis=1))
tau_values = tau(data[:,[3, 4]])
max_T = np.max(data[:,1])

max_S_nn = np.max(solutions[:,0])
max_T_nn = np.max(solutions[:,1])
max_E_nn = np.max(solutions[:,2])
max_A_nn = np.max(solutions[:,3])
max_Y_nn = np.max(solutions[:,4])
max_Q_nn = np.max(solutions[:,5])
max_D_nn = np.max(solutions[:,6])
max_R_nn = np.max(solutions[:,7])
max_F_nn = np.max(solutions[:,8])

# simulate PDE. First grab the parameter values from the loaded BINN model.
params['yita_lb'] = model.model.yita_lb
params['yita_ub'] = model.model.yita_ub
params['beta_lb'] = model.model.beta_lb
params['beta_ub'] = model.model.beta_ub
params['tau_lb'] = model.model.tau_lb
params['tau_ub'] = model.model.tau_ub







