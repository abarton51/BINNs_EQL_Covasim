import numpy as np
import pandas as pd

from Modules.Utils.Imports import *
from Modules.Models.BuildBINNs import BINNCovasim
from Modules.Utils.ModelWrapper import ModelWrapper

import Modules.Utils.PDESolver as PDESolver
import Modules.Loaders.DataFormatter as DF
from utils import get_case_name, AIC_OLS, RSS
# helper functions
def to_torch(x):
    return torch.from_numpy(x).float().to(device)
def to_numpy(x):
    return x.detach().cpu().numpy()


device = torch.device(GetLowestGPU(pick_from=[0,1,2,3]))
# instantiate BINN
path = '../Data/covasim_data/'
population = 200000
test_prob = 0.1
trace_prob = 0.3
keep_d = True
retrain = False
dynamic = True
chi_type = 'piecewise'
case_name = get_case_name(population, test_prob, trace_prob, keep_d, dynamic=dynamic, chi_type=chi_type)
params = DF.load_covasim_data(path, population, test_prob, trace_prob, case_name, plot=False)

data = params['data']
data = (data / params['population']).to_numpy()
N = len(data)
t_max = N - 1
t = np.arange(N)
params.pop('data')

tracing_array = params['tracing_array']
# mydir = '../models/covasim/2022-12-19_15-22-50' # constant
# mydir = '../models/covasim/2022-12-22_15-47-23' # constant
# mydir = '../models/covasim/2022-12-16_15-25-39' # sin
# mydir = '../models/covasim/2022-12-08_19-43-15' # piecewise

mydir = '../models/covasim/2023-03-28_14-41-05' # sin

regression_coefs_cr = pd.read_csv(os.path.join(mydir, case_name, case_name + '_regression_coef_eta.csv'), index_col=0).to_numpy()
regression_coefs_qt = pd.read_csv(os.path.join(mydir, case_name, case_name + '_regression_coef_beta.csv'), index_col=0).to_numpy()
regression_coefs_tau = pd.read_csv(os.path.join(mydir, case_name, case_name + '_regression_coef_tau.csv'), index_col=0).to_numpy()
# regression_coefs = pd.read_csv('../Figures/covasim/' + '50000_0.1_0.1' + '_regression_coef.csv', index_col=0).to_numpy()
# regression_coefs_qt = np.append(regression_coefs_qt, 0.0).reshape(-1,1)
binn = BINNCovasim(params, t_max, tracing_array, keep_d=keep_d).to(device)

parameters = binn.parameters()
model = ModelWrapper(binn, None, None, save_name=os.path.join(mydir, case_name))

# prop_cycle = plt.rcParams['axes.prop_cycle']
# colors = prop_cycle.by_key()['color']
# markers = ['x', 'o', 's', 'd', '^']
# fig = plt.figure(figsize=(15 ,8))

# load model weights
if retrain:
    model.save_name += '_retrain'
model.save_name += '_best_val'
model.load(model.save_name + '_model', device=device)

# grab initial condition
u0 = data[0, :].copy()

# learned contact_rate function
def contact_rate(u):
    res = binn.eta_func(to_torch(u)) # [:,[0,3,4]]
    return to_numpy(res)

def beta(u):
    res = binn.beta_func(to_torch(u))
    return to_numpy(res)

def tau(u):
    res = binn.tau_func(to_torch(u))
    return to_numpy(res)

def contact_rate_regression(u):
    s, a, y = u[:, 0][:, None], u[:, 1][:, None], u[:, 2][:, None]
    features = [np.ones_like(a), s, s**2, a, y] #
    features = np.concatenate(features, axis=1)
    res = features @ regression_coefs_cr
    # res *= 1.4
    return res

def beta_regression(u):
    a, b = u[:, 0][:, None], u[:, 1][:, None]
    features = [np.ones_like(a), a, b] #
    features = np.concatenate(features, axis=1)
    res = features @ regression_coefs_qt
    return res

def tau_regression(u):
    a, b = u[:, 0][:, None], u[:, 1][:, None]
    features = [np.ones_like(a), a, b] #
    features = np.concatenate(features, axis=1)
    res = features @ regression_coefs_tau
    return res

# s_min, s_max = data[:,0].min(), data[:,0].max()
# a_min, a_max = data[:,3].min(), data[:,3].max()
# y_min, y_max = data[:,4].min(), data[:,4].max()
# chi_min, chi_max = 0.0, params['eff_ub']
#
# s_grid = np.linspace(s_min, s_max, 10)
# a_grid = np.linspace(a_min, a_max, 10)
# y_grid = np.linspace(y_min, y_max, 10)
# chi_grid = np.linspace(chi_min, chi_max, 10)



# simulate PDE
params['yita_lb'] = model.model.yita_lb
params['yita_ub'] = model.model.yita_ub
params['beta_lb'] = model.model.beta_lb
params['beta_ub'] = model.model.beta_ub
params['tau_lb'] = model.model.tau_lb
params['tau_ub'] = model.model.tau_ub

if keep_d:
    RHS = PDESolver.STEAYDQRF_RHS_dynamic
    u_sim_NN = PDESolver.STEAYDQRF_sim(RHS, u0, t, contact_rate, beta, tau, params, chi_type)
    u_sim_NN *= population
    u_sim_regression = PDESolver.STEAYDQRF_sim(RHS, u0, t, contact_rate_regression, beta_regression, tau_regression, params, chi_type)
    u_sim_regression *= population
# else:
#     RHS = PDESolver.STEAYQRF_RHS
#     u_sim_NN = PDESolver.STEAYQRF_sim(RHS, u0, t, contact_rate, quarantine_test, params)
#     u_sim_NN *= population
#     u_sim_regression = PDESolver.STEAYQRF_sim(RHS, u0, t, contact_rate_regression, quarantine_test_regression, params)
#     u_sim_regression *= population

data *= population
plot=True
if plot:
    # data = params['data']
    n = data.shape[1]
    col_names = list('STEAYDQRF') if keep_d else list('STEAYQRF')
    # t = np.arange(1, data.shape[0] + 1)
    # plot compartments
    fig = plt.figure(figsize=(15, 15))
    for i in range(1, n + 1):
        ax = fig.add_subplot(int(np.ceil(n / 3)), 3, i)
        ax.plot(t, data[:, i - 1], '.k', label='Covasim Data')
        ax.plot(t, u_sim_NN[:, i - 1], '-*r', label='ODE-NN')
        ax.plot(t, u_sim_regression[:, i - 1], '-b', label='ODE-Regression')
        ax.set_title(col_names[i - 1])
        ax.legend(fontsize=8)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.tight_layout(pad=2)
        plt.savefig(os.path.join(model.save_folder, case_name + '.png') )
        # plt.show()

# %% optional, add one comparison based on the results ODE-DE
de_res_file = 'fitted_' + case_name + '.csv'
ode_de_df = pd.read_csv('../Data/covasim_data/' + de_res_file, index_col=0)
ode_de_array = ode_de_df.to_numpy()
# plot compartments
plt.rcParams.update({'font.size': 12})
fig = plt.figure(figsize=(18, 9))
for i in range(1, n):
    ax = fig.add_subplot(2, 4, i) # int(np.ceil(n / 4))
    ax.plot(t, data[:, i - 1], '.k', label='Test-ABM')
    ax.plot(t, u_sim_NN[:, i - 1], 'r.-', label='ODE-NN')
    ax.plot(t, u_sim_regression[:, i - 1], 'b.-', label='ODE-SR')
    ax.plot(t, ode_de_array[:, i - 1], 'g.-', label='ODE-DE')
    if i > 5:
        ax.set_xlabel("Time (Days)")
    if i % 3 == 1:
        ax.set_ylabel("Count")
    ax.set_title(col_names[i - 1])
    if i == 1:
        ax.legend(loc="best")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.tight_layout(pad=2)
    plt.savefig(os.path.join(model.save_folder, case_name + '_without_F' + '.png'), dpi=300)
    # plt.show()

# %% save without R
fig = plt.figure(figsize=(18, 9))
for i in range(1, n + 1):
    if i == n - 1:
        continue
    ax = fig.add_subplot(2, 4, i) if i < n else fig.add_subplot(2, 4, i - 1) # int(np.ceil(n / 4))
    ax.plot(t, data[:, i - 1], '.k', label='Test-ABM')
    ax.plot(t, u_sim_NN[:, i - 1], 'r.-', label='ODE-NN')
    ax.plot(t, u_sim_regression[:, i - 1], 'b.-', label='ODE-SR')
    ax.plot(t, ode_de_array[:, i - 1], 'g.-', label='ODE-DE')
    if i > 4:
        ax.set_xlabel("Time (Days)")
    if i % 4 == 1:
        ax.set_ylabel("Count")
    ax.set_title(col_names[i - 1])
    if i == 1:
        ax.legend(loc="best")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.tight_layout(pad=2)
    plt.savefig(os.path.join(model.save_folder, case_name + '_without_R' + '.png'), dpi=300)

#%% RMSE
from sklearn.metrics import mean_squared_error
rmse_NN = {}
rmse_SR = {}
rmse_DE = {}
for i, col in enumerate(col_names):
    res = mean_squared_error(data[:, i], u_sim_NN[:,i], squared=False)
    rmse_NN[col] = res
    res = mean_squared_error(data[:, i], u_sim_regression[:, i], squared=False)
    rmse_SR[col] = res
    res = mean_squared_error(data[:, i], ode_de_array[:, i], squared=False)
    rmse_DE[col] = res
rmse_df = pd.DataFrame([rmse_NN, rmse_SR, rmse_DE], index=['ODE-NN', 'ODE-SR', 'ODE-DE'])
rmse_df['Mean'] = rmse_df.mean(axis=1)
rmse_df.to_csv(os.path.join(model.save_folder, case_name + '_rmse' + '.csv'))

#%% NRMSE
from sklearn.metrics import mean_squared_error
nrmse_NN = {}
nrmse_SR = {}
nrmse_DE = {}
for i, col in enumerate(col_names):
    y_max, y_min = max(data[:, i]), min(data[:, i])
    res = mean_squared_error(data[:, i], u_sim_NN[:,i], squared=False) / (y_max - y_min)
    nrmse_NN[col] = res
    res = mean_squared_error(data[:, i], u_sim_regression[:, i], squared=False) / (y_max - y_min)
    nrmse_SR[col] = res
    res = mean_squared_error(data[:, i], ode_de_array[:, i], squared=False) / (y_max - y_min)
    nrmse_DE[col] = res
nrmse_df = pd.DataFrame([nrmse_NN, nrmse_SR, nrmse_DE], index=['ODE-NN', 'ODE-SR', 'ODE-DE'])
nrmse_df['Mean'] = nrmse_df.mean(axis=1)
nrmse_df.to_csv(os.path.join(model.save_folder, case_name + '_nrmse' + '.csv'))



#
# # %% optional, calcualte the AIC values for ODE-Regression
#
# aic_regression = AIC_OLS(u_sim_regression, data, 6)
# aic_de = AIC_OLS(ode_de_array, data, 2)
#
# # RSS for each variable
# RSS_regression = RSS(u_sim_regression, data)
# RSS_de = RSS(ode_de_array, data)
#
# RSS_df = pd.DataFrame(data={'RSS_sparse': RSS_regression, 'RSS_de': RSS_de}, index=list('STEAYDQRF'))
#
# RSS_df.to_csv('RSS_results.csv')
#
# # %% optional, save compartment values
# pd.DataFrame(data=data, columns=list('STEAYDQRF')).to_csv('Covasim-Data.csv')
# pd.DataFrame(data=u_sim_regression, columns=list('STEAYDQRF')).to_csv('ODE-regression.csv')
# pd.DataFrame(data=u_sim_NN, columns=list('STEAYDQRF')).to_csv('ODE-NN.csv')
# pd.DataFrame(data=ode_de_array, columns=list('STEAYDQRF')).to_csv('ODE-DE.csv')
#
# # %% optional, get the peak time
# idx = 4
# data[:, idx].argmax(), u_sim_regression[:,idx].argmax(), u_sim_NN[:, idx].argmax(), ode_de_array[:, idx].argmax()
