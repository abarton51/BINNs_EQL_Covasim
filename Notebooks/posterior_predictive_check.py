import joblib
import numpy as np
import pandas as pd
import torch
from Modules.Utils.GetLowestGPU import *
from utils import get_case_name
import Modules.Loaders.DataFormatter as DF
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import empirical_covariance
from scipy.linalg import block_diag
import Modules.Utils.PDESolver as PDESolver
from Modules.Models.BuildBINNs import BINNCovasim
import scipy

device = torch.device(GetLowestGPU(pick_from=[0,1,2,3]))
path = '../Data/covasim_data/'
population = 200000
test_prob = 0.1
trace_prob = 0.3
keep_d = True
retrain = False
dynamic = True
chi_type = 'piecewise'
case_name = get_case_name(population, test_prob, trace_prob, keep_d, dynamic=dynamic, chi_type=chi_type)

n_runs = 100
n_samples = 50
params = DF.load_covasim_data(path, population, test_prob, trace_prob, case_name + '_' + str(n_runs), plot=False)
# mydir = '../models/covasim/2023-02-06_10-51-59' # constant
mydir = '../models/covasim/2023-05-01_01-46-08' # piecewise
# mydir = '../models/covasim/2023-05-01_17-03-03' # sin
# mydir = '../models/covasim/2023-05-01_01-21-08'  # constant

file_name = 'estimated_coefs.joblib'
estimated_coefs = joblib.load(os.path.join(mydir, case_name, file_name))

# grab initial condition
data = params['data'][0]
data = (data / params['population']).to_numpy()
N = len(data)
t_max = N - 1
t = np.arange(N)
u0 = data[0, :].copy()
RHS = PDESolver.STEAYDQRF_RHS_dynamic
binn = BINNCovasim(params, t_max, None, keep_d=keep_d).to(device)
weights = binn.weights_c.detach().numpy()
params['yita_lb'] = binn.yita_lb
params['yita_ub'] = binn.yita_ub
params['beta_lb'] = binn.beta_lb
params['beta_ub'] = binn.beta_ub
params['tau_lb'] = binn.tau_lb
params['tau_ub'] = binn.tau_ub

def get_r0(parameters, **fixed_parameters):
    eta_coefs = parameters[:5]
    s, a, y = 1, 0, 0
    features = [np.ones_like(a), s, s ** 2, a, y]
    features = np.array(features)
    eta0 = features @ eta_coefs
    eta0 = fixed_parameters['yita_lb'] + (fixed_parameters['yita_ub'] - fixed_parameters['yita_lb']) * eta0
    r0 = fixed_parameters['p_asymp'] * eta0 / fixed_parameters['lamda']  + (1-fixed_parameters['p_asymp']) * eta0 / (fixed_parameters['lamda'] + fixed_parameters['mu'] + fixed_parameters['delta'])
    return r0

def get_rt(r0, s):
    return r0 * s.reshape(-1,1)

def competition_model(parameters, **fixed_parameters):
    """Return the simulated trajectories."""
    eta, beta, tau = parameters[:5], parameters[5:-3], parameters[-3:]

    def contact_rate_regression(u):
        s, a, y = u[:, 0][:, None], u[:, 1][:, None], u[:, 2][:, None]
        features = [np.ones_like(a), s, s ** 2, a, y]  #
        features = np.concatenate(features, axis=1)
        res = features @ eta
        return res

    def beta_regression(u):
        a, b = u[:, 0][:, None], u[:, 1][:, None]
        features = [np.ones_like(a), a, b]  #
        features = np.concatenate(features, axis=1)
        res = features @ beta
        return res

    def tau_regression(u):
        a, b = u[:, 0][:, None], u[:, 1][:, None]
        features = [np.ones_like(a), a, b]  #
        features = np.concatenate(features, axis=1)
        res = features @ tau
        return res
    u_sim_regression = PDESolver.STEAYDQRF_sim(RHS, u0, t, contact_rate_regression, beta_regression, tau_regression,
                                               fixed_parameters, chi_type)
    u_sim_regression *= population

    return u_sim_regression


# # sample mean as the mean
# eta_mean = estimated_coefs['eta'].mean(axis=1).to_numpy()
# beta_mean = estimated_coefs['beta'].mean(axis=1).to_numpy()
# tau_mean = estimated_coefs['tau'].mean(axis=1).to_numpy()

concat_coefs = np.concatenate(list(estimated_coefs.values()), axis=0)
mean_vec = concat_coefs.mean(axis=1)
cov_matrix = empirical_covariance(concat_coefs.T)

# # covariance matrix estimated with sklearn
# eta_cov_matrix = empirical_covariance(estimated_coefs['eta'].to_numpy().T)
# beta_cov_matrix = empirical_covariance(estimated_coefs['beta'].to_numpy().T)
# tau_cov_matrix = empirical_covariance(estimated_coefs['tau'].to_numpy().T)
# concatenate first
# mean_vec = np.concatenate([eta_mean, beta_mean, tau_mean])
# cov_matrix = block_diag(*[eta_cov_matrix, beta_cov_matrix, tau_cov_matrix])
cov_matrix *= 64


data_simulated = competition_model(parameters = mean_vec, **params)
samples_X = np.random.multivariate_normal(mean_vec, cov_matrix, 500)
samples_Y = [competition_model(parameters= vec, **params) for vec in samples_X]
samples_r0 = [get_r0(parameters=vec, **params) for vec in samples_X]
samples_S_comp = [tmp[:, 0] / population for tmp in samples_Y]
samples_rt = [get_rt(r0_val, S_comp_val) for r0_val, S_comp_val in zip(samples_r0, samples_S_comp)]


def distance(simulated, obs):
    return np.sqrt(np.nanmean(((simulated - obs)/population * weights) ** 2))
train_data = params['data'][:n_samples]
train_data_mean = sum(train_data) / len(train_data)
train_data_mean = train_data_mean.to_numpy()
samples_dis = [distance(sample, train_data_mean) for sample in samples_Y]
threshold = np.quantile(samples_dis, 0.1)
valid = np.array(samples_dis) < threshold
posterior_samples_X = samples_X[valid]
posterior_samples_Y = []
posterior_samples_rt = []
for idx, tmp in enumerate(samples_Y):
    if valid[idx]:
        posterior_samples_Y.append(tmp)
        posterior_samples_rt.append(samples_rt[idx])
posterior_samples_Y_mean = sum(posterior_samples_Y) / len(posterior_samples_Y)
prior_samples_Y_mean = sum(samples_Y) / len(samples_Y)

posterior_samples_rt_mean = sum(posterior_samples_rt) / len(posterior_samples_rt)

def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = a.shape[1]
    m, se = np.mean(a, axis=1), np.std(a, axis=1)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    print(max(se))
    return m, m-h, m+h

n_compartments = posterior_samples_Y_mean.shape[1]
def get_CI(list_arrays, n_compartments):
    list_lb, list_ub = [], []
    for i in range(n_compartments):
        cur_data = [tmp[:, i] for tmp in list_arrays]
        cur_data = np.stack(cur_data, axis=1)
        _, cur_lb, cur_ub = confidence_interval(cur_data)
        list_lb.append(cur_lb)
        list_ub.append(cur_ub)
    lb_array = np.stack(list_lb, axis=1)
    ub_array = np.stack(list_ub, axis=1)
    return lb_array, ub_array

posterior_samples_Y_lb, posterior_samples_Y_ub = get_CI(posterior_samples_Y, n_compartments)
prior_samples_Y_lb, prior_samples_Y_ub = get_CI(samples_Y, n_compartments)

posterior_samples_rt_lb, posterior_samples_rt_ub = get_CI(posterior_samples_rt, 1)


# fig = plt.figure(figsize=(15, 15))
col_names = list('STEAYDQRF')
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
# # plot training data
# for idx in range(n_samples):
#     cur_data = params['data'][idx].to_numpy()
#     m, n = cur_data.shape
#     t = np.arange(m)
#     for i in range(n):
#         # ax = fig.add_subplot(int(np.ceil(n / 3)), 3, i)
#         ax = axs[i // 3, i % 3]
#         if idx == 0 and i == 0:
#             ax.plot(t, cur_data[:, i], '.g', alpha=0.2, label = 'Covasim Train Data')
#             ax.legend()
#         else:
#             ax.plot(t, cur_data[:, i], '.g', alpha=0.2)
# plot inference data
for idx in range(n_samples, n_runs):
    cur_data = params['data'][idx].to_numpy()
    m, n = cur_data.shape
    t = np.arange(m)
    for i in range(n):
        # ax = fig.add_subplot(int(np.ceil(n / 3)), 3, i)
        ax = axs[i // 3, i % 3]
        if idx == n_samples and i == 0:
            ax.plot(t, cur_data[:, i], '-', color='#4d771e', alpha=0.5, label = 'Covasim Test Data')
            ax.legend()
        else:
            ax.plot(t, cur_data[:, i], '-', color='#4d771e', alpha=0.5)
# # plot posterior samples
# for idx in range(len(posterior_samples_Y)):
#     cur_data = posterior_samples_Y[idx]
#     m, n = cur_data.shape
#     t = np.arange(m)
#     for i in range(n):
#         # ax = fig.add_subplot(int(np.ceil(n / 3)), 3, i)
#         ax = axs[i // 3, i % 3]
#         if idx == n_samples and i == 0:
#             ax.plot(t, cur_data[:, i], '.', color='#c78f65', alpha=0.2, label = 'Posterior sample')
#             ax.legend()
#         else:
#             ax.plot(t, cur_data[:, i], '.', color='#c78f65', alpha=0.2)
for i in range(n):
    # ax = fig.add_subplot(int(np.ceil(n / 3)), 3, i)
    ax = axs[i // 3, i % 3]
    ax.plot(t, posterior_samples_Y_mean[:, i], '-', color='#c75649', label='Posterior mean')
    ax.fill_between(t, prior_samples_Y_lb[:, i], prior_samples_Y_ub[:, i], alpha=1, color='#C7DCF1', label='Prior 95% CI')
    ax.fill_between(t, posterior_samples_Y_lb[:, i], posterior_samples_Y_ub[:, i], alpha=1, color='#c78f65',
                    label='Posterior 95% CI')

    ax.set_title(col_names[i])
    if i == 0:
        ax.legend(fontsize=12)
# for i in range(n):
#     # ax = fig.add_subplot(int(np.ceil(n / 3)), 3, i)
#     ax = axs[i // 3, i % 3]
#     ax.plot(t, posterior_samples_Y_lb[:, i], '-', color='#c78f65')
#     ax.set_title(col_names[i])
#     # if i == 0:
#     #     ax.legend()
# for i in range(n):
#     # ax = fig.add_subplot(int(np.ceil(n / 3)), 3, i)
#     ax = axs[i // 3, i % 3]
#     ax.fill_between(t, posterior_samples_Y_lb[:, i], posterior_samples_Y_ub[:, i], color='#c78f65')
#     ax.plot(t, posterior_samples_Y_ub[:, i], '-', color='#c78f65')
#     ax.set_title(col_names[i])
#     if i == 0:
#         ax.legend(fontsize=12)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.tight_layout(pad=2)
fig_name = case_name + '_' + str(threshold) + '.png'
plt.savefig(os.path.join(mydir, fig_name), dpi=300)

# plot rt
fig, ax = plt.subplots(1, 1)
ax.plot(t, posterior_samples_rt_mean[:, 0], '-', color='#c75649', label='Posterior mean')
# ax.fill_between(t, posterior_samples_rt_lb[:, i], posterior_samples_rt_ub[:, i], alpha=1, color='#C7DCF1', label='Prior 95% CI')
ax.fill_between(t, posterior_samples_rt_lb[:, 0], posterior_samples_rt_ub[:, 0], alpha=1, color='#c78f65',
                label='Posterior 95% CI')
fig_name = case_name + '_rt' + '.png'
plt.title(r'$R(t)=R_0S(t)/N$')
plt.savefig(os.path.join(mydir, fig_name))



