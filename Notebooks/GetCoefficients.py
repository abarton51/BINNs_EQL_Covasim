import joblib
import pandas as pd
import torch
from Modules.Utils.GetLowestGPU import *
from utils import get_case_name
import Modules.Loaders.DataFormatter as DF
import os
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

device = torch.device(GetLowestGPU(pick_from=[0,1,2,3]))
# instantiate BINN
path = '../Data/covasim_data/'
population = 200000
test_prob = 0.1
trace_prob = 0.3
keep_d = True
retrain = False
dynamic = True
chi_type = 'sin'
case_name = get_case_name(population, test_prob, trace_prob, keep_d, dynamic=dynamic, chi_type=chi_type)

n_runs = 100
n_samples = 50
params = DF.load_covasim_data(path, population, test_prob, trace_prob, case_name + '_' + str(n_runs), plot=False)
parameter_names = ['eta', 'beta', 'tau']
dict_parameter = {}
terms = {'eta': [r'$\beta_0$', 'S', 'S^2', 'A', 'Y'], 'beta': [r'$\beta_0$', 'S + A + Y', r'$\chi$'], 'tau': [r'$\beta_0$', 'A', 'Y']}
for pn_idx, parameter_name in enumerate(parameter_names):
    list_parameter = []
    for i in range(n_samples): # loop through each sample
        # mydir = '../models/covasim/2023-05-01_01-46-08'  # piecewise
        # mydir = '../models/covasim/2023-02-06_10-51-59'  # constant
        mydir = '../models/covasim/2023-05-01_17-03-03'  # sin
        # mydir = '../models/covasim/2023-05-01_01-21-08'  # constant
        save_path = os.path.join(mydir, case_name, str(i))
        list_parameter.append(pd.read_csv(os.path.join(save_path, case_name + '_regression_coef_' + parameter_name + '.csv'), index_col=[0]))
    df_concat = pd.concat(list_parameter, axis=1)
    df_concat.index = terms[parameter_name]
    dict_parameter[parameter_name] = df_concat
file_name = 'estimated_coefs.joblib'
joblib.dump(dict_parameter, os.path.join(mydir, case_name, file_name), compress=True)
# visualization
# for eta
for param_name in parameter_names:
    data = dict_parameter[param_name].T
    # data.columns = terms[param_name]
    sns.pairplot(data)
    fig_name = 'pairplot_' + param_name + '.png'
    plt.savefig(os.path.join(mydir, case_name, fig_name))
    print(data.corr())
# m, n = data.shape
# fig, ax = plt.subplots(1, m)
# fig.suptitle("Distributions")
# for i in range(m):
#     cur_term = terms[i]
#     ax[i].set_title(cur_term)
#     az.plot_dist(data[i,:], color="C1", label=cur_term, ax=ax[i])
#
#     # ax[1].set_title("Gaussian")
#     # az.plot_dist(data_gaussian, color="C2", label="Gaussian", ax=ax[1])
#     plt.show()
