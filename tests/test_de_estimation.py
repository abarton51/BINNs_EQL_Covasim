import joblib
import pandas as pd
import os
import Modules.Loaders.DataFormatter as DF
from Notebooks.utils import get_case_name

from SIR_models.sir_models import STEAYDQRF
from SIR_models.sir_simulate import SIRSimulator
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
def obj(x, params):
    params['yita'] = x[0]
    params['beta'] = x[1]
    params['tau'] = x[2]

    sir_model = STEAYDQRF(**params)

    simulator = SIRSimulator(sir_model)
    step_n = params['data'].shape[0] - 1
    y0_dict = params['data'].iloc[0,:].to_dict()
    simulated_df = simulator._run(step_n, y0_dict, params['population'])
    # simulated_df = simulated_df.iloc[1:, :]
    real_df = params['data']
    # scale_df = real_df.max(axis=0)
    weights = {key: 1.0 for key in "STEAYDQRF"}
    # weights['S'] = 1000
    # weights['T'] = 1000
    weights['A'] = 1000
    weights['Q'] = 1000
    weights['F'] = 1000
    weights = pd.Series(weights)
    diff = (real_df - simulated_df).abs() * weights

    return diff.sum().sum()

def test_de():
    # file_name = '../Data/covasim_data/covasim_50000_0.1_0.1.joblib'
    path = '../Data/covasim_data/'
    population = 200000
    test_prob = 0.1
    trace_prob = 0.3
    keep_d = True
    retrain = False
    dynamic = True
    chi_type = 'constant'
    case_name = get_case_name(population, test_prob, trace_prob, keep_d, dynamic=dynamic, chi_type=chi_type)

    params = DF.load_covasim_data(path, population, test_prob, trace_prob, case_name, plot=True)
    params['chi_type'] = chi_type
    fixed_args = (params,)
    bound_w = [(0.1, 0.4), (0, 1), (0, 0.1)]
    result = differential_evolution(obj, bound_w, fixed_args, maxiter=10000, tol=1e-10, strategy='best1exp')

    # evaluate solution
    solution = result['x']
    evaluation = obj(solution, params)

    params['yita'] = solution[0]
    params['beta'] = solution[1]
    params['tau'] = solution[2]

    sir_model = STEAYDQRF(**params)

    simulator = SIRSimulator(sir_model)
    step_n = params['data'].shape[0] - 1
    y0_dict = params['data'].iloc[0, :].to_dict()
    simulated_df = simulator._run(step_n, y0_dict, population)
    real_df = params['data']

    plt.figure(figsize=(16,16))
    for idx, col in enumerate(real_df.columns.to_list()):
        plt.subplot(3, 3, idx + 1)
        plt.plot(real_df[col], '.k', label='ABM')
        plt.plot(simulated_df[col], 'r', label='Fitted')
        plt.title(col)
        plt.legend()
    file_path = '../Data/covasim_data/'
    fig_name = 'fitted_' + case_name + '.png'
    plt.savefig(os.path.join(file_path, fig_name))
    plt.close()

    # save simulated_df
    file_name = 'fitted_' + case_name + '.csv'
    simulated_df.to_csv(os.path.join(file_path, file_name))

    # csv_name = os.path.basename(file_name)
    # csv_name = 'fitted' + csv_name[7:-6] + 'csv'
    # pd.Series({key: params[key] for key in ['yita', 'tau']}).to_csv(file_path + csv_name)




