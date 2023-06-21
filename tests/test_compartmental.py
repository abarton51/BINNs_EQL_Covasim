import joblib

from SIR_models.sir_models import STEAYDQRF
from SIR_models.sir_simulate import SIRSimulator
import Modules.Loaders.DataFormatter as DF
from Notebooks.utils import get_case_name
import matplotlib.pyplot as plt
import numpy as np

def test_SEIR():
    path = '../Data/covasim_data/'
    population = 200000
    test_prob = 0.1
    trace_prob = 0.3
    keep_d = True
    retrain = False
    dynamic = True
    chi_type = 'piecewise'
    case_name = get_case_name(population, test_prob, trace_prob, keep_d, dynamic=dynamic, chi_type=chi_type)

    params = DF.load_covasim_data(path, population, test_prob, trace_prob, case_name, plot=True)

    # manually assign theta, yita, epsilon
    params['yita'] = 0.4
    params['beta'] = 0.5
    params['tau'] = 0.05
    params['chi_type'] = chi_type
    # params.pop('population')
    sir_model = STEAYDQRF(**params)

    simulator = SIRSimulator(sir_model)
    step_n = params['data'].shape[0] - 1
    y0_dict = params['data'].iloc[0,:].to_dict()
    simulated_df = simulator._run(step_n, y0_dict, population)
    # simulated_df = simulated_df.iloc[1:, :]
    real_df = params['data']
    scale_df = real_df.max(axis=0)
    diff = (real_df - simulated_df).abs() / scale_df
    diff.sum().sum()

    plot = True
    if plot:
        # data = params['data']
        n = real_df.shape[1]
        col_names = list('STEAYDQRF') if keep_d else list('STEAYQRF')
        t = np.arange(len(real_df))
        # t = np.arange(1, data.shape[0] + 1)
        # plot compartments
        fig = plt.figure(figsize=(15, 15))
        for i in range(1, n + 1):
            ax = fig.add_subplot(int(np.ceil(n / 3)), 3, i)
            ax.plot(t, real_df.iloc[:, i - 1], '.k', label='Covasim Data')
            ax.plot(t, simulated_df.iloc[:, i - 1], '-*r', label='Simulated')
            ax.set_title(col_names[i - 1])
            ax.legend(fontsize=8)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.tight_layout(pad=2)
