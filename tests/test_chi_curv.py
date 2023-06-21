from tests.test_covasim_data_generator_dynamic import get_dynamic_eff
from Modules.Models.BuildBINNs import chi
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import torch
matplotlib.use('Agg')
plt.rcParams.update({'font.size': 12})
def test_chi_curv():

    chi_type = ['constant', 'piecewise', 'sin',] #
    chi_str = {'piecewise': 'Piecewise', 'constant': 'Constant', 'sin': 'Sinusoidal'}
    colors = ['r', 'b', 'k']
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1)
    for idx, ct in enumerate(chi_type):
        chi_array = get_dynamic_eff(ct, 0.3)
        t = np.arange(0, len(chi_array), 1)
        ax.plot(t, chi_array, colors[idx], label = chi_str[ct])
    ax.set_xlabel('Days')
    ax.set_ylabel(r'$h(t)$')
    ax.legend()
    plt.savefig(os.path.join('chi_curvs' + '.png'), dpi=300)
    plt.close()

    # chi_type = ['Piecewise', 'Constant', 'sin',]  #

    # colors = ['r', 'b', 'k']
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(1, 1, 1)
    # for idx, ct in enumerate(chi_type):
    #     t = torch.arange(0, 200, 1)
    #     chi_array = chi(t, 0.3, ct)
    #     ax.plot(t, chi_array, colors[idx], label=chi_str[ct])
    # ax.set_xlabel('Days')
    # ax.set_ylabel(r'$h(t)$')
    # ax.legend()
    # plt.savefig(os.path.join('chi_curvs_BINN' + '.png'), dpi=300)
    # plt.close()
