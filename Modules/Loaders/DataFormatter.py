import pandas as pd
import torch, pdb
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import joblib

def load_cell_migration_data(file_path, initial_density, plot=False):
    
    densities = ['dens_10000', 'dens_12000', 'dens_14000', 
                 'dens_16000', 'dens_18000', 'dens_20000']
    density = densities[initial_density]
    
    # load data
    file = np.load(file_path, allow_pickle=True).item()

    # extract data
    density = densities[initial_density]
    x = file[density]['x'].copy()[1:, :] 
    t = file[density]['t'].copy()
    X = file[density]['X'].copy()[1:, :]
    T = file[density]['T'].copy()[1:, :]
    U = file[density]['U_mean'].copy()[1:, :]
    shape = U.shape

    # variable scales
    x_scale = 1/1000 # micrometer -> millimeter
    t_scale = 1/24 # hours -> days
    u_scale = 1/(x_scale**2) # cells/um^2 -> cells/mm^2

    # scale variables
    x *= x_scale
    t *= t_scale
    X *= x_scale
    T *= t_scale
    U *= u_scale

    # flatten for MLP
    inputs = np.concatenate([X.reshape(-1)[:, None],
                             T.reshape(-1)[:, None]], axis=1)
    outputs = U.reshape(-1)[:, None]

    if plot:
    
        # plot surface
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(X, T, U, cmap=cm.coolwarm, alpha=1)
        ax.scatter(X.reshape(-1), T.reshape(-1), U.reshape(-1), s=5, c='k')
        plt.title('Initial density: '+density[5:])
        ax.set_xlabel('Position (millimeters)')
        ax.set_ylabel('Time (days)')
        ax.set_zlabel('Cell density (cells/mm^2)')
        ax.set_zlim(0, 2.2e3)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.tight_layout(pad=2)
        plt.show()
        
    return inputs, outputs, shape

def load_covasim_data(file_path, population, test_prob, trace_prob, keep_d, case_name, plot=True):

    # file_name = '_'.join(['covasim', str(population), str(test_prob), str(trace_prob)])
    # if not keep_d:
    #     file_name += '_' + 'noD'
    # if dynamic:
    #     file_name += '_' + 'dynamic'
    file_name = 'covasim_' + str(case_name)
    params = joblib.load(file_path + file_name + '.joblib')

    if plot and isinstance(params['data'], pd.DataFrame):
        data = params['data']
        n = data.shape[1]
        col_names = list(data.columns)
        t = np.arange(1, data.shape[0] + 1)
        # plot compartments
        fig = plt.figure(figsize=(10, 7))
        for i in range(1, n + 1):
            ax = fig.add_subplot(int(np.ceil(n / 3)), 3, i)
            ax.plot(t, data.iloc[:, i - 1], '.-', label=col_names[i - 1])
            ax.legend()
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.tight_layout(pad=2)
        plt.savefig(file_path + file_name + '.png')
        plt.close()
    if plot and isinstance(params['data'], list):
        data = params['data']
        n = data[0].shape[1]
        col_names = list(data[0].columns)
        t = np.arange(1, data[0].shape[0] + 1)
        # plot compartments
        fig = plt.figure(figsize=(10, 7))
        for i in range(1, n + 1):
            ax = fig.add_subplot(int(np.ceil(n / 3)), 3, i)
            for j in range(len(data)):
                ax.plot(t, data[j].iloc[:, i - 1], '.-', label=col_names[i - 1] if j == 0 else '')
            ax.legend()
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.tight_layout(pad=2)
        plt.savefig(file_path + file_name + '.png')
        plt.close()
    return params

def load_covasim_data_drums(file_path, population, keep_d, case_name, plot=True):
    '''
    Load covasim simulation data from .joblib file into dictionary/
    
    Args:
        file_path (str): name of the file path
        population (int): number of agents in population
        keep_d (bool): boolean value indicating whether or not to include D (diagnosed) in model
        case_name (str): case name of the simulation
        plot (bool): whether or not to plot simulation data
    
    Returns:
        params (dict): dictionary with values for each parameter of dataset
    '''

    file_name = 'covasim_' + str(case_name)
    params = joblib.load(file_path + file_name + '.joblib')

    if plot and isinstance(params['data'], pd.DataFrame):
        data = params['data']
        n = data.shape[1]
        col_names = list(data.columns)
        t = np.arange(1, data.shape[0] + 1)
        # plot compartments
        fig = plt.figure(figsize=(10, 7))
        for i in range(1, n + 1):
            ax = fig.add_subplot(int(np.ceil(n / 3)), 3, i)
            ax.plot(t, data.iloc[:, i - 1], '.-', label=col_names[i - 1])
            ax.legend()
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.tight_layout(pad=2)
        plt.savefig(file_path + file_name + '.png')
        plt.close()
    if plot and isinstance(params['data'], list):
        data = params['data']
        n = data[0].shape[1]
        col_names = list(data[0].columns)
        t = np.arange(1, data[0].shape[0] + 1)
        # plot compartments
        fig = plt.figure(figsize=(10, 7))
        for i in range(1, n + 1):
            ax = fig.add_subplot(int(np.ceil(n / 3)), 3, i)
            for j in range(len(data)):
                ax.plot(t, data[j].iloc[:, i - 1], '.-', label=col_names[i - 1] if j == 0 else '')
            ax.legend()
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.tight_layout(pad=2)
        plt.savefig(file_path + file_name + '.png')
        plt.close()
    return params
