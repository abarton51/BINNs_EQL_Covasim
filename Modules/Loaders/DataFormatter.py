import pandas as pd
import torch, pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import joblib

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

def load_covasim_data_drums(file_path, population, case_name, keep_d=True, plot=True):
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

    if plot and isinstance(params['data'], pd.DataFrame):   #if plot is true and the data is a pd dataframe
        data = params['data']   #declaring the data from covasim as data
        n = data.shape[1]   #number of rows in data ??
        col_names = list(data.columns)
        t = np.arange(1, data.shape[0] + 1)  #an array of numbers from 1 to number of rows plus 1
        
        
        # plot compartments FOR ONE SIMULATION
        fig = plt.figure(figsize=(10, 7))
        for i in range(1, n + 1):    #range from 1 to the number of rows plus 1
            ax = fig.add_subplot(int(np.ceil(n / 3)), 3, i) #subplot with n divided by 3 and rounded up num rows, 3 num cols and the element goes in the ith place
            ax.plot(t, data.iloc[:, i - 1], '.-', label=col_names[i - 1])   #plot t on x-axis, all rows of data and the i-1 column of data on the y- axis
            ax.legend()
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.tight_layout(pad=2)
        plt.savefig(file_path + file_name + '.png')
        plt.close()

    if plot and isinstance(params['data'], list):   #if plot is true and data is a list? a list of dataframes maybe?
        data = params['data']
        n = data[0].shape[1]    #num rows in the first element of the list
        col_names = list(data[0].columns)
        t = np.arange(1, data[0].shape[0] + 1)  #an array of numbers from 1 to number of rows in first item plus 1
       
       
       ##For averaging the simulations and plotting the average

        matrix_list = [df.values for df in data]    #turning dataframes into matricies
        matrix3d = np.array(matrix_list)    #turning list of matricies into 3d matrix
        mean_mat = np.mean(matrix3d, axis = 0)  #averaging each of the days for each state
        max_mat = np.max(matrix3d, axis = 0)    #getting the max of each of the days for each state
        min_mat = np.min(matrix3d, axis = 0)    #getting the min of each of the days for each state

        num_days= mean_mat.shape[0] #num rows in matrix
        days = range(num_days)  
        num_cols = mean_mat.shape[1]
        col_name = ["S", "T", "E", "A", "Y", "D", "Q", "R", "F"]    #names for the columns
        fig = plt.figure(figsize=(10, 7))
        for i in range(num_cols):    #iterating through each state and plotting them vs days in separate plots
            ax = fig.add_subplot(3, 3, i+1) #subplot with n divided by 3 and rounded up num rows, 3 num cols and the element goes in the ith place
            ax.plot(days, mean_mat[:,i])    #this is for the average line
            ax.fill_between(days, max_mat[:,i], min_mat[:,i], alpha=0.2, label='Error') #this is for the error cloud
            ax.set_title(col_name[i])   #this is to name each plot for their specific state
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.tight_layout(pad=2)
        #plt.savefig(file_path + file_name + '.png')
        #plt.close()
        plt.show()
        
        '''
        # plot compartments EACH SIMULATION ON TOP OF EACH OHTER
        fig = plt.figure(figsize=(10, 7))
        for i in range(1, n + 1):
            ax = fig.add_subplot(int(np.ceil(n / 3)), 3, i)


            for j in range(len(data)):  
                #I think this for loop is for the multi simulation part, it would plot every iteration of each state on thier respective plot
                ax.plot(t, data[j].iloc[:, i - 1], '.-', label=col_names[i - 1] if j == 0 else '')

            ax.legend()
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.tight_layout(pad=2)
        plt.savefig(file_path + file_name + '.png')
        plt.close()

        '''
    return params
