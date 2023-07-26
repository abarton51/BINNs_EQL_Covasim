import numpy as np

from scipy import integrate
from scipy import sparse
from scipy import interpolate
from scipy.stats import beta
import os
import scipy.io as sio
import scipy.optimize
import itertools
import time

import pdb

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from sklearn.preprocessing import PolynomialFeatures

def STEAYDQRF_RHS(t, y, contact_rate, quarantine_test, params, t_max):
    '''
    RHS evaluation of learned components for the STEAYDQRF model.
    
    Inputs:
    t (float): time.
    y (float): 
    contact_rate (func): the contact rate learned MLP in the BINN model.
    quarantine_test (func): the quarantining rate learned MLP in the BINN model.
    params (dict): parameters of the Covasim model.
    t_max (float): the maximum value of time in the t array.
    '''

    population = params['population']
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    mu = params['mu']
    # tau = params['tau']
    lamda = params['lamda']
    p_asymp = params['p_asymp']
    n_contacts = params['n_contacts']
    delta = params['delta']

    # get contact rates from learned MLP
    cr = contact_rate(y[None, :][:, [0, 3, 4]]).reshape(-1)
    yita = params['yita_lb'] + (params['yita_ub'] - params['yita_lb']) * cr[0]
    qt = quarantine_test(y[None, :][:, [3, 4]]).reshape(-1)
    tau = params['tau_lb'] + (params['tau_ub'] - params['tau_lb']) * qt
    # current compartment values
    s, tq, e, a, y, d, q, r, f = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8]
    new_d = mu * y +  tau * q
    # dS
    ds = - yita * s * (a + y) -  beta * new_d *  n_contacts * s +  alpha * tq

    # dT
    dt =  beta * new_d *  n_contacts * s -  alpha * tq

    # dE
    de = yita * s * (a + y) -  gamma * e

    # dA
    da =  p_asymp *  gamma * e -  lamda * a -  beta * new_d *  n_contacts * a

    # dY
    dy = (1 -  p_asymp) *  gamma * e - ( mu +  lamda +  delta) * y -  beta * new_d *  n_contacts * y

    # dD
    dd =  new_d -  lamda * d - delta * d

    # dQ
    dq =  beta * new_d *  n_contacts * (a + y) - (tau +  lamda + delta) * q

    # dR
    dr =  lamda * (a + y + d + q)

    # dF
    df =  delta * (y + d + q)

    # print(np.array([ds, dt, de, da, dy, dd, dq, dr, df]).sum())

    return np.array([ds, dt, de, da, dy, dd, dq, dr, df])

def chi_func(t, chi_type):
    eff_ub = 0.3
    if chi_type == 'linear':
        rate = eff_ub / 75
        if t < 75:
            factor = rate * (t + 1)
        elif 75 <= t < 150:
            factor = eff_ub - rate * (t - 75 + 1)
        else:
            factor = 0
        # factor = torch.where(t < 30.0, rate * t, eff_ub * torch.ones_like(t))
    elif chi_type == 'sin':
        rad_times = t * np.pi / 40.
        factor = 0.3 * (1 + np.sin(rad_times)) / 2
    elif chi_type == 'piecewise':
        a, b = 3, 3
        t_max = 159
        max_val = beta.pdf(0.5, a, b, loc=0, scale=1)
        if t < 80:
            factor = beta.pdf(t / t_max, a, b, loc=0, scale=1) * eff_ub / max_val
        elif t >= 120:
            factor = beta.pdf((t - 40) / t_max, a, b, loc=0, scale=1) * eff_ub / max_val
        else:
            factor = eff_ub
    elif chi_type == 'constant':
        factor = eff_ub
    return factor

def STEAYDQRF_RHS_dynamic(t, y, contact_rate, quarantine_test, tau_func, params, t_max, chi_type):
    '''
    RHS evaluation of learned components for the STEAYDQRF model.
    
    Args:
        t (array): time vector.
        y (array): vector of values of STEAYDQRF.
        contact_rate (func): the contact rate learned MLP in the BINN model.
        quarantine_test (func): the quarantining rate learned MLP in the BINN model.
        tau_func (func): the quarantine diagnoses rate learned MLP in the BINN model.
        params (dict): paramters of COVASIM model.
        t_max (float): the maximum value of time in the t array.
        chi_type (str): string indicated the type of function chi is.
    
    Returns:
        (array): numpy array of values of each differential term in the ODE system.
    '''

    population = params['population']
    alpha = params['alpha']
    gamma = params['gamma']
    mu = params['mu']
    lamda = params['lamda']
    p_asymp = params['p_asymp']
    n_contacts = params['n_contacts']
    delta = params['delta']

    eff_ub = params['eff_ub']
    chi = chi_func(t, chi_type)
    # get contact rates from learned MLP
    array = y[None, :][:, [0, 3, 4]].reshape(1,-1) # , chi
    cr = contact_rate(array).reshape(-1)
    yita = params['yita_lb'] + (params['yita_ub'] - params['yita_lb']) * cr[0]
    yq_array = np.append(y[None, :][:,[0, 3, 4]].sum(axis=1, keepdims=True), chi).reshape(1,-1)
    beta0 = quarantine_test(yq_array).reshape(-1)
    beta = chi * beta0
    
    ay_array = y[None, :][:, [3, 4]].reshape(1,-1)
    tau0 = tau_func(ay_array)
    tau = params['tau_lb'] + (params['tau_ub'] - params['tau_lb']) * tau0
    # current compartment values
    s, tq, e, a, y, d, q, r, f = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8]
    new_d = mu * y +  tau * q
    # dS
    ds = - yita * s * (a + y) - beta * new_d *  n_contacts * s + alpha * tq

    # dT
    dt =  beta * new_d *  n_contacts * s - alpha * tq

    # dE
    de = yita * s * (a + y) - gamma * e

    # dA
    da =  p_asymp * gamma * e - lamda * a - beta * new_d *  n_contacts * a

    # dY
    dy = (1 - p_asymp) * gamma * e - (mu + lamda + delta) * y - beta * new_d *  n_contacts * y

    # dD
    dd =  mu * y + tau * q - lamda * d - delta * d

    # dQ
    dq =  beta * new_d *  n_contacts * (a + y) - (tau + delta) * q

    # dR
    dr =  lamda * (a + y + d ) #

    # dF
    df =  delta * (y + d + q) #

    return np.array([ds, dt, de, da, dy, dd, dq, dr, df])

def STEAYDQRF_RHS_dynamic(t, y, contact_rate, quarantine_test, tau_func, params, t_max, chi_type, regression=False):
    '''
    RHS evaluation of learned components for the STEAYDQRF model.
    
    Args:
        t (array): time vector.
        y (array): vector of values of STEAYDQRF.
        contact_rate (func): the contact rate learned MLP in the BINN model.
        quarantine_test (func): the quarantining rate learned MLP in the BINN model.
        tau_func (func): the quarantine diagnoses rate learned MLP in the BINN model.
        params (dict): paramters of COVASIM model.
        t_max (float): the maximum value of time in the t array.
        chi_type (str): string indicated the type of function chi is.
    
    Returns:
        (array): numpy array of values of each differential term in the ODE system.
    '''

    population = params['population']
    alpha = params['alpha']
    gamma = params['gamma']
    mu = params['mu']
    lamda = params['lamda']
    p_asymp = params['p_asymp']
    n_contacts = params['n_contacts']
    delta = params['delta']

    eff_ub = params['eff_ub']
    chi = chi_func(t, chi_type)
    # get contact rates from learned MLP
    if not regression:
        eta_input = y[None, :][:, [0, 3, 4]].reshape(1,-1) # , chi
    else:
        eta_input = y[None, :].reshape(1,-1)
    eta = contact_rate(eta_input).reshape(-1)
    yita = params['yita_lb'] + (params['yita_ub'] - params['yita_lb']) * eta[0]
    
    if not regression:
        beta_input = np.append(y[None, :][:,[0, 3, 4]].sum(axis=1, keepdims=True), chi).reshape(1,-1)
    else:
        beta_input = y[None, :].reshape(1, -1)
    beta0 = quarantine_test(beta_input).reshape(-1)
    beta = chi * beta0
    
    if not regression:
        tau_input = y[None, :][:, [3, 4]].reshape(1,-1)
    else:
        tau_input = y[None, :].reshape(1, -1)
    tau0 = tau_func(tau_input)
    tau = params['tau_lb'] + (params['tau_ub'] - params['tau_lb']) * tau0
    # current compartment values
    s, tq, e, a, y, d, q, r, f = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8]
    new_d = mu * y +  tau * q
    # dS
    ds = - yita * s * (a + y) - beta * new_d *  n_contacts * s + alpha * tq

    # dT
    dt =  beta * new_d *  n_contacts * s - alpha * tq

    # dE
    de = yita * s * (a + y) - gamma * e

    # dA
    da =  p_asymp * gamma * e - lamda * a - beta * new_d *  n_contacts * a

    # dY
    dy = (1 - p_asymp) * gamma * e - (mu + lamda + delta) * y - beta * new_d *  n_contacts * y

    # dD
    dd =  mu * y + tau * q - lamda * d - delta * d

    # dQ
    dq =  beta * new_d *  n_contacts * (a + y) - (tau + delta) * q

    # dR
    dr =  lamda * (a + y + d ) #

    # dF
    df =  delta * (y + d + q) #

    return np.array([ds, dt, de, da, dy, dd, dq, dr, df])

def STEAYDQRF_RHS_dynamic_maskarr(t, 
                                  y, 
                                  eta_func, 
                                  beta_func, 
                                  tau_func, 
                                  params, 
                                  t_max, 
                                  chi_type, 
                                  regression=False,
                                  eta_all_comps=False,
                                  beta_all_comps=False,
                                  tau_all_comps=False,
                                  eta_degree=-1,
                                  beta_degree=-1,
                                  tau_degree=-1):
    '''
    RHS evaluation of learned components for the STEAYDQRF model.
    
    Args:
        t (array): time vector.
        y (array): vector of values of STEAYDQRF.
        contact_rate (func): the contact rate learned MLP in the BINN model.
        quarantine_test (func): the quarantining rate learned MLP in the BINN model.
        tau_func (func): the quarantine diagnoses rate learned MLP in the BINN model.
        params (dict): paramters of COVASIM model.
        t_max (float): the maximum value of time in the t array.
        chi_type (str): string indicated the type of function chi is.
    
    Returns:
        (array): numpy array of values of each differential term in the ODE system.
    '''

    population = params['population']
    alpha = params['alpha']
    gamma = params['gamma']
    mu = params['mu']
    lamda = params['lamda']
    p_asymp = params['p_asymp']
    n_contacts = params['n_contacts']
    delta = params['delta']
    avg_masking = params['avg_masking']
    eff_ub = params['eff_ub']
    
    chi = chi_func(t, chi_type)
    # get contact rates from learned MLP
    # eta
    '''
    if not regression:
        eta_input = y[None, :][:, [0, 3, 4]].reshape(1,-1)
        if int(t * t_max) < 183:
            eta_input = np.append(eta_input, avg_masking[int(t * t_max)]).reshape(1, -1)
        else:
            eta_input = np.append(eta_input, avg_masking[-1]).reshape(1, -1)
    else:
        eta_input = y[None, :].reshape(1,-1)
        if int(t * t_max) < 183:
            eta_input = np.append(eta_input, avg_masking[int(t * t_max)]).reshape(1, -1)
        else:
            eta_input = np.append(eta_input, avg_masking[-1]).reshape(1, -1)
    '''
    eta_input = y[None, :][:, [0, 3, 4]].reshape(1,-1) if not eta_all_comps else y[None, :].reshape(1,-1)
    if int(t * t_max) < 183:
        eta_input = np.append(eta_input, avg_masking[int(t * t_max)]).reshape(1, -1)
    else:
        eta_input = np.append(eta_input, avg_masking[-1]).reshape(1, -1)
    if eta_degree==-1:
        cr = eta_func(eta_input).reshape(-1)
    else:
        poly = PolynomialFeatures(eta_degree)
        eta_input = poly.fit_transform(eta_input)
        cr = eta_func(eta_input).reshape(-1)
    yita = params['yita_lb'] + (params['yita_ub'] - params['yita_lb']) * cr[0]
    
    # beta
    '''
    if not regression:
        beta_input = np.append(y[None, :][:,[0, 3, 4]].sum(axis=1, keepdims=True), chi).reshape(1,-1)
    else:
        beta_input = y[None, :].reshape(1, -1)
    '''
    if not beta_all_comps:
        beta_input = np.append(y[None, :][:,[0, 3, 4]].sum(axis=1, keepdims=True), chi).reshape(1,-1)
    else:
        beta_input = y[None, :].reshape(1, -1)
    if beta_degree==-1:
        beta0 = beta_func(beta_input).reshape(-1)
    else:
        poly = PolynomialFeatures(beta_degree)
        beta_input = poly.fit_transform(beta_input)
        beta0 = beta_func(beta_input).reshape(-1)
    beta = chi * beta0
    
    # tau
    '''
    if not regression:
        tau_input = y[None, :][:, [3, 4]].reshape(1,-1)
    else:
        tau_input = y[None, :].reshape(1, -1)
    '''
    tau_input = y[None, :][:, [3, 4]].reshape(1,-1) if not tau_all_comps else y[None, :].reshape(1,-1)
    if tau_degree==-1:
        tau0 = tau_func(tau_input)
    else:
        poly = PolynomialFeatures(tau_degree)
        tau_input = poly.fit_transform(tau_input)
        tau0 = tau_func(tau_input)
    tau = params['tau_lb'] + (params['tau_ub'] - params['tau_lb']) * tau0
    
    # current compartment values
    s, tq, e, a, y, d, q, r, f = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8]
    new_d = mu * y +  tau * q
    # dS
    ds = - yita * s * (a + y) - beta * new_d *  n_contacts * s + alpha * tq

    # dT
    dt =  beta * new_d *  n_contacts * s - alpha * tq

    # dE
    de = yita * s * (a + y) - gamma * e

    # dA
    da =  p_asymp * gamma * e - lamda * a - beta * new_d *  n_contacts * a

    # dY
    dy = (1 - p_asymp) * gamma * e - (mu + lamda + delta) * y - beta * new_d *  n_contacts * y

    # dD
    dd =  mu * y + tau * q - lamda * d - delta * d

    # dQ
    dq =  beta * new_d *  n_contacts * (a + y) - (tau + delta) * q

    # dR
    dr =  lamda * (a + y + d ) #

    # dF
    df =  delta * (y + d + q) #

    return np.array([ds, dt, de, da, dy, dd, dq, dr, df])

def STEAYDQRF_sim(RHS, 
                  IC, 
                  t, 
                  eta_func, 
                  beta_func, 
                  tau_func, 
                  params, 
                  chi_type, 
                  regression=False,
                  eta_all_comps=False,
                  beta_all_comps=False,
                  tau_all_comps=False,
                  eta_degree=-1,
                  beta_degree=-1,
                  tau_degree=-1):
    '''
    Simulator for the STEAYDQRF model using numerical integration.
    
    Args:
        RHS (array): array of derivative values of ODE system.
        IC (array): initial conditions vector.
        contact_rate (func): the contact rate learned MLP in the BINN model.
        quarantine_test (func): the quarantining rate learned MLP in the BINN model.
        tau_func (func): the quarantine diagnoses rate learned MLP in the BINN model.
        params (dict):
        chi_type (str): string indicating the type of function chi is.
        regression (bool): boolean indicating if the parameters are NN are linear models.
    
    Returns:
        y (array): numpy array of values of each term in STEAYDQRF.
    '''
    # grids for numerical integration
    t_max = np.max(t)
    t_sim = np.linspace(np.min(t), t_max, 1000)

    # indices for integration to write to file for
    for tp in t:
        tp_ind = np.abs(tp - t_sim).argmin()
        if tp == t[0]:
            t_sim_write_ind = np.array(tp_ind)
        else:
            t_sim_write_ind = np.hstack((t_sim_write_ind, tp_ind))

    # make RHS a function of t,y
    def RHS_ty(t, y):
        return RHS(t, 
                   y, 
                   eta_func, 
                   beta_func, 
                   tau_func, 
                   params, 
                   t_max, 
                   chi_type, 
                   regression,
                   eta_all_comps,
                   beta_all_comps,
                   tau_all_comps, 
                   eta_degree,
                   beta_degree,
                   tau_degree)

    # initialize array for solution
    y = np.zeros((len(t), len(IC)))

    y[0,:] = IC
    
    write_count = 0
    r = integrate.ode(RHS_ty).set_integrator("dopri5")  # choice of method
    r.set_initial_value(y[0,:], t[0])  # initial values
    
    for i in range(1, t_sim.size):

        # write to y for write indices
        if np.any(i == t_sim_write_ind):
            write_count += 1
            y[write_count, :] = r.integrate(t_sim[i])
        else:
            # otherwise just integrate
            r.integrate(t_sim[i])  # get one more value, add it to the array
        if not r.successful():
            print("integration failed")
            return 1e6 * np.ones(y.shape)

    return y