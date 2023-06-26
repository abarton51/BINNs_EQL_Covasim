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

def D_u(D,dx):
    
    '''
    Create the Matrix operator for (D(u)u_x)_x, where D is a vector of values of D(u),
    and dx is the spatial resolution based on methods from Kurganov and Tadmoor 2000
    (https://www.sciencedirect.com/science/article/pii/S0021999100964593?via%3Dihub)
    '''

    D_ind = np.arange(len(D))

    #first consruct interior portion of D
    #exclude first and last point and include those in boundary
    D_ind = D_ind[1:-1] 
    #Du_mat : du_j/dt = [(D_j + D_{j+1})u_{j+1}
    #                   -(D_{j-1} + 2D_j + D_{j+1})u_j
    #                   + (D_j + D_{j-1})u_{j-1}] 
    Du_mat_row = np.hstack((D_ind,D_ind,D_ind))
    Du_mat_col = np.hstack((D_ind+1,D_ind,D_ind-1))
    Du_mat_entry = (1/(2*dx**2))*np.hstack((D[D_ind+1]+D[D_ind],
                   -(D[D_ind-1]+2*D[D_ind]+D[D_ind+1]),D[D_ind-1]+D[D_ind]))
    
    #boundary points
    Du_mat_row_bd = np.array((0,0,len(D)-1,len(D)-1))
    Du_mat_col_bd = np.array((0,1,len(D)-1,len(D)-2))
    Du_mat_entry_bd = (1.0/(2*dx**2))*np.array((-2*(D[0]+D[1]),
                    2*(D[0]+D[1]),-2*(D[-2]+D[-1]),2*(D[-2]+D[-1])))
    #add in boundary points
    Du_mat_row = np.hstack((Du_mat_row,Du_mat_row_bd))
    Du_mat_col = np.hstack((Du_mat_col,Du_mat_col_bd))
    Du_mat_entry = np.hstack((Du_mat_entry,Du_mat_entry_bd))

    return sparse.coo_matrix((Du_mat_entry,(Du_mat_row,Du_mat_col)))


def PDE_RHS(t,y,x,D,f):
    
    ''' 
    Returns a RHS of the form:
    
        q[0]*(g(u)u_x)_x + q[1]*f(u)
        
    where f(u) is a two-phase model and q[2] is carrying capacity
    '''
    
    dx = x[1] - x[0]
    
    try:
        
        # density and time dependent diffusion
        Du_mat = D_u(D(y,t),dx)
        return  Du_mat.dot(y) + y*f(y,t)
    
    except:
        
        # density dependent diffusion
        Du_mat = D_u(D(y),dx)
        return  Du_mat.dot(y) + y*f(y)
    
    


def PDE_sim(RHS,IC,x,t,D,f):
    
    # grids for numerical integration
    t_sim = np.linspace(np.min(t), np.max(t), 1000)
    x_sim = np.linspace(np.min(x), np.max(x), 200)
    
    # interpolate initial condition to new grid
    f_interpolate = interpolate.interp1d(x,IC)
    y0 = f_interpolate(x_sim)
        
    # indices for integration to write to file for
    for tp in t:
        tp_ind = np.abs(tp-t_sim).argmin()
        if tp == t[0]:
            t_sim_write_ind = np.array(tp_ind)
        else:
            t_sim_write_ind = np.hstack((t_sim_write_ind,tp_ind))

    # make RHS a function of t,y
    def RHS_ty(t,y):
        return RHS(t,y,x_sim,D,f)
            
    # initialize array for solution
    y = np.zeros((len(x),len(t)))  
    
    y[:, 0] = IC
    write_count = 0
    r = integrate.ode(RHS_ty).set_integrator("dopri5")  # choice of method
    r.set_initial_value(y0, t[0])   # initial values
    for i in range(1, t_sim.size):
        
        # write to y for write indices
        if np.any(i==t_sim_write_ind):
            write_count+=1
            f_interpolate = interpolate.interp1d(x_sim,r.integrate(t_sim[i]))
            y[:,write_count] = f_interpolate(x)
        else:
            # otherwise just integrate
            r.integrate(t_sim[i]) # get one more value, add it to the array
        if not r.successful():
            print("integration failed")
            return 1e6*np.ones(y.shape)

    return y


def STEAYDQRF_RHS(t, y, contact_rate, quarantine_test, params, t_max):
    '''
    RHS evaluation of learned components for the STEAYDQRF model.
    
    Inputs:
    t (float): time.
    y (float): 
    contact_rate (func): the contact rate learned MLP in the BINN model.
    quarantine_test (func): the quarantining rate learned MLP in the BINN model.
    params (dict):
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
    # beta = params['beta']
    gamma = params['gamma']
    mu = params['mu']
    # tau = params['tau'] / 4
    lamda = params['lamda']
    p_asymp = params['p_asymp']
    n_contacts = params['n_contacts']
    delta = params['delta']

    eff_ub = params['eff_ub']
    # chi = (eff_ub / 30) * (t + 1) if t < 30.0 else  eff_ub
    # chi = eff_ub
    # chi = 0.3 *  (1 + np.sin((t + 1) * np.pi / 40.)) / 2
    chi = chi_func(t, chi_type)
    # get contact rates from learned MLP
    array = y[None, :][:, [0, 3, 4]].reshape(1,-1) # , chi
    cr = contact_rate(array).reshape(-1)
    yita = params['yita_lb'] + (params['yita_ub'] - params['yita_lb']) * cr[0]
    yq_array = np.append(y[None, :][:,[0, 3, 4]].sum(axis=1, keepdims=True), chi).reshape(1,-1)
    # yq_array = y[None, :][:, [4, 6]].reshape(1,-1)
    beta0 = quarantine_test(yq_array).reshape(-1)
    # beta = params['beta_lb'] + (params['beta_ub'] - params['beta_lb']) * beta0
    beta = chi * beta0
    # print(beta)

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

    # print(np.array([ds, dt, de, da, dy, dd, dq, dr, df]).sum())

    return np.array([ds, dt, de, da, dy, dd, dq, dr, df])


def STEAYDQRF_sim(RHS, IC, t, contact_rate, quarantine_test, tau, params, chi_type):
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
    
    Returns:
        y (array): numpy array of values of each term in STEAYDQRF.
    '''
    # grids for numerical integration
    t_sim = np.linspace(np.min(t), np.max(t), 1000)
    t_max = np.max(t)
    # x_sim = np.linspace(np.min(x), np.max(x), 200)
    #
    # # interpolate initial condition to new grid
    # f_interpolate = interpolate.interp1d(x, IC)
    # y0 = f_interpolate(x_sim)

    # indices for integration to write to file for
    for tp in t:
        tp_ind = np.abs(tp - t_sim).argmin()
        if tp == t[0]:
            t_sim_write_ind = np.array(tp_ind)
        else:
            t_sim_write_ind = np.hstack((t_sim_write_ind, tp_ind))

    # make RHS a function of t,y
    def RHS_ty(t, y):
        return RHS(t, y, contact_rate, quarantine_test, tau, params, t_max, chi_type)

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


def STEAYQRF_RHS(t, y, contact_rate, quarantine_test, params, t_max):

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
    tau = params['tau_lb'] + (params['tau_ub'] - params['tau_lb']) * quarantine_test(y[None, :][:, [3, 4]])
    # current compartment values
    s, tq, e, a, y, q, r, f = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]
    new_d = mu * y +  tau * q
    # dS
    ds = - yita * s * (a + y) -  beta * new_d *  n_contacts * s +  alpha * tq
    # dT
    dt =  beta * new_d *  n_contacts * s -  alpha * tq

    # dE
    de = yita * s  * (a + y) -  gamma * e

    # dA
    da =  p_asymp *  gamma * e -  lamda * a -  beta * new_d *  n_contacts * a

    # dY
    dy = (1 -  p_asymp) *  gamma * e - ( mu +  lamda +  delta) * y -  beta * new_d *  n_contacts * y

    # dQ
    dq =  beta * new_d *  n_contacts * (a + y) - ( delta +  lamda) * q  + mu * y

    # dR
    dr =  lamda * (a + y + q)

    # dF
    df =  delta * (y + q)

    # print(np.array([ds, dt, de, da, dy, dd, dq, dr, df]).sum())

    return np.array([ds, dt, de, da, dy, dq, dr, df])


def STEAYQRF_sim(RHS, IC, t, contact_rate, quarantine_test, params):
    # grids for numerical integration
    t_sim = np.linspace(np.min(t), np.max(t), 1000)
    t_max = np.max(t)
    # x_sim = np.linspace(np.min(x), np.max(x), 200)
    #
    # # interpolate initial condition to new grid
    # f_interpolate = interpolate.interp1d(x, IC)
    # y0 = f_interpolate(x_sim)

    # indices for integration to write to file for
    for tp in t:
        tp_ind = np.abs(tp - t_sim).argmin()
        if tp == t[0]:
            t_sim_write_ind = np.array(tp_ind)
        else:
            t_sim_write_ind = np.hstack((t_sim_write_ind, tp_ind))

    # make RHS a function of t,y
    def RHS_ty(t, y):
        return RHS(t, y, contact_rate, quarantine_test, params, t_max)

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

def SEPAYRHD_sim(RHS, IC, t, contact_rate, quarantine_test):
    # grids for numerical integration
    t_sim = np.linspace(np.min(t), np.max(t), 1000)
    t_max = np.max(t)
    # x_sim = np.linspace(np.min(x), np.max(x), 200)
    #
    # # interpolate initial condition to new grid
    # f_interpolate = interpolate.interp1d(x, IC)
    # y0 = f_interpolate(x_sim)

    # indices for integration to write to file for
    for tp in t:
        tp_ind = np.abs(tp - t_sim).argmin()
        if tp == t[0]:
            t_sim_write_ind = np.array(tp_ind)
        else:
            t_sim_write_ind = np.hstack((t_sim_write_ind, tp_ind))

    # make RHS a function of t,y
    def RHS_ty(t, y):
        return RHS(t, y, contact_rate, quarantine_test, t_max)

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

def SEPAYRHD_RHS(t, y, contact_rate, varying_parameters, t_max):

    population = 10.5e6
    gamma = 1 / 1.75
    lamda = 2  # /0.5
    epsilon = 1 / 2
    theta = 1 / 3
    symp_prob = 0.67
    yita_lb = 0.0
    yita_ub = 0.1
    beta_lb = 0.0
    beta_ub = 1e-4
    h_rate_lb = 0.0
    h_rate_ub = 1

    # get contact rates from learned MLP
    s, i, h = y[0], y[2:5].sum(), y[6]
    cat_array = np.array([t / t_max, s, i, h])
    cr = contact_rate(cat_array[None, :]).reshape(-1)
    eta = yita_lb + (yita_ub - yita_lb) * cr[0]
    qt = varying_parameters(np.array([t])[None, :]).reshape(-1)
    h_rate, d_rate, vacc_rate = qt[0], qt[1], qt[2]

    beta = beta_lb + (beta_ub - beta_lb) * vacc_rate
    h_rate_scaled = h_rate_lb + (h_rate_ub - h_rate_lb) * h_rate
    # get time-dependent parameters
    alpha = alpha_factor(t)
    mu = mu_factor(t)
    # current compartment values
    s, e, p, a, y, r, h, d = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]

    i = p + a + y
    # dS
    ds = - eta * s * i  + alpha * r - beta * s

    # dE
    de =  eta * s * i - gamma * e

    # dP
    dp = gamma * e - lamda * p

    # dA
    da =  (1 - symp_prob) * lamda * p - epsilon * a

    # dY
    dy = symp_prob * lamda * p - theta * y

    # dR
    dr =  epsilon * a + (1 - h_rate_scaled) * theta * y + beta * s + \
                      (1 - d_rate) * mu * h - alpha * r

    # dH
    dh =  h_rate_scaled * theta * y - mu * h

    # dD
    dd =  d_rate * mu * h

    # print(np.array([ds, dt, de, da, dy, dd, dq, dr, df]).sum())

    return np.array([ds, de, dp, da, dy, dr, dh, dd])

def alpha_factor(t):
    # immunity escape at days 150, 157, 164 and 171
    peak_tensor = (0.8 * 0.25 + 0.2/100) * np.ones_like(t)
    alpha_tensor = np.where(np.isclose(t, 150.0 * np.ones_like(t)), peak_tensor, 0.2/100 * np.ones_like(t))
    alpha_tensor = np.where(np.isclose(t, 157.0 * np.ones_like(t)), peak_tensor , alpha_tensor)
    alpha_tensor = np.where(np.isclose(t, 164.0 * np.ones_like(t)), peak_tensor, alpha_tensor)
    alpha_tensor = np.where(np.isclose(t, 171.0 * np.ones_like(t)), peak_tensor, alpha_tensor)
    return alpha_tensor

def mu_factor(t):
    # dynamic time in hospital reduce by 42% over days 150-170
    return np.where(np.logical_and(150.0 <= t, t < 170.0), 1.0/(10.4 * 0.58), 1.0/10.4)
