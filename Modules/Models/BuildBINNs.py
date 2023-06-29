import numpy as np
import torch, pdb
import torch.nn as nn

import sys
sys.path.append(sys.path[0] + '\\../')

from Modules.Models.BuildMLP import BuildMLP
from Modules.Activations.SoftplusReLU import SoftplusReLU
from Modules.Utils.Gradient import Gradient

from scipy.stats import beta

#--------------------------------DEMO for adding mu_MLP----------------------------------------#
class main_MLP(nn.Module):
    '''
    Construct MLP surrogate model for the solution of the governing ODE system.
    Includes three hidden layers. The first with 512 neurons and the latter two with 
    256 neurons. All neurons in hidden layers are ReLU-activated. Output
    is softmax-activated to keep predicted values of S,T,E,A,Y,D,Q,R,F between 0 and 1
    and adding up to one since they are dimensionless ratios of the population.

    Inputs:
        num_outputs (int): number of outputs

    Args:
        inputs (torch tensor): time vector, t, with shape (N, 1)

    Returns:
        outputs (torch tensor): predicted u = (S, T, E, A, Y, D, Q, R, F) values with shape (N, 9)
    '''

    def __init__(self, num_outputs):
        super().__init__()
        self.mlp = BuildMLP(
            input_features=1,
            layers=[512, 256, 256, num_outputs],
            activation=nn.ReLU(),
            linear_output=False,
            output_activation=None)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        outputs = self.mlp(inputs)
        outputs = self.softmax(outputs)

        return outputs


class infect_rate_MLP(nn.Module):
    '''
    Construct MLP surrogate model for the contact rate.
    Includes one hidden layer with 256 ReLU-activated neurons. Output
    is sigmoid-activated to keep predicted rates between 0 and 1.

    Inputs:
        N/A

    Args:
        inputs (torch tensor): S, A, Y with shape (N, 3)

    Returns:
        outputs (torch tensor): predicted contact rate, eta(S, A, Y), values with shape (N, 1)
    '''

    def __init__(self):
        super().__init__()
        self.mlp = BuildMLP(
            input_features=3,
            layers=[256, 1],
            activation=nn.ReLU(),
            linear_output=False,
            output_activation=nn.Sigmoid())

    def forward(self, inputs):
        outputs = self.mlp(inputs)

        return outputs

class beta_MLP(nn.Module):
    '''
    Construct MLP surrogate model for the effective tracing rate.
    Includes one hidden layer with 256 ReLU-activated neurons. Output
    is sigmoid-activated to keep predicted tracing rates, beta(S+A+Y), between 0 and 1.

    Inputs:
        N/A

    Args:
        inputs (torch tensor): S+A+Y with shape (N, 1)

    Returns:
        outputs (torch tensor): predicted beta(S+A+Y) values with shape (N, 1)
    '''

    def __init__(self):
        super().__init__()
        self.mlp = BuildMLP(
            input_features=2,
            layers=[256, 1],
            activation=nn.ReLU(),
            linear_output=False,
            output_activation=nn.Sigmoid())

    def forward(self, inputs):
        outputs = self.mlp(inputs)

        return outputs

class tau_MLP(nn.Module):
    '''
    Construct MLP surrogate model for the diagnoses rate of quarantined invidividuals.
    Includes one hidden layer with 256 ReLU-activated neurons. Output
    is sigmoid-activated to keep predicted rates between 0 and 1.

    Inputs:
        N/A

    Args:
        inputs (torch tensor): A, Y values with shape (N, 2)

    Returns:
        outputs (torch tensor): predicted diagnoses rates of quarantined, tau(A, Y), values with shape (N, 1)
    '''

    def __init__(self):
        super().__init__()
        self.mlp = BuildMLP(
            input_features=2,
            layers=[256, 1],
            activation=nn.ReLU(),
            linear_output=False,
            output_activation=nn.Sigmoid())

    def forward(self, inputs):
        outputs = self.mlp(inputs)

        return outputs
    
class mu_MLP(nn.Module):
    '''
    Construct MLP surrogate model for the diagnoses rate of symptomatic individuals.
    Includes one hidden layer with 256 ReLU-activated neurons. Output
    is sigmoid-activated to keep predicted rates between 0 and 1.

    Inputs:
        N/A

    Args:
        inputs (torch tensor): Y, F with shape (N, 2)

    Returns:
        outputs (torch tensor): predicted diagnoses rates of symptomatic, mu(Y, F), values with shape (N, 1)
    '''

    def __init__(self):
        super().__init__()
        self.mlp = BuildMLP(
            input_features=2,
            layers=[256, 1],
            activation=nn.ReLU(),
            linear_output=False,
            output_activation=nn.Sigmoid())

    def forward(self, inputs):
        outputs = self.mlp(inputs)

        return outputs

def chi(t, eff_ub, chi_type):
    '''
    chi(t) function that interacts with tracing rate, beta, to determine probability of transitioning into
    one of two quarantine states, T or Q.
    
    Args:
        t (array): the time points (floats) to evaluate the function at.
        eff_ub (float): the effective upper bound on the value of chi(t).
        chi_type (str): the type of function we wish to model chi(t) as.
    
    Returns:
        factor (array): chi(t).
    '''
    
    if chi_type is None or chi_type == 'linear':
        rate = eff_ub / 75
        res = torch.zeros_like(t)
        res += (t < 75) * rate * (t + 1)
        res += (t >= 75) * (t < 150) * eff_ub
        res -= (t >= 75) * (t < 150) * (rate * (t - 75 + 1))
        factor = res
        
    elif chi_type == 'sin':
        # times = np.arange(0, 300, 1)
        rad_times = t * np.pi / 40.
        factor = 0.3 * (1 + torch.sin(rad_times)) / 2
        
    elif chi_type == 'piecewise':
        factor = torch.zeros_like(t)
        # use pdf of beta distribution
        a, b = 3, 3
        t_max = 159
        max_val = beta.pdf(0.5, a, b, loc=0, scale=1)

        # t < 80
        factor = factor + (t < 80) * torch.Tensor(beta.pdf(t.cpu().detach().numpy() / t_max, a, b, loc=0, scale=1)).to(
            t.device) * eff_ub / max_val

        # t > 120
        factor = factor + (t >= 120) * torch.Tensor(beta.pdf((t.cpu().detach().numpy() - 40) / t_max, a, b, loc=0,
                                                             scale=1)).to(t.device) * eff_ub / max_val

        # otherwise
        factor = factor + (t >= 80) * (t < 120) * eff_ub

    elif chi_type == 'constant':
        factor = eff_ub * torch.ones_like(t)
    return factor

class BINNCovasim_DRUMS(nn.Module):
    '''
    Constructs a biologically-informed neural network (BINN) composed of
    average number of contacts sufficient to transmit infection per unit of time (eta),
    the effective tracing rate (beta), the rate of diagnoses from people in 
    quarantine (tau), and the rate of diagnoses from people who are symptomatic (mu).

    Args:
        params (dict): dictionary of parameters from COVASIM model.
        t_max_real (float): the unscaled maximum time point (t).
        tracing_array (array): array values of tracing probabilities as a function of time (t).
        yita_lb (float): yita lower bound.
        yita_ub (float): yita upper bound.
        keep_d (bool): If true, then include D (diagnosed) in model, otherwise exlcude it.
        chi_type (func): real-valued function of function that affects the quarantining rate.

    '''

    def __init__(self, params, t_max_real, tracing_array, yita_lb=None, yita_ub=None, keep_d=False, chi_type=None):

        super().__init__()

        self.n_com = 9 if keep_d else 8
        # surface fitter
        self.yita_loss = None
        self.yita_lb = yita_lb if yita_lb is not None else 0.2
        self.yita_ub = yita_ub if yita_ub is not None else 0.4
        self.beta_lb = 0.1
        self.beta_ub = 0.3
        self.tau_lb = 0.1
        self.tau_ub =  0.3 #  params['tau_ub']
        self.mu_lb = 0.01
        self.mu_ub = 0.5
        self.surface_fitter = main_MLP(self.n_com)

        # pde functions/components
        self.eta_func = infect_rate_MLP()
        self.beta_func = beta_MLP()
        self.tau_func = tau_MLP()
        self.mu_func = mu_MLP()

        # input extrema
        self.t_min = 0.0
        self.t_max = 1.0
        self.t_max_real = t_max_real # what is max(t) in the real unaltered timescale

        # loss weights
        # initial condition loss weight
        self.IC_weight = 1e1
        # surface loss weight (solution of system of ODEs)
        self.surface_weight = 1e2
        # pde loss weight
        self.pde_weight = 1e4  # 1e4
        
        if keep_d:
            self.weights_c = torch.tensor(np.array([1, 1000, 1, 1000, 1000, 1, 1000, 1, 1000])[None, :], dtype=torch.float) # [1, 1, 1, 1000, 1, 1, 1000, 1, 1000]
        else:
            self.weights_c = torch.tensor(np.array([1, 1, 1, 1, 1, 1000, 1, 1000])[None, :], dtype=torch.float)
        
        self.pde_loss_weight = 1e0
        self.eta_loss_weight = 1e5
        self.tau_loss_weight = 1e5
        self.mu_loss_weight = 1e3

        # proportionality constant
        self.gamma = 0.2

        # number of samples for pde loss
        self.num_samples = 1000

        # name of BINN model
        self.name = 'covasim_fitter'

        self.params = params

        self.population = params['population']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.gamma = params['gamma']
        self.lamda = params['lamda']
        self.p_asymp = params['p_asymp']
        self.n_contacts = params['n_contacts']
        self.delta = params['delta']
        self.tracing_array = tracing_array

        self.keep_d = keep_d
        # if dynamic
        if 'dynamic_tracing' in params:
            self.is_dynamic = True
        self.eff_ub = params['eff_ub']

        self.chi_type = chi_type if chi_type is not None else None
        
        # we comment out these two since they are assumed unknown
        # self.mu = params['mu']
        # self.tau = params['tau'] / 4 if 'tau' in params else None

    def forward(self, inputs):

        # cache input batch for pde loss
        self.inputs = inputs

        return self.surface_fitter(self.inputs)

    def gls_loss(self, pred, true):

        residual = (pred - true) ** 2

        # add weight to initial condition
        residual *= torch.where(self.inputs[:, 0][:, None] == 0,
                                self.IC_weight * torch.ones_like(pred),
                                torch.ones_like(pred))

        # proportional GLS weighting
        residual *= pred.abs().clamp(min=1.0) ** (-self.gamma)

        # apply weights on compartments
        residual *= self.weights_c

        return torch.mean(residual)

    def pde_loss(self, inputs, outputs, return_mean=True):

        #initialize pde_loss
        pde_loss = 0
        
        # unpack inputs (time t) as shape (N, 1)
        t = inputs[:, 0][:, None]

        # partial derivative computations
        u = outputs.clone()
        
        chi_t = chi(1 + t * self.t_max_real, self.eff_ub, self.chi_type)
        
        '''Contact rate: eta_MLP'''
        #-----------------------------------------------------------------------#
        # store inputs of eta, (S, A, Y), into tensor of floats
        cat_tensor = torch.cat([u[:,[0,3,4]]], dim=1).float().to(inputs.device)
        # evaluate eta_MLP(S, A, Y)
        eta = self.eta_func(cat_tensor)
        # transform eta_MLP(S, A, Y) into the interval [yita_lb, yita_ub]
        yita = self.yita_lb + (self.yita_ub - self.yita_lb) * eta[:, 0][:, None]
        #-----------------------------------------------------------------------#
        
        '''Tracing rate: beta_MLP'''
        #-----------------------------------------------------------------------#
        # store inputs of beta, (S+A+Y), into tensor of floats
        yq_tensor = torch.cat([u[:,[0,3,4]].sum(dim=1, keepdim=True), chi_t], dim=1).float().to(inputs.device)
        # evaluate beta_MLP(S+A+Y) NOTE: We are learning beta(S+A+Y)_MLP, not beta_MLP(S+A+Y, chi(t))
        beta0 = self.beta_func(yq_tensor)
        # transform beta_MLP(S+A+Y) into the interval [beta_lb, beta_ub]
        #beta = self.beta_lb + (self.beta_ub - self.beta_lb) * beta0
        # evaluate beta_MLP(S+A+Y)*chi(t)
        beta = chi_t * beta0
        #-----------------------------------------------------------------------#
        
        '''Quarantined Diagnosis rate: tau_MLP'''
        #-----------------------------------------------------------------------#
        # store inputs of tau, (A, Y), into tensor of floats
        ay_tensor = torch.Tensor(u[:,[3,4]]).float().to(inputs.device)
        # evaluate tau_MLP(A, Y)
        tau0 = self.tau_func(ay_tensor)
        # transform tau_MLP(A, Y) into the interval [tau_lb, tau_ub]
        tau = self.tau_lb + (self.tau_ub - self.tau_lb) * tau0
        #-----------------------------------------------------------------------#
        
        '''Symptomatic Diagnosis rate: mu_MLP'''
        #-----------------------------------------------------------------------#
        # store inputs of mu, (Y, F), into tensor of floats
        y_tensor = torch.Tensor(u[:,[4, 8]]).float().to(inputs.device)
        # evaluate mu_MLP(Y, F)
        mu0 = self.mu_func(y_tensor)
        # transform mu_MLP(Y, F) into the interval [mu_lb, mu_ub]
        mu = self.mu_lb + (self.mu_ub - self.mu_lb) * mu0
        #-----------------------------------------------------------------------#
        
        # STEAYDQRF model, loop through each compartment
        s, tq, e, a, y, d, q, r, f = u[:, 0][:, None], u[:, 1][:, None], u[:, 2][:, None], u[:, 3][:, None],\
                                    u[:, 4][:, None], u[:, 5][:, None], u[:, 6][:, None], u[:, 7][:, None],\
                                    u[:, 8][:, None]
        new_d = mu * y + tau * q
        for i in range(self.n_com):
            d1 = Gradient(u[:, i], inputs, order=1)
            ut = d1[:, 0][:, None]
            LHS = ut / self.t_max_real
            if i == 0:
                # dS
                # RHS = - yita * s * (a + y)  - self.beta * new_d * self.n_contacts * s + self.alpha * tq
                RHS = - yita * s  * (a + y) - beta * new_d * self.n_contacts * s + self.alpha * tq
            elif i == 1:
                # dT
                # RHS = self.beta * new_d * self.n_contacts * s  - self.alpha * tq
                RHS = beta * new_d * self.n_contacts * s - self.alpha * tq
            elif i == 2:
                # dE
                # RHS = yita * s  * (a + y) - self.gamma * e
                RHS = yita * s * (a + y) - self.gamma * e
            elif i == 3:
                # dA
                # RHS = self.p_asymp * self.gamma * e - self.lamda * a - self.beta * new_d * self.n_contacts * a
                RHS = self.p_asymp * self.gamma * e - self.lamda * a - beta * new_d * self.n_contacts * a
            elif i == 4:
                # dY
                # RHS = (1 - self.p_asymp) * self.gamma * e - (mu + self.lamda + self.delta) * y - self.beta * new_d * self.n_contacts * y
                RHS = (1 - self.p_asymp) * self.gamma * e - (mu + self.lamda + self.delta) * y - beta * new_d * self.n_contacts * y
            elif i == 5:
                # dD
                # RHS = new_d - self.lamda * d - self.delta * d
                RHS = mu * y + tau * q - self.lamda * d - self.delta * d
            elif i == 6:
                # dQ
                # RHS = self.beta * new_d * self.n_contacts * (a + y) - (tau + self.lamda) * q - self.delta * q
                RHS = beta * new_d * self.n_contacts * (a + y) - (tau + self.lamda + self.delta) * q
            elif i == 7:
                # dR
                RHS = self.lamda * (a + y + d + q)
                # self.drdt_loss = self.drdt_weight * torch.where(LHS < 0, LHS ** 2, torch.zeros_like(LHS))
            elif i == 8:
                # dF
                RHS = self.delta * (y + d + q)
                # self.dfdt_loss = self.dfdt_weight * torch.where(LHS < 0, LHS ** 2, torch.zeros_like(LHS))

            if i in [0, 1, 2, 3, 4, 5, 6]:
                pde_loss += (LHS - RHS) ** 2

        pde_loss *= self.pde_loss_weight

        # constraints on contact_rate function
        yita_final = yita * (a + y)
        deta = Gradient(yita_final, cat_tensor, order=1)
        self.eta_a_loss = 0
        self.eta_a_loss += self.eta_loss_weight * torch.where(deta[:,0] < 0, deta[:,0] ** 2, torch.zeros_like(deta[:,0]))

        self.eta_y_loss = 0
        self.eta_y_loss += self.eta_loss_weight * torch.where(deta[:,1] < 0, deta[:,1] ** 2, torch.zeros_like(deta[:,1]))

        # constraint on tau function
        dtau = Gradient(tau, ay_tensor, order=1)
        self.tau_a_loss = 0
        self.tau_a_loss += self.tau_loss_weight * torch.where(dtau[:,0] < 0, dtau[:,0] ** 2, torch.zeros_like(dtau[:,0]))

        self.tau_y_loss = 0
        self.tau_y_loss += self.tau_loss_weight * torch.where(dtau[:,1] < 0, dtau[:,1] ** 2, torch.zeros_like(dtau[:,1]))
        
        # constraint on mu function
        dmu = Gradient(mu, y_tensor, order=1)
        self.mu_y_loss = 0
        self.mu_y_loss += self.mu_loss_weight * torch.where(dmu[:,0] < 0, dmu[:,0] ** 2, torch.zeros_like(dmu[:,0]))
        
        self.mu_f_loss = 0
        self.mu_f_loss += self.mu_loss_weight * torch.where(dmu[:,1] < 0, dmu[:,1] ** 2, torch.zeros_like(dmu[:,1]))

        if return_mean:
            return torch.mean(pde_loss  + self.eta_a_loss + self.eta_y_loss + self.tau_a_loss + self.tau_y_loss + self.mu_y_loss + self.mu_f_loss)
        else:
            return pde_loss

    def pde_loss_no_d(self, inputs, outputs, return_mean=True):
        """ pde loss for the case of removing compartment D"""
        pde_loss = 0
        # unpack inputs input (N,1) shape
        t = inputs[:, 0][:, None]

        # partial derivative computations
        u = outputs.clone()

        contact_rate = self.contact_rate(u[:,[0,3,4]])  # what to input contact_rate MLP
        yita = self.yita_lb + (self.yita_ub - self.yita_lb) * contact_rate[:, 0][:, None]
        tau = self.tau_lb + (self.tau_ub - self.tau_lb) * self.quarantine_test_prob(u[:,[3,4]])
        # STEADYQRF model, loop through each compartment
        s, tq, e, a, y, q, r, f = u[:, 0][:, None], u[:, 1][:, None], u[:, 2][:, None], u[:, 3][:, None],\
                                    u[:, 4][:, None], u[:, 5][:, None], u[:, 6][:, None], u[:, 7][:, None]
        for i in range(self.n_com):
            d1 = Gradient(u[:, i], inputs, order=1)
            ut = d1[:, 0][:, None]
            LHS = ut / self.t_max_real
            new_d = self.mu * y + tau * q
            if i == 0:
                # dS
                RHS = - yita * s * (a + y)  - self.beta * new_d * self.n_contacts * s + self.alpha * tq
            elif i == 1:
                # dT
                RHS = self.beta * new_d * self.n_contacts * s - self.alpha * tq
            elif i == 2:
                # dE
                RHS = yita * s * (a + y) - self.gamma * e
            elif i == 3:
                # dA
                RHS = self.p_asymp * self.gamma * e - self.lamda * a - self.beta * new_d * self.n_contacts * a
            elif i == 4:
                # dY
                RHS = (1 - self.p_asymp) * self.gamma * e - (self.mu + self.lamda + self.delta) * y - self.beta * new_d * self.n_contacts * y
            elif i == 5:
                # dQ
                RHS = self.beta * new_d * self.n_contacts * (a + y) + self.mu * q - self.delta * q
            elif i == 6:
                # dR
                RHS = self.lamda * (a + y + q)
                # self.drdt_loss = self.drdt_weight * torch.where(LHS < 0, LHS ** 2, torch.zeros_like(LHS))
            elif i == 7:
                # dF
                RHS = self.delta * (y + q)
                # self.dfdt_loss = self.dfdt_weight * torch.where(LHS < 0, LHS ** 2, torch.zeros_like(LHS))
                
            if i in [0, 1, 2, 3, 4, 5]:
                pde_loss += (LHS - RHS) ** 2

        pde_loss *= self.pde_loss_weight


        if return_mean:
            return torch.mean(pde_loss)  #  + self.dfdt_loss + self.drdt_loss
        else:
            return pde_loss

    def loss(self, pred, true):

        self.gls_loss_val = 0
        self.pde_loss_val = 0

        # load cached inputs from forward pass
        inputs = self.inputs

        # randomly sample from input domain
        t = torch.rand(self.num_samples, 1, requires_grad=True)
        t = t * (self.t_max - self.t_min) + self.t_min
        inputs_rand = t.to(inputs.device)
        # inputs_rand = torch.cat([x, t], dim=1).float().to(inputs.device)

        # predict surface fitter at sampled points
        outputs_rand = self.surface_fitter(t)

        # compute surface loss
        self.gls_loss_val = self.surface_weight * self.gls_loss(pred, true)
        # self.gls_loss_val += self.surface_weight * self.kl_loss(torch.log(pred), true)
        # compute PDE loss at sampled locations
        if self.pde_weight != 0:
            if self.keep_d:
                self.pde_loss_val += self.pde_weight * self.pde_loss(inputs_rand, outputs_rand)
            else:
                self.pde_loss_val += self.pde_weight * self.pde_loss_no_d(inputs_rand, outputs_rand)

        return self.gls_loss_val + self.pde_loss_val
    
#--------------------------------DEMO for adding mu_MLP-----------------------------------------#
#-----------------------------------------------------------------------------------------------#
#--------------------------------Original COVASIM_BINN by Xin Li--------------------------------#
class main_MLP(nn.Module):
    '''
    Construct MLP surrogate model for the solution of the governing ODE system.
    Includes three hidden layers. The first with 512 neurons and the latter two with 
    256 neurons. All neurons in hidden layers are ReLU-activated. Output
    is softmax-activated to keep predicted values of S,T,E,A,Y,D,Q,R,F between 0 and 1
    and adding up to one since they are dimensionless ratios of the population.

    Inputs:
        num_outputs (int): number of outputs

    Args:
        inputs (torch tensor): time vector, t, with shape (N, 1)

    Returns:
        outputs (torch tensor): predicted u = (S, T, E, A, Y, D, Q, R, F) values with shape (N, 9)
    '''

    def __init__(self, num_outputs):
        super().__init__()
        self.mlp = BuildMLP(
            input_features=1,
            layers=[512, 256, 256, num_outputs],
            activation=nn.ReLU(),
            linear_output=False,
            output_activation=None)  #  SoftplusReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        outputs = self.mlp(inputs)
        outputs = self.softmax(outputs)

        return outputs


class infect_rate_MLP(nn.Module):
    '''
    Construct MLP surrogate model for the contact rate.
    Includes one hidden layer with 256 ReLU-activated neurons. Output
    is sigmoid-activated to keep predicted rates between 0 and 1.

    Inputs:
        N/A

    Args:
        inputs (torch tensor): S, A, Y with shape (N, 3)

    Returns:
        outputs (torch tensor): predicted contact rate, eta(S, A, Y), values with shape (N, 1)
    '''

    def __init__(self):
        super().__init__()
        self.mlp = BuildMLP(
            input_features=3,
            layers=[256, 1],
            activation=nn.ReLU(),
            linear_output=False,
            output_activation=nn.Sigmoid())  #  SoftplusReLU()

    def forward(self, inputs):
        outputs = self.mlp(inputs)

        return outputs

class beta_MLP(nn.Module):
    '''
    Construct MLP surrogate model for the effective tracing rate.
    Includes one hidden layer with 256 ReLU-activated neurons. Output
    is sigmoid-activated to keep predicted tracing rates, beta(S+A+Y), between 0 and 1.

    Inputs:
        N/A

    Args:
        inputs (torch tensor): S+A+Y with shape (N, 1)

    Returns:
        outputs (torch tensor): predicted beta(S+A+Y) values with shape (N, 1)
    '''

    def __init__(self):
        super().__init__()
        self.mlp = BuildMLP(
            input_features=2,
            layers=[256, 1],
            activation=nn.ReLU(),
            linear_output=False,
            output_activation=nn.Sigmoid())  #  SoftplusReLU()

    def forward(self, inputs):
        outputs = self.mlp(inputs)

        return outputs

class tau_MLP(nn.Module):
    '''
    Construct MLP surrogate model for the diagnoses rate of quarantined invidividuals.
    Includes one hidden layer with 256 ReLU-activated neurons. Output
    is sigmoid-activated to keep predicted rates between 0 and 1.

    Inputs:
        N/A

    Args:
        inputs (torch tensor): A, Y values with shape (N, 2)

    Returns:
        outputs (torch tensor): predicted diagnoses rates of quarantined, tau(A, Y), values with shape (N, 1)
    '''

    def __init__(self):
        super().__init__()
        self.mlp = BuildMLP(
            input_features=2,
            layers=[256, 1],
            activation=nn.ReLU(),
            linear_output=False,
            output_activation=nn.Sigmoid())  #  SoftplusReLU()

    def forward(self, inputs):
        outputs = self.mlp(inputs)

        return outputs

def chi(t, eff_ub, chi_type):
    if chi_type is None or chi_type == 'linear':
        rate = eff_ub / 75
        # factor = torch.where(t < 30.0, rate * t, eff_ub * torch.ones_like(t))
        res = torch.zeros_like(t)
        res += (t < 75) * rate * (t + 1)
        res += (t >= 75) * (t < 150) * eff_ub
        res -= (t >= 75) * (t < 150) * (rate * (t - 75 + 1))
        factor = res
    elif chi_type == 'sin':
        # times = np.arange(0, 300, 1)
        rad_times = t * np.pi / 40.
        factor = 0.3 * (1 + torch.sin(rad_times)) / 2
    elif chi_type == 'piecewise':
        factor = torch.zeros_like(t)
        # use pdf of beta distribution
        a, b = 3, 3
        t_max = 159
        max_val = beta.pdf(0.5, a, b, loc=0, scale=1)

        # t < 80
        factor = factor + (t < 80) * torch.Tensor(beta.pdf(t.cpu().detach().numpy() / t_max, a, b, loc=0, scale=1)).to(
            t.device) * eff_ub / max_val

        # t > 120
        factor = factor + (t >= 120) * torch.Tensor(beta.pdf((t.cpu().detach().numpy() - 40) / t_max, a, b, loc=0,
                                                             scale=1)).to(t.device) * eff_ub / max_val

        # otherwise
        factor = factor + (t >= 80) * (t < 120) * eff_ub

    elif chi_type == 'constant':
        factor = eff_ub * torch.ones_like(t)
    return factor

class BINNCovasim(nn.Module):
    '''
    Constructs a biologically-informed neural network (BINN) composed of
    average number of contacts sufficient to transmit infection per unit of time (eta),
    the effective tracing rate (beta), and the rate of diagnoses from people in 
    quarantine (tau).

    Args:
        params (dict): dictionary of parameters from COVASIM model.
        t_max_real (float): the unscaled maximum time point (t).
        tracing_array (array): array values of tracing probabilities as a function of time (t).
        yita_lb (float): yita lower bound.
        yita_ub (float): yita upper bound.
        keep_d (bool): If true, then include D (diagnosed) in model, otherwise exlcude it.
        chi_type (func): real-valued function of function that affects the quarantining rate.

    '''

    def __init__(self, params, t_max_real, tracing_array, yita_lb=None, yita_ub=None, keep_d=False, chi_type=None):

        super().__init__()

        self.n_com = 9 if keep_d else 8
        # surface fitter
        self.yita_loss = None
        self.yita_lb = yita_lb if yita_lb is not None else 0.2
        self.yita_ub = yita_ub if yita_ub is not None else 0.4
        self.beta_lb = 0.1
        self.beta_ub = 0.3
        self.tau_lb = 0.1 #   0.01
        self.tau_ub =  0.3 #  params['tau_ub']
        self.surface_fitter = main_MLP(self.n_com)

        # pde functions/components
        self.eta_func = infect_rate_MLP()
        self.beta_func = beta_MLP()
        self.tau_func = tau_MLP()

        # input extrema
        self.t_min = 0.0
        self.t_max = 1.0
        self.t_max_real = t_max_real # what is max(t) in the real unaltered timescale

        # loss weights
        self.IC_weight = 1e1
        self.surface_weight = 1e2
        self.pde_weight = 1e4  # 1e4
        
        if keep_d:
            self.weights_c = torch.tensor(np.array([1, 1000, 1, 1000, 1000, 1, 1000, 1, 1000])[None, :], dtype=torch.float) # [1, 1, 1, 1000, 1, 1, 1000, 1, 1000]
        else:
            self.weights_c = torch.tensor(np.array([1, 1, 1, 1, 1, 1000, 1, 1000])[None, :], dtype=torch.float)
        # self.yita_weight = 0
        self.pde_loss_weight = 1e0
        self.eta_loss_weight = 1e5
        self.tau_loss_weight = 1e5

        # proportionality constant
        self.gamma = 0.2

        # number of samples for pde loss
        self.num_samples = 1000

        self.name = 'covasim_fitter'

        self.params = params

        self.population = params['population']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.gamma = params['gamma']
        self.mu = params['mu']
        # self.tau = params['tau'] / 4 if 'tau' in params else None
        self.lamda = params['lamda']
        self.p_asymp = params['p_asymp']
        self.n_contacts = params['n_contacts']
        self.delta = params['delta']
        self.tracing_array = tracing_array

        self.keep_d = keep_d

        # if dynamic
        if 'dynamic_tracing' in params:
            self.is_dynamic = True
        self.eff_ub = params['eff_ub']

        self.chi_type = chi_type if chi_type is not None else None


    def forward(self, inputs):

        # cache input batch for pde loss
        self.inputs = inputs

        return self.surface_fitter(self.inputs)

    def gls_loss(self, pred, true):

        residual = (pred - true) ** 2

        # add weight to initial condition
        residual *= torch.where(self.inputs[:, 0][:, None] == 0,
                                self.IC_weight * torch.ones_like(pred),
                                torch.ones_like(pred))

        # proportional GLS weighting
        residual *= pred.abs().clamp(min=1.0) ** (-self.gamma)

        # apply weights on compartments
        residual *= self.weights_c

        return torch.mean(residual)

    def pde_loss(self, inputs, outputs, return_mean=True):

        pde_loss = 0
        # unpack inputs
        t = inputs[:, 0][:, None]

        # partial derivative computations
        u = outputs.clone()
        
        # h(t) values
        chi_t = chi(1 + t * self.t_max_real, self.eff_ub, self.chi_type)
        # chi_t = torch.nn.functional.interpolate()
        
        cat_tensor = torch.cat([u[:,[0,3,4]]], dim=1).float().to(inputs.device) # t,
        eta = self.eta_func(cat_tensor)
        yita = self.yita_lb + (self.yita_ub - self.yita_lb) * eta[:, 0][:, None]
        
        yq_tensor = torch.cat([u[:,[0,3,4]].sum(dim=1, keepdim=True), chi_t], dim=1).float().to(inputs.device) # 5, 7, 8
        beta0 = self.beta_func(yq_tensor)
        # beta = self.beta_lb + (self.beta_ub - self.beta_lb) * beta0
        # beta(S+A+Y) * h(t)
        beta = chi_t * beta0
        
        ay_tensor = torch.Tensor(u[:,[3,4]]).float().to(inputs.device)
        tau0 = self.tau_func(ay_tensor)
        tau = self.tau_lb + (self.tau_ub - self.tau_lb) * tau0 # quarantine_test[:, 0][:, None]
        
        # STEAYDQRF model, loop through each compartment
        s, tq, e, a, y, d, q, r, f = u[:, 0][:, None], u[:, 1][:, None], u[:, 2][:, None], u[:, 3][:, None],\
                                    u[:, 4][:, None], u[:, 5][:, None], u[:, 6][:, None], u[:, 7][:, None],\
                                    u[:, 8][:, None]
        # (mu * Y + tau * Q)
        new_d = self.mu * y + tau * q
        for i in range(self.n_com):
            d1 = Gradient(u[:, i], inputs, order=1)
            ut = d1[:, 0][:, None]
            LHS = ut / self.t_max_real
            if i == 0:
                # dS
                # RHS = - yita * s * (a + y)  - self.beta * new_d * self.n_contacts * s + self.alpha * tq
                RHS = - yita * s  * (a + y) - beta * new_d * self.n_contacts * s + self.alpha * tq
            elif i == 1:
                # dT
                # RHS = self.beta * new_d * self.n_contacts * s  - self.alpha * tq
                RHS = beta * new_d * self.n_contacts * s - self.alpha * tq
            elif i == 2:
                # dE
                # RHS = yita * s  * (a + y) - self.gamma * e
                RHS = yita * s * (a + y) - self.gamma * e
            elif i == 3:
                # dA
                # RHS = self.p_asymp * self.gamma * e - self.lamda * a - self.beta * new_d * self.n_contacts * a
                RHS = self.p_asymp * self.gamma * e - self.lamda * a - beta * new_d * self.n_contacts * a
            elif i == 4:
                # dY
                # RHS = (1 - self.p_asymp) * self.gamma * e - (self.mu + self.lamda + self.delta) * y - self.beta * new_d * self.n_contacts * y
                RHS = (1 - self.p_asymp) * self.gamma * e - (self.mu + self.lamda + self.delta) * y - beta * new_d * self.n_contacts * y
            elif i == 5:
                # dD
                # RHS = new_d - self.lamda * d - self.delta * d
                RHS = self.mu * y + tau * q - self.lamda * d - self.delta * d
            elif i == 6:
                # dQ
                # RHS = self.beta * new_d * self.n_contacts * (a + y) - (tau + self.lamda) * q - self.delta * q
                RHS = beta * new_d * self.n_contacts * (a + y) - (tau + self.lamda + self.delta) * q
            elif i == 7:
                # dR
                RHS = self.lamda * (a + y + d + q)
                # self.drdt_loss = self.drdt_weight * torch.where(LHS < 0, LHS ** 2, torch.zeros_like(LHS))
            elif i == 8:
                # dF
                RHS = self.delta * (y + d + q)
                # self.dfdt_loss = self.dfdt_weight * torch.where(LHS < 0, LHS ** 2, torch.zeros_like(LHS))

            if i in [0, 1, 2, 3, 4, 5, 6]:
                pde_loss += (LHS - RHS) ** 2

        pde_loss *= self.pde_loss_weight

        # constraints on contact_rate function
        yita_final = yita * (a + y)
        deta = Gradient(yita_final, cat_tensor, order=1)
        self.eta_a_loss = 0
        self.eta_a_loss += self.eta_loss_weight * torch.where(deta[:,0] < 0, deta[:,0] ** 2, torch.zeros_like(deta[:,0]))

        self.eta_y_loss = 0
        self.eta_y_loss += self.eta_loss_weight * torch.where(deta[:,1] < 0, deta[:,1] ** 2, torch.zeros_like(deta[:,1]))

        # constraint on tau function
        dtau = Gradient(tau, ay_tensor, order=1)
        self.tau_a_loss = 0
        self.tau_a_loss += self.tau_loss_weight * torch.where(dtau[:,0] < 0, dtau[:,0] ** 2, torch.zeros_like(dtau[:,0]))

        self.tau_y_loss = 0
        self.tau_y_loss += self.tau_loss_weight * torch.where(dtau[:,1] < 0, dtau[:,1] ** 2, torch.zeros_like(dtau[:,1]))

        if return_mean:
            return torch.mean(pde_loss  + self.eta_a_loss + self.eta_y_loss + self.tau_a_loss + self.tau_y_loss) #
        else:
            return pde_loss  # + self.D_loss + self.G_loss + self.T_loss

    def pde_loss_no_d(self, inputs, outputs, return_mean=True):
        """ pde loss for the case of removing compartment D"""
        pde_loss = 0
        # unpack inputs input (N,1) shape
        t = inputs[:, 0][:, None]

        # partial derivative computations
        u = outputs.clone()

        contact_rate = self.contact_rate(u[:,[0,3,4]])  # what to input contact_rate MLP
        yita = self.yita_lb + (self.yita_ub - self.yita_lb) * contact_rate[:, 0][:, None]
        tau = self.tau_lb + (self.tau_ub - self.tau_lb) * self.quarantine_test_prob(u[:,[3,4]])
        # STEADYQRF model, loop through each compartment
        s, tq, e, a, y, q, r, f = u[:, 0][:, None], u[:, 1][:, None], u[:, 2][:, None], u[:, 3][:, None],\
                                    u[:, 4][:, None], u[:, 5][:, None], u[:, 6][:, None], u[:, 7][:, None]
        for i in range(self.n_com):
            d1 = Gradient(u[:, i], inputs, order=1)
            ut = d1[:, 0][:, None]
            LHS = ut / self.t_max_real
            new_d = self.mu * y + tau * q
            if i == 0:
                # dS
                RHS = - yita * s * (a + y)  - self.beta * new_d * self.n_contacts * s + self.alpha * tq
            elif i == 1:
                # dT
                RHS = self.beta * new_d * self.n_contacts * s - self.alpha * tq
            elif i == 2:
                # dE
                RHS = yita * s * (a + y) - self.gamma * e
            elif i == 3:
                # dA
                RHS = self.p_asymp * self.gamma * e - self.lamda * a - self.beta * new_d * self.n_contacts * a
            elif i == 4:
                # dY
                RHS = (1 - self.p_asymp) * self.gamma * e - (self.mu + self.lamda + self.delta) * y - self.beta * new_d * self.n_contacts * y
            elif i == 5:
                # dQ
                RHS = self.beta * new_d * self.n_contacts * (a + y) + self.mu * q - self.delta * q
            elif i == 6:
                # dR
                RHS = self.lamda * (a + y + q)
            elif i == 7:
                # dF
                RHS = self.delta * (y + q)
                
            if i in [0, 1, 2, 3, 4, 5]:
                pde_loss += (LHS - RHS) ** 2

        pde_loss *= self.pde_loss_weight


        if return_mean:
            return torch.mean(pde_loss)
        else:
            return pde_loss

    def loss(self, pred, true):

        self.gls_loss_val = 0
        self.pde_loss_val = 0

        # load cached inputs from forward pass
        inputs = self.inputs

        # randomly sample from input domain
        t = torch.rand(self.num_samples, 1, requires_grad=True)
        t = t * (self.t_max - self.t_min) + self.t_min
        inputs_rand = t.to(inputs.device)
        # inputs_rand = torch.cat([x, t], dim=1).float().to(inputs.device)

        # predict surface fitter at sampled points
        outputs_rand = self.surface_fitter(t)

        # compute surface loss
        self.gls_loss_val = self.surface_weight * self.gls_loss(pred, true)
        
        # compute PDE loss at sampled locations
        if self.pde_weight != 0:
            if self.keep_d:
                self.pde_loss_val += self.pde_weight * self.pde_loss(inputs_rand, outputs_rand)
            else:
                self.pde_loss_val += self.pde_weight * self.pde_loss_no_d(inputs_rand, outputs_rand)

        return self.gls_loss_val + self.pde_loss_val
#--------------------------------Original COVASIM_BINN by Xin Li--------------------------------#

#------------------------------------No main_MLP------------------------------------------------#
class identity_MLP(nn.Module):
    '''

    Inputs:
        num_outputs (int): number of outputs

    Args:
        inputs (torch tensor): time vector, t, with shape (N, 1)

    Returns:
        outputs (torch tensor): the corresponding average u = (S, T, E, A, Y, D, Q, R, F) values with shape (N, 9) from
            averaged/denoised data.
    '''

    def __init__(self, num_outputs, smooth_data):
        super().__init__()
        self.mlp = BuildMLP(
            input_features=1,
            layers=[num_outputs],
            activation=nn.Identity(),
            linear_output=False,
            output_activation=None)
        self.smooth_data = smooth_data
        # self.smooth_data.requires_grad = False

    def forward(self, inputs):
        # inputs = (inputs * 182).int().numpy()
        inputs = inputs.detach().numpy()
        print(inputs.shape)
        print(self.smooth_data.shape)
        outputs = self.smooth_data[inputs - 1][:,0,:]
        print(outputs.shape)

        return outputs

class MLPComponentsCovasim(nn.Module):
    '''
    Constructs a neural network composed of three distinct MLPs corresponding to 
    the average number of contacts sufficient to transmit infection per unit of 
    time (eta),the effective tracing rate (beta), and the rate of diagnoses from 
    people in quarantine (tau).

    Args:
        params (dict): dictionary of parameters from COVASIM model.
        smooth_data (ndarray): array of averaged compartmental values.
        t_max_real (float): the unscaled maximum time point (t).
        tracing_array (array): array values of tracing probabilities as a function of time (t).
        yita_lb (float): yita lower bound.
        yita_ub (float): yita upper bound.
        keep_d (bool): If true, then include D (diagnosed) in model, otherwise exlcude it.
        chi_type (func): real-valued function of function that affects the quarantining rate.

    '''

    def __init__(self, params, smooth_data, ut, t_max_real, tracing_array, yita_lb=None, yita_ub=None, keep_d=True, chi_type=None):

        super().__init__()

        self.n_com = 9 if keep_d else 8
        
        self.yita_loss = None
        self.yita_lb = yita_lb if yita_lb is not None else 0.2
        self.yita_ub = yita_ub if yita_ub is not None else 0.4
        self.beta_lb = 0.1
        self.beta_ub = 0.3
        self.tau_lb = 0.1
        self.tau_ub =  0.3
        self.surface_fitter = identity_MLP(self.n_com, smooth_data)

        # pde functions/components
        self.eta_func = infect_rate_MLP()
        self.beta_func = beta_MLP()
        self.tau_func = tau_MLP()
        
        # table of derivatives of u with respect to t
        self.ut = ut

        # input extrema
        self.t_min = 0.0
        self.t_max = t_max_real
        self.t_max_real = t_max_real # what is max(t) in the real unaltered timescale

        # loss weights
        self.IC_weight = 1e1
        # self.surface_weight = 1e2
        self.pde_weight = 1e4  # 1e4
        
        if keep_d:
            self.weights_c = torch.tensor(np.array([1, 1000, 1, 1000, 1000, 1, 1000, 1, 1000])[None, :], dtype=torch.float) # [1, 1, 1, 1000, 1, 1, 1000, 1, 1000]
        else:
            self.weights_c = torch.tensor(np.array([1, 1, 1, 1, 1, 1000, 1, 1000])[None, :], dtype=torch.float)
        # self.yita_weight = 0
        self.pde_loss_weight = 1e0
        self.eta_loss_weight = 1e5
        self.tau_loss_weight = 1e5

        # proportionality constant
        self.gamma = 0.2

        # number of samples for pde loss
        self.num_samples = 1000

        self.name = 'covasim_smoothed_fitter'

        self.params = params

        self.population = params['population']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.gamma = params['gamma']
        self.mu = params['mu']
        self.lamda = params['lamda']
        self.p_asymp = params['p_asymp']
        self.n_contacts = params['n_contacts']
        self.delta = params['delta']
        self.tracing_array = tracing_array

        self.keep_d = keep_d

        # if dynamic
        if 'dynamic_tracing' in params:
            self.is_dynamic = True
        self.eff_ub = params['eff_ub']

        self.chi_type = chi_type if chi_type is not None else None


    def forward(self, inputs):

        # cache input batch for pde loss
        self.inputs = inputs

        return self.surface_fitter(self.inputs) # returns self.surface_fitter(self.inputs) which is equal to smooth_data.copy()

    def pde_loss(self, inputs, outputs, return_mean=True):

        pde_loss = 0
        # unpack inputs
        t = inputs[:, 0][:, None]
        N = t.shape[0]
        print(N)
        print(f't : {t}')

        # partial derivative computations
        u = outputs.clone()
        LHS = self.ut
        
        # h(t) values
        chi_t = chi(1 + t, self.eff_ub, self.chi_type)
        
        cat_tensor = torch.cat([u[:,[0,3,4]]], dim=1).float().to(inputs.device)
        eta = self.eta_func(cat_tensor)
        yita = self.yita_lb + (self.yita_ub - self.yita_lb) * eta[:, 0][:, None]
        
        yq_tensor = torch.cat([u[:,[0,3,4]].sum(dim=1, keepdim=True), chi_t], dim=1).float().to(inputs.device)
        beta0 = self.beta_func(yq_tensor)
        # beta = self.beta_lb + (self.beta_ub - self.beta_lb) * beta0
        # beta(S+A+Y) * h(t)
        beta = chi_t * beta0
        
        ay_tensor = torch.Tensor(u[:,[3,4]]).float().to(inputs.device)
        tau0 = self.tau_func(ay_tensor)
        tau = self.tau_lb + (self.tau_ub - self.tau_lb) * tau0
        
        # STEAYDQRF model, loop through each compartment
        s, tq, e, a, y, d, q, r, f = u[:, 0][:, None], u[:, 1][:, None], u[:, 2][:, None], u[:, 3][:, None],\
                                    u[:, 4][:, None], u[:, 5][:, None], u[:, 6][:, None], u[:, 7][:, None],\
                                    u[:, 8][:, None]
        # (mu * Y + tau * Q)
        new_d = self.mu * y + tau * q
        for i in range(self.n_com):
            # d1 = Gradient(u[:, i], inputs, order=1)
            # ut = d1[:, 0][:, None]
            if i == 0:
                # dS
                # RHS = - yita * s * (a + y)  - self.beta * new_d * self.n_contacts * s + self.alpha * tq
                RHS = - yita * s  * (a + y) - beta * new_d * self.n_contacts * s + self.alpha * tq
            elif i == 1:
                # dT
                # RHS = self.beta * new_d * self.n_contacts * s  - self.alpha * tq
                RHS = beta * new_d * self.n_contacts * s - self.alpha * tq
            elif i == 2:
                # dE
                # RHS = yita * s  * (a + y) - self.gamma * e
                RHS = yita * s * (a + y) - self.gamma * e
            elif i == 3:
                # dA
                # RHS = self.p_asymp * self.gamma * e - self.lamda * a - self.beta * new_d * self.n_contacts * a
                RHS = self.p_asymp * self.gamma * e - self.lamda * a - beta * new_d * self.n_contacts * a
            elif i == 4:
                # dY
                # RHS = (1 - self.p_asymp) * self.gamma * e - (self.mu + self.lamda + self.delta) * y - self.beta * new_d * self.n_contacts * y
                RHS = (1 - self.p_asymp) * self.gamma * e - (self.mu + self.lamda + self.delta) * y - beta * new_d * self.n_contacts * y
            elif i == 5:
                # dD
                # RHS = new_d - self.lamda * d - self.delta * d
                RHS = self.mu * y + tau * q - self.lamda * d - self.delta * d
            elif i == 6:
                # dQ
                # RHS = self.beta * new_d * self.n_contacts * (a + y) - (tau + self.lamda) * q - self.delta * q
                RHS = beta * new_d * self.n_contacts * (a + y) - (tau + self.lamda + self.delta) * q
            elif i == 7:
                # dR
                RHS = self.lamda * (a + y + d + q)
                # self.drdt_loss = self.drdt_weight * torch.where(LHS < 0, LHS ** 2, torch.zeros_like(LHS))
            elif i == 8:
                # dF
                RHS = self.delta * (y + d + q)
                # self.dfdt_loss = self.dfdt_weight * torch.where(LHS < 0, LHS ** 2, torch.zeros_like(LHS))

            if i in [0, 1, 2, 3, 4, 5, 6]:
                pde_loss += (LHS[:,i] - RHS) ** 2

        pde_loss *= self.pde_loss_weight

        # constraints on contact_rate function
        yita_final = yita * (a + y)
        deta = Gradient(yita_final, cat_tensor, order=1)
        self.eta_a_loss = 0
        self.eta_a_loss += self.eta_loss_weight * torch.where(deta[:,0] < 0, deta[:,0] ** 2, torch.zeros_like(deta[:,0]))

        self.eta_y_loss = 0
        self.eta_y_loss += self.eta_loss_weight * torch.where(deta[:,1] < 0, deta[:,1] ** 2, torch.zeros_like(deta[:,1]))

        # constraint on tau function
        dtau = Gradient(tau, ay_tensor, order=1)
        self.tau_a_loss = 0
        self.tau_a_loss += self.tau_loss_weight * torch.where(dtau[:,0] < 0, dtau[:,0] ** 2, torch.zeros_like(dtau[:,0]))

        self.tau_y_loss = 0
        self.tau_y_loss += self.tau_loss_weight * torch.where(dtau[:,1] < 0, dtau[:,1] ** 2, torch.zeros_like(dtau[:,1]))

        print(pde_loss.shape)
        print(self.eta_a_loss.shape)

        if return_mean:
            return torch.mean(pde_loss  + self.eta_a_loss + self.eta_y_loss + self.tau_a_loss + self.tau_y_loss)
        else:
            return pde_loss

    def loss(self, pred, true):

        self.pde_loss_val = 0

        # load cached inputs from forward pass
        inputs = self.inputs

        # randomly sample from input domain
        # t = torch.rand(self.num_samples, 1, requires_grad=True)
        # t = torch.from_numpy(np.random.permutation(self.t_max_real) + 1)[:,None]
        t = torch.from_numpy(np.random.randint(1, self.t_max_real, size=1000) + 1)[:,None]
        print(f'min t: {t.min()}')
        print(f'max t: {t.max()}')
        # t = t * (self.t_max - self.t_min) + self.t_min
        inputs_rand = t.to(inputs.device)
        print(f'inputs shape: {inputs_rand.shape}')
        # inputs_rand = torch.cat([x, t], dim=1).float().to(inputs.device)

        # predict surface fitter at sampled points
        outputs_rand = self.surface_fitter(t)
        
        # compute PDE loss at sampled locations
        if self.pde_weight != 0:
            if self.keep_d:
                self.pde_loss_val += self.pde_weight * self.pde_loss(inputs_rand, outputs_rand)
            else:
                self.pde_loss_val += self.pde_weight * self.pde_loss_no_d(inputs_rand, outputs_rand)

        return self.pde_loss_val
#------------------------------------No main_MLP------------------------------------------------$

#------------------------------------No main_MLP 2.0------------------------------------------------$
def surface_fit(t, u):
    
    return u[t - 1]

def est_deriv(t, ut):
    
    return ut[t - 1]

class MLPComponentsCovasim2(nn.Module):
    '''
    Constructs a neural network composed of three distinct MLPs corresponding to 
    the average number of contacts sufficient to transmit infection per unit of 
    time (eta),the effective tracing rate (beta), and the rate of diagnoses from 
    people in quarantine (tau).

    Args:
        params (dict): dictionary of parameters from COVASIM model.
        smooth_data (ndarray): array of averaged compartmental values.
        t_max_real (float): the unscaled maximum time point (t).
        tracing_array (array): array values of tracing probabilities as a function of time (t).
        yita_lb (float): yita lower bound.
        yita_ub (float): yita upper bound.
        keep_d (bool): If true, then include D (diagnosed) in model, otherwise exlcude it.
        chi_type (func): real-valued function of function that affects the quarantining rate.

    '''

    def __init__(self, params, p, smooth_data, ut, t_max_real, tracing_array, yita_lb=None, yita_ub=None, keep_d=True, chi_type=None):

        super().__init__()

        self.n_com = 9 if keep_d else 8
        
        self.yita_loss = None
        self.yita_lb = yita_lb if yita_lb is not None else 0.2
        self.yita_ub = yita_ub if yita_ub is not None else 0.4
        self.beta_lb = 0.1
        self.beta_ub = 0.3
        self.tau_lb = 0.1
        self.tau_ub =  0.3
        self.surface_fitter = surface_fit()

        # pde functions/components
        self.eta_func = infect_rate_MLP()
        self.beta_func = beta_MLP()
        self.tau_func = tau_MLP()
        
        # table of derivatives of u with respect to t
        self.ut = ut

        # input extrema
        self.t_min = 0.0
        self.t_max = t_max_real
        self.t_max_real = t_max_real # what is max(t) in the real unaltered timescale

        # loss weights
        self.IC_weight = 1e1
        # self.surface_weight = 1e2
        self.pde_weight = 1e4  # 1e4
        
        if keep_d:
            self.weights_c = torch.tensor(np.array([1, 1000, 1, 1000, 1000, 1, 1000, 1, 1000])[None, :], dtype=torch.float) # [1, 1, 1, 1000, 1, 1, 1000, 1, 1000]
        else:
            self.weights_c = torch.tensor(np.array([1, 1, 1, 1, 1, 1000, 1, 1000])[None, :], dtype=torch.float)
        # self.yita_weight = 0
        self.pde_loss_weight = 1e0
        self.eta_loss_weight = 1e5
        self.tau_loss_weight = 1e5

        # proportionality constant
        self.gamma = 0.2

        # number of samples for pde loss
        self.num_samples = 1000

        self.name = 'covasim_smoothed_fitter'

        self.params = params

        self.population = params['population']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.gamma = params['gamma']
        self.mu = params['mu']
        self.lamda = params['lamda']
        self.p_asymp = params['p_asymp']
        self.n_contacts = params['n_contacts']
        self.delta = params['delta']
        self.tracing_array = tracing_array

        self.keep_d = keep_d

        # if dynamic
        if 'dynamic_tracing' in params:
            self.is_dynamic = True
        self.eff_ub = params['eff_ub']

        self.chi_type = chi_type if chi_type is not None else None


    def forward(self, inputs):

        # cache input batch for pde loss
        self.inputs = inputs

        return self.surface_fitter(self.inputs) # returns self.surface_fitter(self.inputs) which is equal to smooth_data.copy()

    def pde_loss(self, inputs, outputs, return_mean=True):

        pde_loss = 0
        # unpack inputs
        t = inputs[:, 0][:, None]
        N = t.shape[0]
        print(N)
        print(f't : {t}')

        # partial derivative computations
        u = outputs.clone()
        LHS = self.ut
        
        # h(t) values
        chi_t = chi(1 + t, self.eff_ub, self.chi_type)
        
        cat_tensor = torch.cat([u[:,[0,3,4]]], dim=1).float().to(inputs.device)
        eta = self.eta_func(cat_tensor)
        yita = self.yita_lb + (self.yita_ub - self.yita_lb) * eta[:, 0][:, None]
        
        yq_tensor = torch.cat([u[:,[0,3,4]].sum(dim=1, keepdim=True), chi_t], dim=1).float().to(inputs.device)
        beta0 = self.beta_func(yq_tensor)
        # beta = self.beta_lb + (self.beta_ub - self.beta_lb) * beta0
        # beta(S+A+Y) * h(t)
        beta = chi_t * beta0
        
        ay_tensor = torch.Tensor(u[:,[3,4]]).float().to(inputs.device)
        tau0 = self.tau_func(ay_tensor)
        tau = self.tau_lb + (self.tau_ub - self.tau_lb) * tau0
        
        # STEAYDQRF model, loop through each compartment
        s, tq, e, a, y, d, q, r, f = u[:, 0][:, None], u[:, 1][:, None], u[:, 2][:, None], u[:, 3][:, None],\
                                    u[:, 4][:, None], u[:, 5][:, None], u[:, 6][:, None], u[:, 7][:, None],\
                                    u[:, 8][:, None]
        # (mu * Y + tau * Q)
        new_d = self.mu * y + tau * q
        for i in range(self.n_com):
            # d1 = Gradient(u[:, i], inputs, order=1)
            # ut = d1[:, 0][:, None]
            if i == 0:
                # dS
                # RHS = - yita * s * (a + y)  - self.beta * new_d * self.n_contacts * s + self.alpha * tq
                RHS = - yita * s  * (a + y) - beta * new_d * self.n_contacts * s + self.alpha * tq
            elif i == 1:
                # dT
                # RHS = self.beta * new_d * self.n_contacts * s  - self.alpha * tq
                RHS = beta * new_d * self.n_contacts * s - self.alpha * tq
            elif i == 2:
                # dE
                # RHS = yita * s  * (a + y) - self.gamma * e
                RHS = yita * s * (a + y) - self.gamma * e
            elif i == 3:
                # dA
                # RHS = self.p_asymp * self.gamma * e - self.lamda * a - self.beta * new_d * self.n_contacts * a
                RHS = self.p_asymp * self.gamma * e - self.lamda * a - beta * new_d * self.n_contacts * a
            elif i == 4:
                # dY
                # RHS = (1 - self.p_asymp) * self.gamma * e - (self.mu + self.lamda + self.delta) * y - self.beta * new_d * self.n_contacts * y
                RHS = (1 - self.p_asymp) * self.gamma * e - (self.mu + self.lamda + self.delta) * y - beta * new_d * self.n_contacts * y
            elif i == 5:
                # dD
                # RHS = new_d - self.lamda * d - self.delta * d
                RHS = self.mu * y + tau * q - self.lamda * d - self.delta * d
            elif i == 6:
                # dQ
                # RHS = self.beta * new_d * self.n_contacts * (a + y) - (tau + self.lamda) * q - self.delta * q
                RHS = beta * new_d * self.n_contacts * (a + y) - (tau + self.lamda + self.delta) * q
            elif i == 7:
                # dR
                RHS = self.lamda * (a + y + d + q)
                # self.drdt_loss = self.drdt_weight * torch.where(LHS < 0, LHS ** 2, torch.zeros_like(LHS))
            elif i == 8:
                # dF
                RHS = self.delta * (y + d + q)
                # self.dfdt_loss = self.dfdt_weight * torch.where(LHS < 0, LHS ** 2, torch.zeros_like(LHS))

            if i in [0, 1, 2, 3, 4, 5, 6]:
                pde_loss += (LHS[:,i] - RHS) ** 2

        pde_loss *= self.pde_loss_weight

        # constraints on contact_rate function
        yita_final = yita * (a + y)
        deta = Gradient(yita_final, cat_tensor, order=1)
        self.eta_a_loss = 0
        self.eta_a_loss += self.eta_loss_weight * torch.where(deta[:,0] < 0, deta[:,0] ** 2, torch.zeros_like(deta[:,0]))

        self.eta_y_loss = 0
        self.eta_y_loss += self.eta_loss_weight * torch.where(deta[:,1] < 0, deta[:,1] ** 2, torch.zeros_like(deta[:,1]))

        # constraint on tau function
        dtau = Gradient(tau, ay_tensor, order=1)
        self.tau_a_loss = 0
        self.tau_a_loss += self.tau_loss_weight * torch.where(dtau[:,0] < 0, dtau[:,0] ** 2, torch.zeros_like(dtau[:,0]))

        self.tau_y_loss = 0
        self.tau_y_loss += self.tau_loss_weight * torch.where(dtau[:,1] < 0, dtau[:,1] ** 2, torch.zeros_like(dtau[:,1]))

        print(pde_loss.shape)
        print(self.eta_a_loss.shape)

        if return_mean:
            return torch.mean(pde_loss  + self.eta_a_loss + self.eta_y_loss + self.tau_a_loss + self.tau_y_loss)
        else:
            return pde_loss

    def loss(self, pred, true):

        self.pde_loss_val = 0

        # load cached inputs from forward pass
        inputs = self.inputs

        # randomly sample from input domain
        t = torch.from_numpy(np.random.permutation(self.t_max_real) + 1)
        
        inputs_rand = t.to(inputs.device)

        # predict surface fitter at sampled points
        outputs_rand = self.surface_fitter(t)
        
        # compute PDE loss at sampled locations
        if self.pde_weight != 0:
            self.pde_loss_val += self.pde_weight * self.pde_loss(inputs_rand, outputs_rand)

        return self.pde_loss_val
#------------------------------------No main_MLP 2.0------------------------------------------------$

#--------------------------------no main MLP 3.0--------------------------------#
class MLPComponentsCV(nn.Module):
    '''
    Constructs a biologically-informed neural network (BINN) composed of
    average number of contacts sufficient to transmit infection per unit of time (eta),
    the effective tracing rate (beta), and the rate of diagnoses from people in 
    quarantine (tau).

    Args:
        params (dict): dictionary of parameters from COVASIM model.
        t_max_real (float): the unscaled maximum time point (t).
        tracing_array (array): array values of tracing probabilities as a function of time (t).
        yita_lb (float): yita lower bound.
        yita_ub (float): yita upper bound.
        keep_d (bool): If true, then include D (diagnosed) in model, otherwise exlcude it.
        chi_type (func): real-valued function of function that affects the quarantining rate.

    '''

    def __init__(self, params, u_tensor, t_max_real, tracing_array, yita_lb=None, yita_ub=None, keep_d=False, chi_type=None):

        super().__init__()

        self.n_com = 9 if keep_d else 8
        # surface fitter
        self.yita_loss = None
        self.yita_lb = yita_lb if yita_lb is not None else 0.2
        self.yita_ub = yita_ub if yita_ub is not None else 0.4
        self.beta_lb = 0.1
        self.beta_ub = 0.3
        self.tau_lb = 0.1
        self.tau_ub =  0.3
        self.u_tensor = u_tensor
        self.u = u_tensor[:,:,0]
        self.ut = u_tensor[:,:,1]

        # pde functions/components
        self.eta_func = infect_rate_MLP()
        self.beta_func = beta_MLP()
        self.tau_func = tau_MLP()

        # input extrema
        self.t_min = 0.0
        self.t_max = 1.0
        self.t_max_real = t_max_real # what is max(t) in the real unaltered timescale

        # loss weights
        self.surface_weight = 1e2
        self.pde_weight = 1e4  # 1e4

        if keep_d:
            self.weights_c = torch.tensor(np.array([1, 1000, 1, 1000, 1000, 1, 1000, 1, 1000])[None, :], dtype=torch.float) # [1, 1, 1, 1000, 1, 1, 1000, 1, 1000]
        else:
            self.weights_c = torch.tensor(np.array([1, 1, 1, 1, 1, 1000, 1, 1000])[None, :], dtype=torch.float)
        # self.yita_weight = 0
        self.pde_loss_weight = 1e0
        self.eta_loss_weight = 1e5
        self.tau_loss_weight = 1e5

        # proportionality constant
        self.gamma = 0.2

        # number of samples for pde loss
        self.num_samples = int(0.8 * 181) # QUESTION: What's going on here? What does it mean by number of sample? Samples for what?

        self.name = 'covasim_fitter'

        self.params = params

        self.population = params['population']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.gamma = params['gamma']
        self.mu = params['mu']
        self.lamda = params['lamda']
        self.p_asymp = params['p_asymp']
        self.n_contacts = params['n_contacts']
        self.delta = params['delta']
        self.tracing_array = tracing_array

        self.keep_d = keep_d

        # if dynamic
        if 'dynamic_tracing' in params:
            self.is_dynamic = True
        self.eff_ub = params['eff_ub']

        self.chi_type = chi_type if chi_type is not None else None


    def forward(self, inputs):

        # cache input batch for pde loss
        self.inputs = inputs

        # whatever we return here goes into the NN as an input for the loss function (??)
        return self.u

    def pde_loss(self, inputs, outputs, return_mean=True):

        pde_loss = 0
        # unpack inputs
        t = inputs[:, 0][:, None] / self.t_max_real

        # partial derivative computations
        u = outputs[:,:,0].clone()
        ut = outputs[:,:,1].clone()
        
        # h(t) values
        chi_t = chi(1 + t, self.eff_ub, self.chi_type)
        # chi_t = torch.nn.functional.interpolate()
        
        cat_tensor = torch.cat([u[:,[0,3,4]]], dim=1).float().to(inputs.device)
        eta = self.eta_func(cat_tensor)
        yita = self.yita_lb + (self.yita_ub - self.yita_lb) * eta[:, 0][:, None]
        
        yq_tensor = torch.cat([u[:,[0,3,4]].sum(dim=1, keepdim=True), chi_t], dim=1).float().to(inputs.device)
        beta0 = self.beta_func(yq_tensor)
        # beta(S+A+Y) * h(t)
        beta = chi_t * beta0
        
        ay_tensor = torch.Tensor(u[:,[3,4]]).float().to(inputs.device)
        tau0 = self.tau_func(ay_tensor)
        tau = self.tau_lb + (self.tau_ub - self.tau_lb) * tau0
        
        # STEAYDQRF model, loop through each compartment
        s, tq, e, a, y, d, q, r, f = u[:, 0][:, None], u[:, 1][:, None], u[:, 2][:, None], u[:, 3][:, None],\
                                    u[:, 4][:, None], u[:, 5][:, None], u[:, 6][:, None], u[:, 7][:, None],\
                                    u[:, 8][:, None]
        # (mu * Y + tau * Q)
        new_d = self.mu * y + tau * q
        LHS = ut
        for i in range(self.n_com):
            # d1 = Gradient(u[:, i], inputs, order=1) <------- We don't need this since we numerically approximated these
            # ut = d1[:, 0][:, None] <------------ Instead use the LHS matrix of derivatives where the column <-> compartment
            LHS_i = LHS[:,i]
            if i == 0:
                # dS
                # RHS = - yita * s * (a + y)  - self.beta * new_d * self.n_contacts * s + self.alpha * tq
                RHS = - yita * s  * (a + y) - beta * new_d * self.n_contacts * s + self.alpha * tq
            elif i == 1:
                # dT
                # RHS = self.beta * new_d * self.n_contacts * s  - self.alpha * tq
                RHS = beta * new_d * self.n_contacts * s - self.alpha * tq
            elif i == 2:
                # dE
                # RHS = yita * s  * (a + y) - self.gamma * e
                RHS = yita * s * (a + y) - self.gamma * e
            elif i == 3:
                # dA
                # RHS = self.p_asymp * self.gamma * e - self.lamda * a - self.beta * new_d * self.n_contacts * a
                RHS = self.p_asymp * self.gamma * e - self.lamda * a - beta * new_d * self.n_contacts * a
            elif i == 4:
                # dY
                # RHS = (1 - self.p_asymp) * self.gamma * e - (self.mu + self.lamda + self.delta) * y - self.beta * new_d * self.n_contacts * y
                RHS = (1 - self.p_asymp) * self.gamma * e - (self.mu + self.lamda + self.delta) * y - beta * new_d * self.n_contacts * y
            elif i == 5:
                # dD
                # RHS = new_d - self.lamda * d - self.delta * d
                RHS = self.mu * y + tau * q - self.lamda * d - self.delta * d
            elif i == 6:
                # dQ
                # RHS = self.beta * new_d * self.n_contacts * (a + y) - (tau + self.lamda) * q - self.delta * q
                RHS = beta * new_d * self.n_contacts * (a + y) - (tau + self.lamda + self.delta) * q
            elif i == 7:
                # dR
                RHS = self.lamda * (a + y + d + q)
                # self.drdt_loss = self.drdt_weight * torch.where(LHS < 0, LHS ** 2, torch.zeros_like(LHS))
            elif i == 8:
                # dF
                RHS = self.delta * (y + d + q)
                # self.dfdt_loss = self.dfdt_weight * torch.where(LHS < 0, LHS ** 2, torch.zeros_like(LHS))

            if i in [0, 1, 2, 3, 4, 5, 6]:
                pde_loss += (LHS_i - RHS) ** 2

        pde_loss *= self.pde_loss_weight

        # constraints on contact_rate function
        yita_final = yita * (a + y)
        deta = Gradient(yita_final, cat_tensor, order=1)
        self.eta_a_loss = 0
        self.eta_a_loss += self.eta_loss_weight * torch.where(deta[:,0] < 0, deta[:,0] ** 2, torch.zeros_like(deta[:,0]))

        self.eta_y_loss = 0
        self.eta_y_loss += self.eta_loss_weight * torch.where(deta[:,1] < 0, deta[:,1] ** 2, torch.zeros_like(deta[:,1]))

        # constraint on tau function
        dtau = Gradient(tau, ay_tensor, order=1)
        self.tau_a_loss = 0
        self.tau_a_loss += self.tau_loss_weight * torch.where(dtau[:,0] < 0, dtau[:,0] ** 2, torch.zeros_like(dtau[:,0]))

        self.tau_y_loss = 0
        self.tau_y_loss += self.tau_loss_weight * torch.where(dtau[:,1] < 0, dtau[:,1] ** 2, torch.zeros_like(dtau[:,1]))

        if return_mean:
            return torch.mean(pde_loss  + self.eta_a_loss + self.eta_y_loss + self.tau_a_loss + self.tau_y_loss)
        else:
            return pde_loss

    def pde_loss_no_d(self, inputs, outputs, return_mean=True):
        """ pde loss for the case of removing compartment D"""
        pde_loss = 0
        # unpack inputs input (N,1) shape
        t = inputs[:, 0][:, None] / self.t_max_real

        # partial derivative computations
        u = outputs[:,:,0].clone()
        ut = outputs[:,:,1].clone()

        contact_rate = self.contact_rate(u[:,[0,3,4]])  # what to input contact_rate MLP
        yita = self.yita_lb + (self.yita_ub - self.yita_lb) * contact_rate[:, 0][:, None]
        tau = self.tau_lb + (self.tau_ub - self.tau_lb) * self.quarantine_test_prob(u[:,[3,4]])
        # STEADYQRF model, loop through each compartment
        s, tq, e, a, y, q, r, f = u[:, 0][:, None], u[:, 1][:, None], u[:, 2][:, None], u[:, 3][:, None],\
                                    u[:, 4][:, None], u[:, 5][:, None], u[:, 6][:, None], u[:, 7][:, None]
        LHS = ut / self.t_max_real
        for i in range(self.n_com):
            # d1 = Gradient(u[:, i], inputs, order=1)
            # ut = d1[:, 0][:, None]
            LHS_i = LHS[:,i]
            new_d = self.mu * y + tau * q
            if i == 0:
                # dS
                RHS = - yita * s * (a + y)  - self.beta * new_d * self.n_contacts * s + self.alpha * tq
            elif i == 1:
                # dT
                RHS = self.beta * new_d * self.n_contacts * s - self.alpha * tq
            elif i == 2:
                # dE
                RHS = yita * s * (a + y) - self.gamma * e
            elif i == 3:
                # dA
                RHS = self.p_asymp * self.gamma * e - self.lamda * a - self.beta * new_d * self.n_contacts * a
            elif i == 4:
                # dY
                RHS = (1 - self.p_asymp) * self.gamma * e - (self.mu + self.lamda + self.delta) * y - self.beta * new_d * self.n_contacts * y
            elif i == 5:
                # dQ
                RHS = self.beta * new_d * self.n_contacts * (a + y) + self.mu * q - self.delta * q
            elif i == 6:
                # dR
                RHS = self.lamda * (a + y + q)
            elif i == 7:
                # dF
                RHS = self.delta * (y + q)
                
            if i in [0, 1, 2, 3, 4, 5]:
                pde_loss += (LHS_i - RHS) ** 2

        pde_loss *= self.pde_loss_weight


        if return_mean:
            return torch.mean(pde_loss)
        else:
            return pde_loss

    def loss(self, pred, true):

        self.pde_loss_val = 0

        # load cached inputs from forward pass - QUESTION: What does this mean and do?
        inputs = self.inputs

        # randomly sample from input domain
        t = torch.randint(1, self.t_max_real, (self.t_max_real, 1), requires_grad=False)
        u = self.u[t-1]
        ut = self.ut[t-1]
        # t = t * (self.t_max - self.t_min) + self.t_min
        inputs_rand = t.to(inputs.device)
        # inputs_rand = torch.cat([x, t], dim=1).float().to(inputs.device)

        # get predicted surface fit at sampled points
        outputs_rand = torch.cat(u[:,:,None], ut[:,:,None], axis=2)
        
        # compute PDE loss at sampled locations
        if self.pde_weight != 0:
            if self.keep_d:
                self.pde_loss_val += self.pde_weight * self.pde_loss(inputs_rand, outputs_rand)
            else:
                self.pde_loss_val += self.pde_weight * self.pde_loss_no_d(inputs_rand, outputs_rand)

        return self.pde_loss_val
#--------------------------------Original COVASIM_BINN by Xin Li--------------------------------#