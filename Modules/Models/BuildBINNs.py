import numpy as np
import torch, pdb
import torch.nn as nn

import sys
sys.path.append(sys.path[0] + '\\../')

from Modules.Models.BuildMLP import BuildMLP
from Modules.Activations.SoftplusReLU import SoftplusReLU
from Modules.Utils.Gradient import Gradient

from scipy.stats import beta

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

class eta_NN(nn.Module):
    '''
    Construct NN surrogate model for the contact rate.
    Includes one hidden layer with 3 layers of 256 ReLU-activated neurons. Output
    is sigmoid-activated to keep predicted rates between 0 and 1.

    Inputs:
        num_features (int): number of inputs
        is_mlp (bool): indicate whether or not to be an MLP or a single layer NN

    Args:
        inputs (torch tensor): S, A, Y, M with shape (N, 4)

    Returns:
        outputs (torch tensor): predicted contact rate, eta(S, A, Y, M), values with shape (N, 1)
    '''

    def __init__(self, num_features=3, is_mlp=False):
        super().__init__()
        self.mlp = BuildMLP(
            input_features=num_features,
            layers=[256, 1] if not is_mlp else [256, 256, 256, 1],
            activation=nn.ReLU(),
            linear_output=False,
            output_activation=nn.Sigmoid())  #  SoftplusReLU()

    def forward(self, inputs):
        outputs = self.mlp(inputs)

        return outputs

class beta_NN(nn.Module):
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

    def __init__(self, is_mlp=False):
        super().__init__()
        self.mlp = BuildMLP(
            input_features=2,
            layers=[256, 1] if not is_mlp else [256, 256, 256, 1],
            activation=nn.ReLU(),
            linear_output=False,
            output_activation=nn.Sigmoid())  #  SoftplusReLU()

    def forward(self, inputs):
        outputs = self.mlp(inputs)

        return outputs

class tau_NN(nn.Module):
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

    def __init__(self, is_mlp=False):
        super().__init__()
        self.mlp = BuildMLP(
            input_features=2,
            layers=[256, 1] if not is_mlp else [256, 256, 256, 1],
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
        self.yita_loss = None
        self.yita_lb = yita_lb if yita_lb is not None else 0.2
        self.yita_ub = yita_ub if yita_ub is not None else 0.4
        self.beta_lb = 0.1
        self.beta_ub = 0.3
        self.tau_lb = 0.1
        self.tau_ub =  0.3
        self.surface_fitter = main_MLP(self.n_com)

        # pde functions/components
        self.eta_func = eta_NN()
        self.beta_func = beta_NN()
        self.tau_func = tau_NN()

        # input extrema
        self.t_min = 0.0
        self.t_max = 1.0
        self.t_max_real = t_max_real

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
        # self.num_samples = 1000

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
        residual *= self.weights_c.to(self.inputs.device)

        return torch.mean(residual)

    def pde_loss(self, inputs, outputs, return_mean=True):

        pde_loss = 0
        # unpack inputs
        t = inputs[:, 0][:, None]

        # partial derivative computations
        u = outputs.clone()
        
        # h(t) values
        chi_t = chi(1 + t * self.t_max_real, self.eff_ub, self.chi_type)
        
        cat_tensor = torch.cat([u[:,[0,3,4]]], dim=1) #.float().to(inputs.device)
        eta = self.eta_func(cat_tensor)
        yita = self.yita_lb + (self.yita_ub - self.yita_lb) * eta[:, 0][:, None]
        
        yq_tensor = torch.cat([u[:,[0,3,4]].sum(dim=1, keepdim=True), chi_t], dim=1) #.float().to(inputs.device) # 5, 7, 8
        beta0 = self.beta_func(yq_tensor)
        # beta = self.beta_lb + (self.beta_ub - self.beta_lb) * beta0
        beta = chi_t * beta0
        
        ay_tensor = u[:,[3,4]]
        tau0 = self.tau_func(ay_tensor)
        tau = self.tau_lb + (self.tau_ub - self.tau_lb) * tau0
        
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
        self.eta_a_loss += self.eta_loss_weight * torch.where(deta[:,1] < 0, deta[:,1] ** 2, torch.zeros_like(deta[:,1]))

        self.eta_y_loss = 0
        self.eta_y_loss += self.eta_loss_weight * torch.where(deta[:,2] < 0, deta[:,2] ** 2, torch.zeros_like(deta[:,2]))

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
        p = np.random.permutation(self.inputs.shape[0])
        inputs_rand = self.inputs[p]

        # randomly sample from input domain
        # t = torch.rand(self.num_samples, 1, requires_grad=True)
        # t = t * (self.t_max - self.t_min) + self.t_min
        # inputs_rand = t.to(inputs.device)
        # inputs_rand = torch.cat([x, t], dim=1).float().to(inputs.device)

        # predict surface fitter at sampled points
        outputs_rand = self.surface_fitter(inputs_rand)

        # compute surface loss
        self.gls_loss_val = self.surface_weight * self.gls_loss(pred, true)
        
        # compute PDE loss at sampled locations
        if self.pde_weight != 0:
            if self.keep_d:
                self.pde_loss_val += self.pde_weight * self.pde_loss(inputs_rand, outputs_rand)
            else:
                self.pde_loss_val += self.pde_weight * self.pde_loss_no_d(inputs_rand, outputs_rand)

        return self.gls_loss_val + self.pde_loss_val
    
#--------------------------------Original COVASIM_BINN by Xin Li------------------------------------------#
#---------------------------------------------------------------------------------------------------------#
#--------------------------------Adaptive Masking---------------------------------------------------------#
class AdaMaskBINNCovasim(nn.Module):
    '''
    Constructs a biologically-informed neural network (BINN) for the Covasim model with 
    an adaptive masking behavior that is composed of the average number of contacts sufficient 
    to transmit infection per unit of time (eta), the effective tracing rate (beta), and the rate 
    of diagnoses from people in quarantine (tau).

    Args:
        params (dict): dictionary of parameters from COVASIM model.
        t_max_real (float): the unscaled maximum time point (t).
        tracing_array (array): array values of tracing probabilities as a function of time (t).
        yita_lb (float): yita lower bound.
        yita_ub (float): yita upper bound.
        keep_d (bool): If true, then include D (diagnosed) in model, otherwise exlcude it.
        chi_type (func): real-valued function of function that affects the quarantining rate.

    '''

    def __init__(self, 
                params, 
                t_max_real, 
                tracing_array, 
                yita_lb=None, 
                yita_ub=None,
                beta_lb=None,
                beta_ub=None,
                tau_lb=None,
                tau_ub=None,
                keep_d=False, 
                chi_type=None,
                eta_deep=False,
                beta_deep=False,
                tau_deep=False):

        super().__init__()

        self.n_com = 9
        # surface fitter
        self.yita_loss = None
        self.yita_lb = yita_lb if yita_lb is not None else 0.0
        self.yita_ub = yita_ub if yita_ub is not None else 1.0
        self.beta_lb = beta_lb if beta_lb is not None else 0.0
        self.beta_ub = beta_ub if beta_ub is not None else 0.5
        self.tau_lb = tau_lb if tau_lb is not None else 0.0
        self.tau_ub =  tau_ub if tau_ub is not None else 0.5
        self.surface_fitter = main_MLP(self.n_com)

        # pde functions/components
        self.eta_mask_func = eta_NN(4, eta_deep)
        self.beta_func = beta_NN(beta_deep)
        self.tau_func = tau_NN(tau_deep)

        # input extrema
        self.t_min = 0.0
        self.t_max = 1.0
        self.t_max_real = t_max_real

        # loss weights
        self.IC_weight = 1e1
        self.surface_weight = 1e2
        self.pde_weight = 1e4
        
        self.weights_c = torch.tensor(np.array([1, 1000, 1, 1000, 1000, 1, 1000, 1, 1000])[None, :], dtype=torch.float)
        
        self.pde_loss_weight = 1e0
        self.eta_loss_weight = 1e5
        self.tau_loss_weight = 1e5

        # proportionality constant
        self.gamma = 0.2

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
        self.avg_masking = params['avg_masking']

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
        residual *= self.weights_c.to(self.inputs.device)

        return torch.mean(residual)

    def pde_loss(self, inputs, outputs, return_mean=True):
        pde_loss = 0
        # unpack inputs
        t = inputs[:, 0][:, None]
        
        # partial derivative computations
        u = outputs.clone()

        chi_t = chi(1 + t * self.t_max_real, self.eff_ub, self.chi_type)

        avg_masking = torch.tensor(self.avg_masking, dtype=torch.float).to(inputs.device)
        avg_masking = avg_masking[(t * self.t_max_real).long()]
        cat_tensor = torch.cat([u[:,[0,3,4]], avg_masking], dim=1).float()
        eta = self.eta_mask_func(cat_tensor)
        yita = self.yita_lb + (self.yita_ub - self.yita_lb) * eta[:, 0][:, None]

        yq_tensor = torch.cat([u[:,[0,3,4]].sum(dim=1, keepdim=True), chi_t], dim=1)
        beta0 = self.beta_func(yq_tensor)
        # beta = self.beta_lb + (self.beta_ub - self.beta_lb) * beta0
        beta = chi_t * beta0

        ay_tensor = u[:,[3,4]]
        tau0 = self.tau_func(ay_tensor)
        tau = self.tau_lb + (self.tau_ub - self.tau_lb) * tau0

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
                RHS = - yita * s  * (a + y) - beta * new_d * self.n_contacts * s + self.alpha * tq
            elif i == 1:
                # dT
                RHS = beta * new_d * self.n_contacts * s - self.alpha * tq
            elif i == 2:
                # dE
                RHS = yita * s * (a + y) - self.gamma * e
            elif i == 3:
                # dA
                RHS = self.p_asymp * self.gamma * e - self.lamda * a - beta * new_d * self.n_contacts * a
            elif i == 4:
                # dY
                RHS = (1 - self.p_asymp) * self.gamma * e - (self.mu + self.lamda + self.delta) * y - beta * new_d * self.n_contacts * y
            elif i == 5:
                # dD
                RHS = self.mu * y + tau * q - self.lamda * d - self.delta * d
            elif i == 6:
                # dQ
                RHS = beta * new_d * self.n_contacts * (a + y) - (tau + self.lamda + self.delta) * q
            elif i == 7:
                # dR
                RHS = self.lamda * (a + y + d + q)
            elif i == 8:
                # dF
                RHS = self.delta * (y + d + q)

            if i in [0, 1, 2, 3, 4, 5, 6]:
                pde_loss += (LHS - RHS) ** 2

        pde_loss *= self.pde_loss_weight

        # constraints on contact_rate function
        yita_final = yita * (a + y)
        deta = Gradient(yita_final, cat_tensor, order=1)
        self.eta_a_loss = 0
        self.eta_a_loss += self.eta_loss_weight * torch.where(deta[:,1] < 0, deta[:,1] ** 2, torch.zeros_like(deta[:,1]))

        self.eta_y_loss = 0
        self.eta_y_loss += self.eta_loss_weight * torch.where(deta[:,2] < 0, deta[:,2] ** 2, torch.zeros_like(deta[:,2]))

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

    def loss(self, pred, true):

        self.gls_loss_val = 0
        self.pde_loss_val = 0

        # load inputs from forward pass
        p = np.random.permutation(self.inputs.shape[0])
        inputs_rand = self.inputs[p]

        # predict surface fitter at sampled points
        outputs_rand = self.surface_fitter(inputs_rand)

        # compute surface loss
        self.gls_loss_val = self.surface_weight * self.gls_loss(pred, true)
        
        # compute PDE loss at sampled locations
        if self.pde_weight != 0:
            self.pde_loss_val += self.pde_weight * self.pde_loss(inputs_rand, outputs_rand)

        return self.gls_loss_val + self.pde_loss_val
    
#--------------------------------Adaptive Masking input------------------------------------------------#
#-------------------------------------------------------------------------------------------------------#
#--------------------------------deep, mask, no main----------------------------------------------------#
class NNComponentsCV(nn.Module):
    '''
    Constructs a neural network that takes in denoised data and numerically approximated derivatives and
    uses them as inputs and loss components for 3 embedded multilayer perceptrons consisting of the average 
    number of contacts sufficient to transmit infection per unit of time (eta), the effective tracing rate (beta), 
    and the rate of diagnoses from people in quarantine (tau).

    Args:
        params (dict): dictionary of parameters from COVASIM model.
        u_tensor (tensor): Tensor object containing matrix of approx. solutions and derivatives wrt time.
        t_max_real (float): the unscaled maximum time point (t).
        tracing_array (array): array values of tracing probabilities as a function of time (t).
        yita_lb (float): yita lower bound.
        yita_ub (float): yita upper bound.
        keep_d (bool): If true, then include D (diagnosed) in model, otherwise exlcude it.
        chi_type (func): real-valued function of function that affects the quarantining rate.
        masking (bool): Indicate if average masking is an input into contact rate.
        eta_deep (bool): Indiactes whether eta is 3 layers (true) or 1 layer (false).
        beta_deep (bool): Indiactes whether beta is 3 layers (true) or 1 layer (false).
        tau_deep (bool): Indiactes whether tau is 3 layers (true) or 1 layer (false).
    '''

    def __init__(self, 
                params, 
                u_tensor, 
                t_max_real, 
                tracing_array, 
                yita_lb=None, 
                yita_ub=None,
                beta_lb=None,
                beta_ub=None,
                tau_lb=None,
                tau_ub=None,
                keep_d=True, 
                chi_type=None,
                mask_input=False,
                eta_deep=False,
                beta_deep=False,
                tau_deep=False):

        super().__init__()

        self.n_com = 9
        
        self.yita_loss = None
        self.yita_lb = yita_lb if yita_lb is not None else 0.0
        self.yita_ub = yita_ub if yita_ub is not None else 1.0
        self.beta_lb = beta_lb if beta_lb is not None else 0.0
        self.beta_ub = beta_ub if beta_ub is not None else 0.5
        self.tau_lb = tau_lb if tau_lb is not None else 0.0
        self.tau_ub =  tau_ub if tau_ub is not None else 0.5
        
        # store denoised data and numerically approximated derivatives
        self.u_tensor = u_tensor
        self.u = u_tensor[:,:,0]
        self.ut = u_tensor[:,:,1]

        # pde functions/components
        self.eta_func = eta_NN(eta_deep) if not mask_input else eta_NN(4, eta_deep)
        self.beta_func = beta_NN(beta_deep)
        self.tau_func = tau_NN(tau_deep)

        # input extrema
        self.t_min = 0.0
        self.t_max = 1.0
        self.t_max_real = t_max_real

        # loss weights
        self.pde_weight = 1e5
            
        self.pde_loss_weight = 1e0
        self.eta_loss_weight = 1e5
        self.tau_loss_weight = 1e5

        # proportionality constant
        self.gamma = 0.2

        self.name = 'denoised_covasim_fitter'

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
        if mask_input:
            self.avg_masking = params['avg_masking'][1:t_max_real]

        self.keep_d = keep_d
        self.mask_input = mask_input

        # if dynamic
        if 'dynamic_tracing' in params:
            self.is_dynamic = True
        self.eff_ub = params['eff_ub']

        self.chi_type = chi_type if chi_type is not None else None


    def forward(self, inputs):
        # cache input batch for pde loss
        self.inputs = inputs
        # What we return here goes into the NN as an input for the loss function.
        # In our case, we use the time points to get the indices of the corresponding denoised data
        # and its approximated time derivatives.
        return self.u[((inputs * self.t_max_real - 1).long()).flatten()]

    def pde_loss(self, inputs, outputs, return_mean=True):

        pde_loss = 0
        # unpack inputs
        t = inputs.clone()

        # surface and partial derivative approximations
        u = outputs[:,:,0].clone()
        ut = outputs[:,:,1].clone()
        
        chi_t = chi(1 + t * self.t_max_real, self.eff_ub, self.chi_type)

        if self.mask_input:
            avg_masking = torch.tensor(self.avg_masking, dtype=torch.float).to(inputs.device)
            avg_masking = avg_masking[(t * self.t_max_real - 1).long()]
            cat_tensor = torch.cat([u[:,[0,3,4]], avg_masking], dim=1).float()
        else:
            cat_tensor = torch.cat([u[:,[0,3,4]]], dim=1).float()
        eta = self.eta_func(cat_tensor)
        yita = self.yita_lb + (self.yita_ub - self.yita_lb) * eta[:, 0][:, None]
        
        yq_tensor = torch.cat([u[:,[0,3,4]].sum(dim=1, keepdim=True), chi_t], dim=1)
        beta0 = self.beta_func(yq_tensor)
        beta = chi_t * beta0
        
        ay_tensor = u[:,[3,4]]
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
            LHS_i = LHS[:,i]
            if i == 0:
                # dS
                RHS = - yita * s  * (a + y) - beta * new_d * self.n_contacts * s + self.alpha * tq
            elif i == 1:
                # dT
                RHS = beta * new_d * self.n_contacts * s - self.alpha * tq
            elif i == 2:
                # dE
                RHS = yita * s * (a + y) - self.gamma * e
            elif i == 3:
                # dA
                RHS = self.p_asymp * self.gamma * e - self.lamda * a - beta * new_d * self.n_contacts * a
            elif i == 4:
                # dY
                RHS = (1 - self.p_asymp) * self.gamma * e - (self.mu + self.lamda + self.delta) * y - beta * new_d * self.n_contacts * y
            elif i == 5:
                # dD
                RHS = self.mu * y + tau * q - self.lamda * d - self.delta * d
            elif i == 6:
                # dQ
                RHS = beta * new_d * self.n_contacts * (a + y) - (tau + self.lamda + self.delta) * q
            elif i == 7:
                # dR
                RHS = self.lamda * (a + y + d + q)
            elif i == 8:
                # dF
                RHS = self.delta * (y + d + q)

            if i in [0, 1, 2, 3, 4, 5, 6]:
                pde_loss += (LHS_i - RHS) ** 2

        pde_loss *= self.pde_loss_weight

        # constraints on contact_rate function
        yita_final = yita * (a + y)
        deta = Gradient(yita_final, cat_tensor, order=1)
        self.eta_a_loss = 0
        self.eta_a_loss += self.eta_loss_weight * torch.where(deta[:,1] < 0, deta[:,1] ** 2, torch.zeros_like(deta[:,1]))

        self.eta_y_loss = 0
        self.eta_y_loss += self.eta_loss_weight * torch.where(deta[:,2] < 0, deta[:,2] ** 2, torch.zeros_like(deta[:,2]))

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

    def loss(self, pred, true):

        self.pde_loss_val = 0

        # load cached inputs from forward pass - QUESTION: What does this mean and do?
        p = np.random.permutation(self.inputs.shape[0])
        inputs_rand = self.inputs[p]
        
        t = (inputs_rand * self.t_max_real - 1).long().flatten().to(self.inputs.device)
        u = self.u[t]
        ut = self.ut[t]

        # get predicted surface fit at sampled points
        outputs_rand = torch.cat([u[:,:,None], ut[:,:,None]], axis=2)
        
        # compute PDE loss at sampled locations
        if self.pde_weight != 0:
            self.pde_loss_val += self.pde_weight * self.pde_loss(inputs_rand, outputs_rand)

        return self.pde_loss_val
#--------------------------------deep, mask, no main-----------------------------------------------------#