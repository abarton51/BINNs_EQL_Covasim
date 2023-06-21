import numpy as np
import torch, pdb
import torch.nn as nn

from Modules.Models.BuildMLP import BuildMLP
from Modules.Activations.SoftplusReLU import SoftplusReLU
from Modules.Utils.Gradient import Gradient

from scipy.stats import beta

class u_MLP(nn.Module):
    
    '''
    Construct MLP surrogate model for the solution of the governing PDE. 
    Includes three hidden layers with 128 sigmoid-activated neurons. Output
    is softplus-activated to keep predicted cell densities non-negative.
    
    Inputs:
        scale (float): output scaling factor, defaults to carrying capacity
    
    Args:
        inputs (torch tensor): x and t pairs with shape (N, 2)
        
    Returns:
        outputs (torch tensor): predicted u values with shape (N, 1)
    '''
    
    def __init__(self, scale=1.7e3):
        
        super().__init__()
        self.scale = scale
        self.mlp = BuildMLP(
            input_features=2, 
            layers=[128, 128, 128, 1],
            activation=nn.Sigmoid(), 
            linear_output=False,
            output_activation=SoftplusReLU())
    
    def forward(self, inputs):
        
        outputs = self.scale * self.mlp(inputs)
        
        return outputs

class D_MLP(nn.Module):
    
    '''
    Construct MLP surrogate model for the unknown diffusivity function. 
    Includes three hidden layers with 32 sigmoid-activated neurons. Output
    is softplus-activated to keep predicted diffusivities non-negative.
    
    Inputs:
        input_features (int): number of input features
        scale        (float): input scaling factor
    
    Args:
        u (torch tensor): predicted u values with shape (N, 1)
        t (torch tensor): optional time values with shape (N, 1)
        
    Returns:
        D (torch tensor): predicted diffusivities with shape (N, 1)
    '''
    
    
    def __init__(self, input_features=1, scale=1.7e3):
        
        super().__init__()
        self.inputs = input_features
        self.scale = scale
        self.min = 0 / (1000**2) / (1/24) # um^2/hr -> mm^2/d
        self.max = 4000 / (1000**2) / (1/24) # um^2/hr -> mm^2/d
        self.mlp = BuildMLP(
            input_features=input_features, 
            layers=[32, 32, 32, 1],
            activation=nn.Sigmoid(), 
            linear_output=False,
            output_activation=SoftplusReLU())
        
    def forward(self, u, t=None):
        
        if t is None:
            D = self.mlp(u/self.scale)
        else:
            D = self.mlp(torch.cat([u/self.scale, t], dim=1))    
        D = self.max * D
        
        return D

class G_MLP(nn.Module):
    
    '''
    Construct MLP surrogate model for the unknown growth function. Includes 
    three hidden layers with 32 sigmoid-activated neurons. Output is linearly 
    activated to allow positive and negative growth values.
    
    Inputs:
        input_features (int): number of input features
        scale        (float): input scaling factor
    
    Args:
        u (torch tensor): predicted u values with shape (N, 1)
        t (torch tensor): optional time values with shape (N, 1)
        
    Returns:
        G (torch tensor): predicted growth values with shape (N, 1)
    '''
    
    def __init__(self, input_features=1, scale=1.7e3):
        
        super().__init__()
        self.inputs = input_features
        self.scale = scale
        self.min = -0.02 / (1/24) # 1/hr -> 1/d
        self.max = 0.1 / (1/24) # 1/hr -> 1/d
        self.mlp = BuildMLP(
            input_features=input_features, 
            layers=[32, 32, 32, 1],
            activation=nn.Sigmoid(), 
            linear_output=True)
    
    def forward(self, u, t=None):
        
        if t is None:
            G = self.mlp(u/self.scale)
        else:
            G = self.mlp(torch.cat([u/self.scale, t], dim=1))
        G = self.max * G
        
        return G

class NoDelay(nn.Module):
    
    '''
    Trivial delay function.
    
    Args:
        t (torch tensor): time values with shape (N, 1)
        
    Returns:
        T (torch tensor): ones with shape (N, 1)
    '''
    
    
    def __init__(self):
        
        super().__init__()
        
    def forward(self, t):
        
        T = torch.ones_like(t)
        
        return T

class T_MLP(nn.Module):
    
    '''
    Construct MLP surrogate model for the unknown time delay function. 
    Includes three hidden layers with 32 sigmoid-activated neurons. Output is 
    linearly sigmoid-activated to constrain outputs to between 0 and 1.
    
    Args:
        t (torch tensor): time values with shape (N, 1)
        
    Returns:
        T (torch tensor): predicted delay values with shape (N, 1)
    '''
    
    
    def __init__(self):
        
        super().__init__()
        self.mlp = BuildMLP(
            input_features=1, 
            layers=[32, 32, 32, 1],
            activation=nn.Sigmoid(), 
            linear_output=False,
            output_activation=nn.Sigmoid())
        
    def forward(self, t):
        
        T = self.mlp(t) 
        
        return T
    
class BINN(nn.Module):
    
    '''
    Constructs a biologically-informed neural network (BINN) composed of
    cell density dependent diffusion and growth MLPs with an optional time 
    delay MLP.
    
    Inputs:
        delay (bool): whether to include time delay MLP
        
    
    '''
    
    def __init__(self, delay=False):
        
        super().__init__()
        
        # surface fitter
        self.surface_fitter = u_MLP()
        
        # pde functions
        self.diffusion = D_MLP()
        self.growth = G_MLP()
        self.delay1 = T_MLP() if delay else NoDelay()
        self.delay2 = self.delay1
        
        # parameter extrema
        self.D_min = self.diffusion.min
        self.D_max = self.diffusion.max
        self.G_min = self.growth.min
        self.G_max = self.growth.max
        self.K = 1.7e3
        
        # input extrema
        self.x_min = 0.075 
        self.x_max = 1.875 
        self.t_min = 0.0
        self.t_max = 2.0

        # loss weights
        self.IC_weight = 1e1
        self.surface_weight = 1e0
        self.pde_weight = 1e0
        self.D_weight = 1e10 / self.D_max
        self.G_weight = 1e10 / self.G_max
        self.T_weight = 1e10 
        self.dDdu_weight = self.D_weight * self.K
        self.dGdu_weight = self.G_weight * self.K
        self.dTdt_weight = self.T_weight * 2.0
        
        # proportionality constant
        self.gamma = 0.2

        # number of samples for pde loss
        self.num_samples = 10000
        
        # model name
        self.name = 'TmlpDmlp_TmlpGmlp' if delay else 'Dmlp_Gmlp'
    
    def forward(self, inputs):
        
        # cache input batch for pde loss
        self.inputs = inputs
        
        return self.surface_fitter(self.inputs)
    
    def gls_loss(self, pred, true):
        
        residual = (pred - true)**2
        
        # add weight to initial condition
        residual *= torch.where(self.inputs[:, 1][:, None]==0, 
                                self.IC_weight*torch.ones_like(pred), 
                                torch.ones_like(pred))
        
        # proportional GLS weighting
        residual *= pred.abs().clamp(min=1.0)**(-self.gamma)
        
        return torch.mean(residual)
    
    def pde_loss(self, inputs, outputs, return_mean=True):
        
        # unpack inputs
        x = inputs[:, 0][:,None]
        t = inputs[:, 1][:,None]

        # partial derivative computations 
        u = outputs.clone()
        d1 = Gradient(u, inputs, order=1)
        ux = d1[:, 0][:, None]
        ut = d1[:, 1][:, None]

        # diffusion
        if self.diffusion.inputs == 1:
            D = self.diffusion(u)
        else:
            D = self.diffusion(u, t)

        # growth
        if self.growth.inputs == 1:
            G = self.growth(u)
        else:
            G = self.growth(u, t)

        # time delays
        T1 = self.delay1(t)
        T2 = self.delay2(t)

        # Fisher-KPP equation
        LHS = ut
        RHS = T1*Gradient(D*ux, inputs)[:, 0][:,None] + T2*G*u
        pde_loss = (LHS - RHS)**2

        # constraints on learned parameters
        self.D_loss = 0
        self.G_loss = 0
        self.T_loss = 0
        self.D_loss += self.D_weight*torch.where(
            D < self.D_min, (D-self.D_min)**2, torch.zeros_like(D))
        self.D_loss += self.D_weight*torch.where(
            D > self.D_max, (D-self.D_max)**2, torch.zeros_like(D))
        self.G_loss += self.G_weight*torch.where(
            G < self.G_min, (G-self.G_min)**2, torch.zeros_like(G))
        self.G_loss += self.G_weight*torch.where(
            G > self.G_max, (G-self.G_max)**2, torch.zeros_like(G))

        # derivative constraints on eligible parameter terms
        try:
            dDdu = Gradient(D, u, order=1)
            self.D_loss += self.dDdu_weight*torch.where(
                dDdu < 0.0, dDdu**2, torch.zeros_like(dDdu))
        except:
            pass
        try:
            dGdu = Gradient(G, u, order=1)
            self.G_loss += self.dGdu_weight*torch.where(
                dGdu > 0.0, dGdu**2, torch.zeros_like(dGdu))
        except:
            pass
        try:
            dTdt = Gradient(T1, t, order=1)
            self.T_loss += self.dTdt_weight*torch.where(
                dTdt < 0.0, dTdt**2, torch.zeros_like(dTdt))
        except:
            pass
        
        if return_mean:
            return torch.mean(pde_loss + self.D_loss + self.G_loss + self.T_loss)
        else:
            return pde_loss + self.D_loss + self.G_loss + self.T_loss
    
    def loss(self, pred, true):
        
        self.gls_loss_val = 0
        self.pde_loss_val = 0
        
        # load cached inputs from forward pass
        inputs = self.inputs
        
        # randomly sample from input domain
        x = torch.rand(self.num_samples, 1, requires_grad=True) 
        x = x*(self.x_max - self.x_min) + self.x_min
        t = torch.rand(self.num_samples, 1, requires_grad=True)
        t = t*(self.t_max - self.t_min) + self.t_min
        inputs_rand = torch.cat([x, t], dim=1).float().to(inputs.device)
        
        # predict surface fitter at sampled points
        outputs_rand = self.surface_fitter(inputs_rand)
        
        # compute surface loss
        self.gls_loss_val = self.surface_weight*self.gls_loss(pred, true)
        
        # compute PDE loss at sampled locations
        self.pde_loss_val += self.pde_weight*self.pde_loss(inputs_rand, outputs_rand)
        
        return self.gls_loss_val + self.pde_loss_val
    
#--------------------------------DRUMS--------------------------------------#
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
        outputs (torch tensor): predicted contact rate values with shape (N, 1)
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
    is sigmoid-activated to keep predicted tracing rates between 0 and 1.

    Inputs:
        N/A

    Args:
        inputs (torch tensor): S + A + Y with shape (N, 1)

    Returns:
        outputs (torch tensor): predicted beta(S + A + Y) values with shape (N, 1)
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
        outputs (torch tensor): predicted tau values with shape (N, 1)
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
        outputs (torch tensor): predicted mu values with shape (N, 1)
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
        # self.dfdt_weight = 1e10
        # self.drdt_weight = 1e10
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
        # d1 = Gradient(u, inputs, order=1)
        # ut = d1[:, 0][:, None]
        chi_t = chi(1 + t * self.t_max_real, self.eff_ub, self.chi_type)
        # chi_t = torch.nn.functional.interpolate()
        cat_tensor = torch.cat([u[:,[0,3,4]]], dim=1).float().to(inputs.device) # t,
        eta = self.eta_func(cat_tensor)
        # contact_rate = self.contact_rate(u[:,[0,3,4]])  # what to input contact_rate MLP
        yita = self.yita_lb + (self.yita_ub - self.yita_lb) * eta[:, 0][:, None]
        # ay_tensor = torch.Tensor(u[:,[3,4]]).float().to(inputs.device)
        yq_tensor = torch.cat([u[:,[0,3,4]].sum(dim=1, keepdim=True), chi_t], dim=1).float().to(inputs.device) # 5, 7, 8
        beta0 = self.beta_func(yq_tensor)
        # beta = self.beta_lb + (self.beta_ub - self.beta_lb) * beta0
        beta = chi_t * beta0
        ay_tensor = torch.Tensor(u[:,[3,4]]).float().to(inputs.device)
        tau0 = self.tau_func(ay_tensor)
        tau = self.tau_lb + (self.tau_ub - self.tau_lb) * tau0 # quarantine_test[:, 0][:, None]
        # theta = 0.1 * contact_rate[:, 1][:, None]
        # epsilon = 0.001 * contact_rate[:, 2][:, None]
        # STEAYDQRF model, loop through each compartment
        s, tq, e, a, y, d, q, r, f = u[:, 0][:, None], u[:, 1][:, None], u[:, 2][:, None], u[:, 3][:, None],\
                                    u[:, 4][:, None], u[:, 5][:, None], u[:, 6][:, None], u[:, 7][:, None],\
                                    u[:, 8][:, None]
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

        # self.eta_chi_loss = 0
        # self.eta_chi_loss += self.eta_loss_weight * torch.where(deta[:,3] > 0, deta[:,3] ** 2, torch.zeros_like(deta[:,3]))

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
#--------------------------------DRUMS--------------------------------------#

class main_MLP(nn.Module):
    '''
    Construct MLP surrogate model for the solution of the governing PDE.
    Includes three hidden layers with 128 sigmoid-activated neurons. Output
    is softplus-activated to keep predicted cell densities non-negative.

    Inputs:
        scale (float): output scaling factor, defaults to carrying capacity

    Args:
        inputs (torch tensor): t with shape (N, 1)

    Returns:
        outputs (torch tensor): predicted u values with shape (N, 9)
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
    Construct MLP surrogate model for the solution of the governing PDE.
    Includes three hidden layers with 128 sigmoid-activated neurons. Output
    is softplus-activated to keep predicted cell densities non-negative.

    Inputs:
        scale (float): output scaling factor, defaults to carrying capacity

    Args:
        inputs (torch tensor): t with shape (N, 9)

    Returns:
        outputs (torch tensor): predicted u values with shape (N, 3)
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
    Construct MLP surrogate model for the solution of the governing PDE.
    Includes three hidden layers with 128 sigmoid-activated neurons. Output
    is softplus-activated to keep predicted cell densities non-negative.

    Inputs:
        scale (float): output scaling factor, defaults to carrying capacity

    Args:
        inputs (torch tensor): t with shape (N, 9)

    Returns:
        outputs (torch tensor): predicted u values with shape (N, 3)
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
    Construct MLP surrogate model for the solution of the governing PDE.
    Includes three hidden layers with 128 sigmoid-activated neurons. Output
    is softplus-activated to keep predicted cell densities non-negative.

    Inputs:
        scale (float): output scaling factor, defaults to carrying capacity

    Args:
        inputs (torch tensor): t with shape (N, 9)

    Returns:
        outputs (torch tensor): predicted u values with shape (N, 3)
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
        # self.dfdt_weight = 1e10
        # self.drdt_weight = 1e10
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
        # d1 = Gradient(u, inputs, order=1)
        # ut = d1[:, 0][:, None]
        chi_t = chi(1 + t * self.t_max_real, self.eff_ub, self.chi_type)
        # chi_t = torch.nn.functional.interpolate()
        cat_tensor = torch.cat([u[:,[0,3,4]]], dim=1).float().to(inputs.device) # t,
        eta = self.eta_func(cat_tensor)
        # contact_rate = self.contact_rate(u[:,[0,3,4]])  # what to input contact_rate MLP
        yita = self.yita_lb + (self.yita_ub - self.yita_lb) * eta[:, 0][:, None]
        # ay_tensor = torch.Tensor(u[:,[3,4]]).float().to(inputs.device)
        yq_tensor = torch.cat([u[:,[0,3,4]].sum(dim=1, keepdim=True), chi_t], dim=1).float().to(inputs.device) # 5, 7, 8
        beta0 = self.beta_func(yq_tensor)
        # beta = self.beta_lb + (self.beta_ub - self.beta_lb) * beta0
        beta = chi_t * beta0
        ay_tensor = torch.Tensor(u[:,[3,4]]).float().to(inputs.device)
        tau0 = self.tau_func(ay_tensor)
        tau = self.tau_lb + (self.tau_ub - self.tau_lb) * tau0 # quarantine_test[:, 0][:, None]
        # theta = 0.1 * contact_rate[:, 1][:, None]
        # epsilon = 0.001 * contact_rate[:, 2][:, None]
        # STEAYDQRF model, loop through each compartment
        s, tq, e, a, y, d, q, r, f = u[:, 0][:, None], u[:, 1][:, None], u[:, 2][:, None], u[:, 3][:, None],\
                                    u[:, 4][:, None], u[:, 5][:, None], u[:, 6][:, None], u[:, 7][:, None],\
                                    u[:, 8][:, None]
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

        # self.eta_chi_loss = 0
        # self.eta_chi_loss += self.eta_loss_weight * torch.where(deta[:,3] > 0, deta[:,3] ** 2, torch.zeros_like(deta[:,3]))

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


class eta_MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.mlp = BuildMLP(
            input_features=4,
            layers=[256, 1],
            activation=nn.ReLU(),
            linear_output=False,
            output_activation=nn.Sigmoid())  #  SoftplusReLU()

    def forward(self, inputs):
        outputs = self.mlp(inputs)

        return outputs

class varying_parameter_MLP(nn.Module):

    def __init__(self, num_outputs):
        super().__init__()
        self.mlp = BuildMLP(
            input_features=1,
            layers=[256, 128, num_outputs],
            activation=nn.ReLU(),
            linear_output=False,
            output_activation=nn.Sigmoid())  #  SoftplusReLU()

    def forward(self, inputs):
        outputs = self.mlp(inputs)

        return outputs

# def eta_factor(t):
#
#     # self.eta_factor = np.ones(t_max_real)
#     # self.eta_factor[:21] = 1.33  # 30% increase in first 3 weeks
#     # self.eta_factor[7*11:7*25] = 1.38  # 38% increase in weeks 11-25
#     # self.eta_factor[7*38:7*40] = 1.66  # 66% increase in weeks 38-40
#     factor = torch.where(t < 21.0, 1.33 , 1.0)
#     factor = torch.where(torch.logical_and(7.0*11 <= t, t < 7.0 * 25), 1.38 * torch.ones_like(factor), factor)
#     factor = torch.where(torch.logical_and(7.0*38 <= t, t < 7.0 * 40), 1.66 * torch.ones_like(factor), factor)
#     return factor

def hosp_factor(t):
    # dynamic probability of hospitalization reduce by 24% over days 150-180
    return torch.where(torch.logical_and(150.0 <= t, t < 180.0), 0.24, 1.0)

def alpha_factor(t):
    # immunity escape at days 150, 157, 164 and 171
    peak_tensor = (0.8 * 0.25 + 0.2/100) * torch.ones_like(t)
    alpha_tensor = torch.where(torch.isclose(t, 150.0 * torch.ones_like(t)), peak_tensor, 0.2/100 * torch.ones_like(t))
    alpha_tensor = torch.where(torch.isclose(t, 157.0 * torch.ones_like(t)), peak_tensor , alpha_tensor)
    alpha_tensor = torch.where(torch.isclose(t, 164.0 * torch.ones_like(t)), peak_tensor, alpha_tensor)
    alpha_tensor = torch.where(torch.isclose(t, 171.0 * torch.ones_like(t)), peak_tensor, alpha_tensor)
    return alpha_tensor

def mu_factor(t):
    # dynamic time in hospital reduce by 42% over days 150-170
    return torch.where(torch.logical_and(150.0 <= t, t < 170.0), 1.0/(10.4 * 0.58), 1.0/10.4)

class BINNCOVSIM(nn.Module):
    '''
    Constructs a biologically-informed neural network (BINN) composed of
    cell density dependent diffusion and growth MLPs with an optional time
    delay MLP.

    Inputs:
        delay (bool): whether to include time delay MLP


    '''

    def __init__(self, t_max_real):

        super().__init__()

        self.n_com = 8
        # surface fitter
        self.yita_loss = None
        self.yita_lb =  0.0
        self.yita_ub = 0.5
        # self.beta_lb = 0.0
        # self.beta_ub = 1e-4
        # self.h_rate_lb = 0.0
        # self.h_rate_ub = 1
        self.surface_fitter = main_MLP(self.n_com)

        # pde functions
        self.ETA = eta_MLP()
        self.varying_parameters = varying_parameter_MLP(num_outputs=3)

        # input extrema
        self.t_min = 0.0
        self.t_max = 1.0
        self.t_max_real = t_max_real # what is the t_max in the real timescale

        # loss weights
        self.IC_weight = 1e1
        self.surface_weight = 1e2
        self.pde_weight =  1e6 #  # 1e4
        # self.kl_loss = nn.KLDivLoss()
        self.weights_c = torch.tensor(np.array([1, 1000, 1, 1, 1, 1, 1000, 1000])[None, :], dtype=torch.float)
        self.yita_weight = 0
        self.dfdt_weight = 1e10
        self.drdt_weight = 1e10
        self.pde_loss_weight = 1e0 #1e0


        # number of samples for pde loss
        self.num_samples = 1000

        self.name = 'COVSIM_fitter'

        self.population = 10.5e6
        self.gamma = 1/4.3
        self.lamda = 1 # /0.5
        self.epsilon = 1/2
        self.theta = 1/3
        self.alpha = 0.2/100
        self.mu = 1/10.4
        self.h_rate = 0.1
        self.d_rate = 0.2

        # dynamic transmission rate
        # self.eta_factor = lambda t: eta_factor(t)
        # self.hosp_factor = lambda t: hosp_factor(t)
        # self.alpha = lambda t: alpha_factor(t)
        # self.mu = lambda t: mu_factor(t)


        self.symp_prob = 0.67


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
        # d1 = Gradient(u, inputs, order=1)
        # ut = d1[:, 0][:, None]

        # theta = 0.1 * contact_rate[:, 1][:, None]
        # epsilon = 0.001 * contact_rate[:, 2][:, None]
        # SEIR model, loop through each compartment
        s, e, p, a, y, r, h, d = u[:, 0][:, None], u[:, 1][:, None], u[:, 2][:, None], u[:, 3][:, None],\
                                    u[:, 4][:, None], u[:, 5][:, None], u[:, 6][:, None], u[:, 7][:, None]
        i = p + a + y
        cat_tensor = torch.cat([t, s, i, h], dim=1).float().to(inputs.device)  # what to input contact_rate MLP
        contact_rate = self.ETA(cat_tensor)
        varying_parameters = self.varying_parameters(t) # h, d, beta
        h_rate, d_rate, beta = varying_parameters[:,0][:, None], varying_parameters[:,1][:,None], varying_parameters[:,2][:,None]
        # hosp_factor = self.hosp_factor(t * self.t_max_real)
        # eta_factor = self.eta_factor(t * self.t_max_real) # todo:remove
        # alpha = self.alpha(t * self.t_max_real)
        # mu = self.mu(t * self.t_max_real)
        eta = self.yita_lb + (self.yita_ub - self.yita_lb) * contact_rate[:, 0][:, None]
        # beta = self.beta_lb + (self.beta_ub - self.beta_lb) * vacc_rate
        # h_rate_scaled = self.h_rate_lb + (self.h_rate_ub - self.h_rate_lb) * h_rate
        RHS_sum = 0.0
        for i in range(self.n_com):
            d1 = Gradient(u[:, i], inputs, order=1)
            ut = d1[:, 0][:, None]
            LHS = ut / self.t_max_real
            if i == 0:
                # dS
                RHS = - eta * s * i  + self.alpha * r - beta * s
            elif i == 1:
                # dE
                RHS = eta * s * i - self.gamma * e
            elif i == 2:
                # dP
                RHS = self.gamma * e - self.lamda * p
            elif i == 3:
                # dA
                RHS = (1-self.symp_prob) * self.lamda * p - self.epsilon * a
            elif i == 4:
                # dY
                RHS = self.symp_prob * self.lamda * p - self.theta * y
            elif i == 5:
                # dR
                RHS = self.epsilon * a + (1 - h_rate) * self.theta * y + \
                      (1 - d_rate) * self.mu * h - self.alpha * r + beta * s
            elif i == 6:
                # dH
                RHS = h_rate * self.theta * y - self.mu * h
            elif i == 7:
                # dD
                RHS = d_rate * self.mu * h
            if i in [0, 1, 2, 3, 4, 5, 6, 7]:
                pde_loss += (LHS - RHS) ** 2
            RHS_sum += RHS
        # print(RHS_sum.median())
        #     pde_loss += (LHS - RHS) ** 2
        pde_loss *= self.pde_loss_weight
        # print(RHS_sum.sum())
        # constraints on learned parameters
        self.yita_loss = 0
        # self.G_loss = 0
        # self.T_loss = 0
        # self.yita_loss += self.yita_weight * torch.where(
        #     D < self.D_min, (D - self.D_min) ** 2, torch.zeros_like(D))
        # self.D_loss += self.D_weight * torch.where(
        #     D > self.D_max, (D - self.D_max) ** 2, torch.zeros_like(D))
        # self.G_loss += self.G_weight * torch.where(
        #     G < self.G_min, (G - self.G_min) ** 2, torch.zeros_like(G))
        # self.G_loss += self.G_weight * torch.where(
        #     G > self.G_max, (G - self.G_max) ** 2, torch.zeros_like(G))

        # derivative constraints on eligible parameter terms

        if return_mean:
            return torch.mean(pde_loss)  #  + self.dfdt_loss + self.drdt_loss
        else:
            return pde_loss  # + self.D_loss + self.G_loss + self.T_loss

    def pde_loss_no_d(self, inputs, outputs, return_mean=True):
        """ pde loss for the case of removing compartment D"""
        pde_loss = 0
        # unpack inputs
        t = inputs[:, 0][:, None]

        # partial derivative computations
        u = outputs.clone()
        # d1 = Gradient(u, inputs, order=1)
        # ut = d1[:, 0][:, None]

        contact_rate = self.contact_rate(u[:,[0,3,4]])  # what to input contact_rate MLP
        yita = self.yita_lb + (self.yita_ub - self.yita_lb) * contact_rate[:, 0][:, None]
        tau = self.tau_lb + (self.tau_ub - self.tau_lb) * self.quarantine_test_prob(u[:,[3,4]])
        # SEIR model, loop through each compartment
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
        # RHS = T1 * Gradient(D * ux, inputs)[:, 0][:, None] + T2 * G * u
            if i in [0, 1, 2, 3, 4, 5]:
                pde_loss += (LHS - RHS) ** 2
        #     pde_loss += (LHS - RHS) ** 2
        pde_loss *= self.pde_loss_weight


        if return_mean:
            return torch.mean(pde_loss)  #  + self.dfdt_loss + self.drdt_loss
        else:
            return pde_loss  # + self.D_loss + self.G_loss + self.T_loss

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
            self.pde_loss_val += self.pde_weight * self.pde_loss(inputs_rand, outputs_rand)


        return self.gls_loss_val + self.pde_loss_val
