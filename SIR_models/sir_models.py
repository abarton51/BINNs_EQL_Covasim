import sys
sys.path.append('../')
import numpy as np

from Modules.Utils.PDESolver import chi_func

class SIR:
    """

    SIR model.

    Args:
        population (int): total population
        rho (float)
        sigma (float)
    """
    # Model name
    NAME = "SIR"
    # names of parameters
    PARAMETERS = ["rho", "sigma"]
    DAY_PARAMETERS = ["1/beta [day]", "1/gamma [day]"]
    # Variable names in (non-dim, dimensional) ODEs
    VAR_DICT = {
        "x": 'Susceptible',
        "y": 'Infectious',
        "z": 'Recovered'
    }
    VARIABLES = list(VAR_DICT.values())

    def __init__(self, population, rho, sigma):
        # Total population
        self.population = population
        # Non-dim parameters
        self.rho = rho
        self.sigma = sigma
        self.non_param_dict = {"rho": rho, "sigma": sigma}

    def __call__(self, t, X):
        """
        Return the list of dS/dt (tau-free) etc.

        Args:
            t (int): time steps
            X (numpy.array): values of the model variables

        Returns:
            (np.array)
        """
        n = self.population
        s, i, *_ = X
        dsdt = 0 - self.rho * s * i / n
        drdt = self.sigma * i
        didt = 0 - dsdt - drdt
        return np.array([dsdt, didt, drdt])


class SEIRD:
    """

    SEIRD model.

    Args:
        population (int): total population
        rho (float)
        sigma (float)
    """
    # Model name
    NAME = "SEIRD"
    # names of parameters
    # PARAMETERS = ["rho", "sigma"]
    # Variable names in (non-dim, dimensional) ODEs
    VAR_DICT = {
        "x": 'S',
        "y": 'E',
        "z": 'I',
        "u": 'R',
        "v": 'D'

    }
    VARIABLES = list(VAR_DICT.values())

    def __init__(self, population, yita, gamma, delta, lamda, yita_1):
        # Total population
        self.population = population
        # Non-dim parameters
        self.yita = yita
        self.gamma = gamma
        self.delta = delta
        self.lamda = lamda
        self.yita_1 = yita_1

    def __call__(self, t, X):
        """
        Return the list of dS/dt (tau-free) etc.

        Args:
            t (int): time steps
            X (numpy.array): values of th model variables

        Returns:
            (np.array)
        """
        n = self.population
        s, e, i, r, d = X
        dsdt = 0 - self.yita * s * i / n - self.yita_1 * i * i / n
        dedt = self.yita * s * i / n + self.yita_1 * i * i / n - self.gamma * e
        didt = self.gamma * e - (self.delta + self.lamda) * i
        drdt = self.lamda * i
        dddt = 0 - dsdt - dedt - didt - drdt
        return np.array([dsdt, dedt, didt, drdt, dddt])


class STEAYDQRF:
    # Model name
    NAME = "STEAYDQRF"
    # names of parameters
    # PARAMETERS = ["rho", "sigma"]
    # Variable names in (non-dim, dimensional) ODEs
    VARIABLES = list(NAME)

    def __init__(self, **kwargs):
        # Total population
        self.population = kwargs['population']
        # Non-dim parameters
        self.alpha = kwargs['alpha']
        self.beta = kwargs['beta']
        # self.theta = kwargs['theta']
        self.yita = kwargs['yita']
        # self.epsilon = kwargs['epsilon']
        self.gamma = kwargs['gamma']
        self.mu = kwargs['mu']
        self.tau = kwargs['tau']
        self.lamda = kwargs['lamda']
        self.delta = kwargs['delta']
        self.p_asymp = kwargs['p_asymp']
        self.n_contacts = kwargs['n_contacts']
        self.chi_type = kwargs['chi_type']

    def __call__(self, t, X):
        """
        Return the list of dS/dt (tau-free) etc.

        Args:
            t (int): time steps
            X (numpy.array): values of th model variables

        Returns:
            (np.array)
        """
        n = self.population
        s, tq, e, a, y, d, q, r, f = X

        chi = chi_func(t, self.chi_type)
        # print(t, chi)
        new_d = self.mu * y + self.tau * q
        dsdt = 0 - self.yita * s * (a + y) / n - self.n_contacts * self.beta * chi * new_d * s / n + self.alpha * tq
        dtqdt = self.n_contacts * self.beta * chi * new_d * s / n - self.alpha * tq
        dedt = self.yita * s * (a + y)/n - self.gamma * e
        dadt = self.p_asymp * self.gamma * e - self.lamda * a - self.n_contacts * self.beta * chi * new_d * a / n
        dydt = (1 - self.p_asymp) * self.gamma * e - (
                    self.mu + self.lamda + self.delta) * y - self.n_contacts * self.beta * chi * new_d * y / n
        dddt = new_d - (self.lamda + self.delta) * d
        dqdt = self.n_contacts * self.beta * chi * new_d *  (a + y) / n - (self.tau + self.lamda + self.delta) * q
        drdt = self.lamda * (a + y + d + q)
        dfdt = self.delta * (d + q + y)

        rhs = np.array([dsdt, dtqdt, dedt, dadt, dydt, dddt, dqdt, drdt, dfdt])
        # assert rhs.sum() == 0
        # print(rhs.sum())
        return rhs
