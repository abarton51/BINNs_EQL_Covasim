import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


class SIRSimulator:
    def __init__(self, model):
        self._model = model  # SIR-like model instance

    def _run(self, step_n, y0_dict, population):
        """
        Solve an initial value problem for a SIR-derived ODE model.

        Args:
            step_n (int): the number of steps
            y0_dict (dict[str, int]): initial values of dimensional variables, including Susceptible
            population (int): total population

        Returns:
            pandas.DataFrame: numerical solution
                Index
                    reset index: time steps
                Columns
                    (int): dimensional variables of the model
        """
        tstart, dt, tend = 0, 1, step_n
        variables = self._model.VARIABLES[:]
        initials = [y0_dict[var] for var in variables]
        sol = solve_ivp(
            fun=self._model,
            t_span=[tstart, tend],
            y0=np.array(initials, dtype=np.int64),
            t_eval=np.arange(tstart, tend + dt, dt),
            dense_output=False
        )
        y_df = pd.DataFrame(data=sol["y"].T.copy(), columns=variables)
        return y_df.round().astype(np.int64)
