import joblib
import pandas as pd

import covasim.covasim as cv
import covasim.covasim.utils as cvu
import pylab as pl
import sciris as sc
import numpy as np
from Notebooks.utils import get_case_name


class store_compartments(cv.Analyzer):

    def __init__(self, keep_d, *args, **kwargs):
        super().__init__(*args, **kwargs) # This is necessary to initialize the class properly
        self.t = []
        self.S = []  # susceptible and not quarantined
        self.T = []  # susceptible and quarantined
        self.E = []  # exposed but not infectious
        self.I = [] # infectious
        self.A = []  # asymptomatic
        self.Y = []  # symptomatic
        self.D = []  # infectious and diagnosed
        self.Q = []  # infectious and quarantined
        self.R = []  # recovered
        self.F = []  # fatal/dead
        self.keep_D = keep_d
        return

    def apply(self, sim):
        ppl = sim.people # Shorthand
        self.t.append(sim.t)
        self.S.append((ppl.susceptible * (1 - ppl.recovered) * (1-ppl.quarantined)).sum()) # remove recovered from susceptible
        self.T.append((ppl.susceptible * (1 - ppl.recovered) * ppl.quarantined).sum())
        self.E.append((ppl.exposed * (1 - ppl.infectious)).sum())
        self.I.append(ppl.infectious.sum())
        # self.A.append((ppl.infectious * (1-ppl.symptomatic)).sum())
        self.Y.append((ppl.infectious * (~np.isnan(ppl.date_symptomatic))).sum() - (ppl.infectious * ppl.diagnosed).sum() - (ppl.infectious * ppl.quarantined).sum())
        if self.keep_D:
            self.D.append((ppl.infectious * ppl.diagnosed).sum())
            self.Q.append((ppl.infectious * ppl.quarantined).sum())
            self.A.append(self.I[-1] - self.Y[-1] - self.D[-1] - self.Q[-1])
            assert self.I[-1] == self.A[-1] + self.Y[-1] + self.D[-1] + self.Q[-1]
        else:
            self.Q.append((ppl.infectious * ppl.quarantined).sum() + (ppl.infectious * ppl.diagnosed).sum())
            self.A.append(self.I[-1] - self.Y[-1] - self.Q[-1])
            assert self.I[-1] == self.A[-1] + self.Y[-1] + self.Q[-1]
        self.R.append(ppl.recovered.sum())
        self.F.append(ppl.dead.sum())
        return

    def plot(self, given_str):
        pl.figure()
        for c in given_str:
            pl.plot(self.t, self.__getattribute__(c), label=c)
        pl.legend()
        pl.xlabel('Day')
        pl.ylabel('People')
        sc.setylim() # Reset y-axis to start at 0
        sc.commaticks() # Use commas in the y-axis labels
        pl.show()
        return

def test_data_generator():
    # Define the testing and contact tracing interventions
    test_scale = 0.1
    # test_quarantine_scale = 0.1   min(test_scale * 4, 1)
    tp = cv.test_prob(symp_prob=test_scale, asymp_prob=0.001, symp_quar_prob=0.8,
                      asymp_quar_prob=0.3, quar_policy='daily')
    trace_prob = dict(h=1.0, s=0.5, w=0.5, c=0.3)
    trace_scale = 0.1
    trace_prob = {key: val*trace_scale for key,val in trace_prob.items()}
    ct = cv.contact_tracing(trace_probs=trace_prob)
    keep_d = True
    population = int(50e3)
    case_name = get_case_name(population, test_scale, trace_scale, keep_d)
    # Define the default parameters
    pars = dict(
        pop_type      = 'hybrid',
        pop_size      = population,
        pop_infected  = 100,
        start_day     = '2020-02-01',
        end_day       = '2020-08-01',
        interventions = [tp, ct],
        analyzers=store_compartments(keep_d, label='get_compartments')
    )

    # Create, run, and plot the simulation
    sim = cv.Sim(pars)
    sim.run()
    sim.plot(to_plot=['new_infections', 'new_tests', 'new_diagnoses', 'cum_diagnoses', 'new_quarantined', 'test_yield'])

    get_data = sim.get_analyzer('get_compartments')  # Retrieve by label

    compartments = 'STEAYDQRF' if get_data.keep_D else 'STEAYQRF'
    get_data.plot(compartments)
    res = None
    for c in compartments:
        if res is None:
            res = np.array(get_data.__getattribute__(c))
        else:
            res += np.array(get_data.__getattribute__(c))
    assert res.max() == sim['pop_size']
    data = pd.DataFrame()
    for c in compartments:
        data[c] = np.array(get_data.__getattribute__(c))

    # prepare the corresponding parameters of compartmental model
    population = sim['pop_size']
    params = {}
    params['population'] = population
    contacts = sim.pars['contacts']
    params['alpha'] = 1 / sim.pars['quar_period']
    params['beta'] = sum([ct.trace_probs[key] * val for key, val in contacts.items()]) / sum(contacts.values())
    params['gamma'] = 1 / sim.pars['dur']['exp2inf']['par1']
    params['mu'] = tp.symp_prob
    # params['tau'] = tp.symp_quar_prob # tau can not be directly determined
    params['tau_lb'] = 0 # tp.asymp_quar_prob
    params['tau_ub'] = tp.symp_quar_prob
    params['lamda'] = 1 / 10.0
    params['p_asymp'] = 1 - sim.people.symp_prob.mean()
    params['n_contacts'] = sum(contacts.values())
    severe_probs = sim.people.severe_prob.mean()
    crit_probs = sim.people.crit_prob.mean()
    params['delta'] = severe_probs * crit_probs * 1 / sim.pars['dur']['crit2die']['par1']
    params['data'] = data
    file_name = '../Data/covasim_data/covasim_'+ case_name + '.joblib'
    joblib.dump(params, file_name, compress=True)