import os.path

import joblib
import pandas as pd
from scipy.stats import beta
from functools import reduce

import covasim.covasim as cv
import covasim.covasim.utils as cvu
import pylab as pl
import sciris as sc
import numpy as np
import matplotlib.pyplot as plt
from Notebooks.utils import get_case_name, import_new_variants
import matplotlib
matplotlib.use('Agg')

eff_lb, eff_ub = 0.0, 0.3
chi_type = 'sin'

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
        # pl.show()
        pl.savefig('../Notebooks/figs/' + 'compartments' + '.png')
        return

def get_dynamic_eff(ftype, eff_ub):

    if ftype == 'linear':
        t = np.arange(0, 200, 1)
        slope = eff_ub / 75
        res = np.zeros(len(t))
        res += (t < 75) * slope * (t + 1)
        res += (t >= 75) * (t < 150) * eff_ub
        res -= (t >= 75) * (t < 150) * (slope * (t - 75 + 1))
    elif ftype == 'sin':
        times = np.arange(0, 200, 1)
        rad_times =  times * np.pi / 40.
        res_all = 0.3 *  (1 + np.sin(rad_times)) / 2
    elif ftype == 'piecewise':
        t = np.arange(0, 160, 1)
        res_all = eff_ub * np.ones(200)
        t_max = max(t)
        t_scaled = t / t_max
        # slope = eff_ub / 50
        # res = np.zeros(len(t))
        # # 0 - 30
        # res += (t < 30) * slope * (t + 1)
        # # 31 to 60
        # res += ((t >= 30) & (t < 60)) * (-0.00015 * t **2 + 0.0177 * t - 0.210)
        # # 61 to 90
        # res += ((60 <= t) & (t < 90)) * eff_ub
        # # 91 to 120
        # res += ((90 <= t) & (t < 120)) * (-0.00015 * t ** 2 + 0.0273 * t - 0.939)
        # # 121 to 150
        # res += ((120 <= t) & (t < 150)) * (eff_ub - slope * (t - 100 + 1))

        # use pdf of beta distribution
        a, b = 3, 3
        res = beta.pdf(t_scaled, a, b, loc=0, scale=1)
        max_val = np.max(res)
        res = res * eff_ub / max_val
        res_all[:80] = res[:80]
        res_all[-80:] = res[-80:]

    elif ftype == 'constant':
        t = np.arange(0, 200, 1)
        res_all = eff_ub * np.ones_like(t)

    return res_all


def dynamic_tracing(sim):

    tracing_array = get_dynamic_eff(chi_type, eff_ub)
    # get tracing intervention
    for cur_inter in sim['interventions']:
        if isinstance(cur_inter, cv.contact_tracing):
            break

    # update the trace_probs
    cur_t = sim.t
    sim_len = sim.npts
    # eff = np.zeros(sim_len)
    # start_day_trace = cur_inter.start_day
    # linear_stage_len, constant_stage_len = 30, 30
    # linear_stage_end = start_day_trace + linear_stage_len
    # constant_stage_end = linear_stage_end + constant_stage_len
    # eff[start_day_trace:linear_stage_end] = np.linspace(eff_lb, eff_ub, constant_stage_len)
    # eff[linear_stage_end:] = eff_ub
    eff = tracing_array.copy()
    trace_prob = dict(h=1.0, s=0.5, w=0.5, c=0.3)
    cur_scale = eff[cur_t]
    trace_prob = {key: val * cur_scale for key, val in trace_prob.items()}
    cur_inter.trace_probs = trace_prob



def test_data_generator():

    n_runs = 100
    # Define the testing and contact tracing interventions
    test_scale = 0.1
    # test_quarantine_scale = 0.1   min(test_scale * 4, 1)
    tp = cv.test_prob(symp_prob=test_scale, asymp_prob=0.001, symp_quar_prob=0.3,
                      asymp_quar_prob=0.3, quar_policy='daily')
    trace_prob = dict(h=1.0, s=0.5, w=0.5, c=0.3)

    trace_prob = {key: val*eff_ub for key,val in trace_prob.items()}
    ct = cv.contact_tracing(trace_probs=trace_prob)
    keep_d = True
    population = int(200e3)
    case_name = get_case_name(population, test_scale, eff_ub, keep_d, dynamic=True)
    case_name = '_'.join([case_name, chi_type])
    # Define the default parameters
    pars = dict(
        pop_type      = 'hybrid',
        pop_size      = population,
        pop_infected  = population / 500,
        start_day     = '2020-02-01',
        end_day       = '2020-08-01',
        interventions = [tp, ct, dynamic_tracing],
        analyzers=store_compartments(keep_d, label='get_compartments'),
        asymp_factor = 0.5
    )

    # consider new variant
    have_new_variant = False

    # Create, run, and plot the simulation
    fig_name = case_name
    sim = cv.Sim(pars)
    if have_new_variant:
        variant_day, n_imports, rel_beta, wild_imm, rel_death_prob = '2020-04-01', 200, 3, 0.5, 1
        sim = import_new_variants(sim, variant_day, n_imports, rel_beta, wild_imm, rel_death_prob=rel_death_prob)
    # sim.run()
    msim = cv.MultiSim(sim)
    msim.run(n_runs=n_runs, parallel=False, keep_people=True)
    msim.mean()
    msim.plot(to_plot=['new_infections_by_variant','new_infections', 'new_tests', 'new_diagnoses', 'cum_diagnoses', 'new_quarantined', 'test_yield'],
             do_show=False)
    plt.savefig('../Notebooks/figs/' + fig_name + '.png', dpi=300)
    plt.close()

    data_replicates = []
    for i in range(n_runs):
        get_data = msim.sims[i].get_analyzer('get_compartments')  # Retrieve by label

        compartments = 'STEAYDQRF' if get_data.keep_D else 'STEAYQRF'
        # get_data.plot(compartments)
        # res = None
        # for c in compartments:
        #     if res is None:
        #         res = np.array(get_data.__getattribute__(c))
        #     else:
        #         res += np.array(get_data.__getattribute__(c))
        # assert res.max() == sim['pop_size']
        data = pd.DataFrame()
        for c in compartments:
            data[c] = np.array(get_data.__getattribute__(c))
        data_replicates.append(data)
    df_final = reduce(lambda x, y: x + y, data_replicates)
    df_final /= n_runs





    # prepare the corresponding parameters of compartmental model
    population = sim['pop_size']
    params = {}
    tracing_array = get_dynamic_eff(chi_type, eff_ub)
    params['tracing_array'] = tracing_array
    params['population'] = population
    contacts = sim.pars['contacts']
    params['alpha'] = 1 / sim.pars['quar_period']
    params['beta'] = sum([ct.trace_probs[key] * val for key, val in contacts.items()]) / sum(contacts.values())
    params['gamma'] = 1 / sim.pars['dur']['exp2inf']['par1']
    params['mu'] = tp.symp_prob
    params['tau'] = tp.symp_quar_prob # tau can not be directly determined
    params['tau_lb'] = 0 # tp.asymp_quar_prob
    params['tau_ub'] = tp.symp_quar_prob
    params['lamda'] = 1 / 10.0
    params['p_asymp'] = 1 - msim.sims[0].people.symp_prob.mean()
    params['n_contacts'] = sum(contacts.values())
    severe_probs =  msim.sims[0].people.severe_prob.mean()
    crit_probs =  msim.sims[0].people.crit_prob.mean()
    params['delta'] = severe_probs * crit_probs * 1 / sim.pars['dur']['crit2die']['par1']
    params['data'] = data_replicates.copy() #df_final
    params['dynamic_tracing'] = True
    params['eff_ub'] = eff_ub
    file_name = 'covasim_'+ case_name + '_' + str(n_runs)
    file_name += '.joblib'
    file_path = '../Data/covasim_data'
    joblib.dump(params, os.path.join(file_path, file_name), compress=True)