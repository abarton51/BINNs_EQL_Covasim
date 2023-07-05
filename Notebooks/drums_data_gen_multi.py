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

class ModelParams():
    
    def __init__(self, 
                 population=int(200e3), 
                 test_prob=0.1, 
                 trace_lb=0, 
                 trace_ub=0.3, 
                 chi_type='constant', 
                 keep_d=True, 
                 dynamic=True,
                 maskb=False):
        
        global chi_type_global
        global eff_ub_global
        chi_type_global = chi_type
        eff_ub_global = trace_ub
        
        self.population = population
        self.test_prob = test_prob
        self.trace_lb = trace_lb
        self.keep_d = keep_d
        self.dynamic = dynamic
        self.maskb = maskb
        return

class masking(cv.Intervention):
  def __init__(self, model_params=None, thresh_scale=None, rel_sus=None, maskprob_ub=None,maskprob_lb=None,*args, **kwargs):
    super().__init__(**kwargs)
    self.population    = model_params.population
    self.thresh_scale  = thresh_scale
    self.rel_sus       = rel_sus
    self.maskprob_lb   = maskprob_lb
    self.maskprob_ub   = maskprob_ub
    return

  def initialize(self, sim):
    super().initialize()
    self.population      = int(len(sim.people))
    self.thresh          = self.population * self.thresh_scale
    self.su_orig_rel_sus = np.float32(1.47)
    self.s_orig_rel_sus  = np.float32(1.24)
    self.a_orig_rel_sus  = np.float32(1.00)
    self.ad_orig_rel_sus = np.float32(0.67)
    self.c_orig_rel_sus  = np.float32(0.34)
    self.child           = sim.people.age < 9
    self.adolescent      = np.logical_and(sim.people.age > 0, sim.people.age <= 19)
    self.adult           = np.logical_and(sim.people.age > 19, sim.people.age <= 69)
    self.senior          = np.logical_and(sim.people.age > 69, sim.people.age <= 79)
    self.supsenior       = sim.people.age > 79
    self.tvec            = sim.tvec

  def apply(self, sim):
    self.num_dead      = sim.people.dead.sum()
    self.num_diagnosed = sim.people.diagnosed.sum()
    self.p             = np.exp((0.001+((self.num_diagnosed+self.num_dead))/(self.population/10)-0.005*(sim.t)))
    self.p             = (self.p/(1+self.p))-0.35
    self.p             = np.clip(self.p,self.maskprob_lb,self.maskprob_ub)
    self.immunocomp    = np.random.choice([True,False],size=len(sim.people),p=[0.03,0.97])
    sim.people.rel_sus[self.child]      = self.c_orig_rel_sus
    sim.people.rel_sus[self.adolescent] = self.ad_orig_rel_sus
    sim.people.rel_sus[self.adult]      = self.a_orig_rel_sus
    sim.people.rel_sus[self.senior]     = self.s_orig_rel_sus
    sim.people.rel_sus[self.supsenior]  = self.su_orig_rel_sus
    if self.num_dead + self.num_diagnosed > self.thresh:
      self.masking       = np.random.choice([True,False],size=len(sim.people),p=[self.p,(1-self.p)])
      sim.people.rel_sus = np.where(self.masking & self.supsenior,(self.rel_sus * self.su_orig_rel_sus).astype(sim.people.rel_sus.dtype),self.su_orig_rel_sus)
      sim.people.rel_sus = np.where(self.masking & self.senior,(self.rel_sus * self.s_orig_rel_sus).astype(sim.people.rel_sus.dtype),self.s_orig_rel_sus)
      sim.people.rel_sus = np.where(self.masking & self.adult,(self.rel_sus * self.a_orig_rel_sus).astype(sim.people.rel_sus.dtype),self.a_orig_rel_sus)
      sim.people.rel_sus = np.where(self.masking & self.adolescent,(self.rel_sus *self.ad_orig_rel_sus).astype(sim.people.rel_sus.dtype),self.ad_orig_rel_sus)
      sim.people.rel_sus = np.where(self.masking & self.child,(self.rel_sus*self.c_orig_rel_sus).astype(sim.people.rel_sus.dtype),self.c_orig_rel_sus )
      sim.people.rel_sus = np.where(self.immunocomp,(self.rel_sus * sim.people.rel_sus).astype(sim.people.rel_sus.dtype),sim.people.rel_sus)
    else:
        None
    return

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
        # plt.show()
        pl.savefig('../Notebooks/figs/' + 'compartments' + '.png')
        return

def get_dynamic_eff(ftype, eff_ub):
    '''
    Dyanmic probability of tracing function.
    Function that interacts with the tracing rate term of the STEAYDQRF model.
    Can be of type linear, piecewise, sin, or constant.
    
    Args:
        ftype (str): the type of function for h(t) to be.
        eff_ub (float): the effective upper bound on the values of h(t).
    
    Returns:
        res_all (float array): Vector of values evaluated at numerous time points.
    '''

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

    tracing_array = get_dynamic_eff(chi_type_global, eff_ub_global)
    # get tracing intervention
    for cur_inter in sim['interventions']:
        if isinstance(cur_inter, cv.contact_tracing):
            break

    # update the trace_probs
    cur_t = sim.t
    sim_len = sim.npts
    eff = tracing_array.copy()
    trace_prob = dict(h=1.0, s=0.5, w=0.5, c=0.3)
    cur_scale = eff[cur_t]
    trace_prob = {key: val * cur_scale for key, val in trace_prob.items()}
    cur_inter.trace_probs = trace_prob



def drums_data_generator_multi(model_params=None, num_runs=100):
    '''
    Data generation function that takes in the model parameters for the COVASIM simulation
    and interacts with the covaism module in order to simulate, save, and store data.
    
    Args:
        model_params (Object): ModelParams object that stores covasim model parameters.
        n_runs (int): number of simulations to complete to computer sample means of results.
    
    Returns:
        None
    '''
    # if no model_params is specified then instantiate ModelParams with default parameter values
    if model_params==None:
        model_params = ModelParams()
    
    n_runs = num_runs
    population = model_params.population
    keep_d = model_params.keep_d
    dynamic = model_params.dynamic
    maskb = model_params.maskb
    
    # Define the testing and contact tracing interventions
    test_scale = model_params.test_prob
    # test_quarantine_scale = 0.1   min(test_scale * 4, 1)
    tp = cv.test_prob(symp_prob=test_scale, asymp_prob=0.001, symp_quar_prob=0.3,
                      asymp_quar_prob=0.3, quar_policy='daily')
    trace_prob = dict(h=1.0, s=0.5, w=0.5, c=0.3)

    mk = masking(model_params, thresh_scale=0.1,rel_sus=1.0, maskprob_lb=0.0, maskprob_ub=0.7)

    trace_prob = {key: val*eff_ub_global for key,val in trace_prob.items()}
    ct = cv.contact_tracing(trace_probs=trace_prob)
    
    case_name = get_case_name(population, test_scale, eff_ub_global, keep_d, dynamic=True)
    case_name = '_'.join([case_name, chi_type_global])
    if maskb:
        # Define the default parameters if there is masking intervention
        pars = dict(
            pop_type      = 'hybrid',
            pop_size      = population,
            pop_infected  = population / 500,
            start_day     = '2020-02-01',
            end_day       = '2020-08-01',
            interventions = [tp, ct, dynamic_tracing, mk],
            analyzers=store_compartments(keep_d, label='get_compartments'),
            asymp_factor = 0.5
        )
    else:
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
    if maskb:
        fig_name = fig_name + '_masking' + '_' + str(n_runs)
    else:
        fig_name = fig_name + '_' + str(n_runs)
        
    sim = cv.Sim(pars)
    if have_new_variant:
        variant_day, n_imports, rel_beta, wild_imm, rel_death_prob = '2020-04-01', 200, 3, 0.5, 1
        sim = import_new_variants(sim, variant_day, n_imports, rel_beta, wild_imm, rel_death_prob=rel_death_prob)
        
    msim = cv.MultiSim(sim)
    msim.run(n_runs=n_runs, parallel=False, keep_people=True)
    msim.mean()
    msim.plot(to_plot=['new_infections_by_variant','new_infections', 'new_tests', 'new_diagnoses', 'cum_diagnoses', 'new_quarantined', 'test_yield'],
             do_show=False)
    plt.savefig('../Notebooks/figs/drums/' + fig_name + '.png', dpi=300)
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
    tracing_array = get_dynamic_eff(chi_type_global, eff_ub_global)
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
    params['eff_ub'] = eff_ub_global
    file_name = 'covasim_'+ fig_name
    file_name += '.joblib'
    file_path = '../Data/covasim_data/drums_data'
    joblib.dump(params, os.path.join(file_path, file_name), compress=True)