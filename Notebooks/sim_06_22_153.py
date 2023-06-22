# -*- coding: utf-8 -*-
"""sim_06/22_153.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VsFhX_fso-agTsdmVBuDk90Fb2olrLN1
"""

!python3 -m pip install covasim
!python3 -m pip install sciris

import covasim as cv
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import pylab as pl
import sciris as sc
import numpy as np

class store_steaydqrf(cv.Analyzer):

    # initialize lists of timestops and number of agents in each state
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = []
        self.S = []
        self.T = []
        self.E = []
        self.A = []
        self.Y = []
        self.D = []
        self.Q = []
        self.R = []
        self.F = []
        return

    # fill in list of agents in each state
    def apply(self, sim):
      ppl = sim.people
      self.t.append(sim.t)
      self.S.append(ppl.susceptible.sum())
      self.T.append((ppl.quarantined & ppl.susceptible).sum())
      self.E.append((ppl.exposed & ~ppl.infectious).sum())
      self.A.append((ppl.infectious & ~ppl.symptomatic).sum())
      self.Y.append(ppl.symptomatic.sum())
      self.D.append(ppl.diagnosed.sum())
      self.Q.append(ppl.quarantined.sum())
      self.R.append(ppl.recovered.sum())
      self.F.append(ppl.dead.sum())
      return

    def save_lists(self, filename):
        data = {
            't': self.t,
            'S': self.S,
            'T': self.T,
            'E': self.E,
            'A': self.A,
            'Y': self.Y,
            'D': self.D,
            'Q': self.Q,
            'R': self.R,
            'F': self.F
        }
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

    # plot STEADYQRF
    def plot(self):
        pl.figure(figsize=(12,6))
        pl.plot(self.t, self.S, label='S')
        pl.plot(self.t, self.T, label='T')
        pl.plot(self.t, self.E, label='E')
        pl.plot(self.t, self.A, label='A')
        pl.plot(self.t, self.Y, label='Y')
        pl.plot(self.t, self.D, label='D')
        pl.plot(self.t, self.Q, label='Q')
        pl.plot(self.t, self.R, label='R')
        pl.plot(self.t, self.F, label='F')
        pl.legend()
        pl.xlabel('Day')
        pl.ylabel('People')
        sc.setylim()
        sc.commaticks()
        return

pars = dict(pop_size = 200e3,
            beta     = 0.03)
cb   = cv.change_beta(days=[0,10,20,30,40,50],changes=[1.5,1.25,1,0.75,0.5,1],layers=None)
tn   = cv.test_num(daily_tests=500,sensitivity=0.89,test_delay=1,loss_prob=0.2)
vc   = cv.vaccinate_num(vaccine='pfizer',num_doses=200,booster=False,sequence='age')
tp   = cv.test_prob(symp_prob=0.1)
ct   = cv.contact_tracing(trace_probs=0.3)

sim1 = cv.Sim(interventions=[ct,tp],pars=pars,analyzers=[store_steaydqrf(label='steaydqrf')])
sim1.run()
steaydqrf = sim1.get_analyzer('steaydqrf')
steaydqrf.plot()

steaydqrf.save_lists('data.pickle')
with open('data.pickle', 'rb') as file:
    data = pickle.load(file)
df = pd.DataFrame(data)
df.set_index('t',inplace=True)
df.to_csv('/content/table.csv', index=True)
from google.colab import files
files.download('/content/table.csv')