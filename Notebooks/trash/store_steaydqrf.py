# -*- coding: utf-8 -*-
"""store_steaydqrf.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bwebX9V7aTMyjY_0nKRcWrkVuaURoq88
"""

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

analyzer = store_steaydqrf()

analyzer.save_lists('data.pickle')

with open('data.pickle', 'rb') as file:
    data = pickle.load(file)

df = pd.DataFrame(data)

df.set_index('t',inplace=True)

df.to_csv('/content/table.csv', index=True)

from google.colab import files
files.download('/content/table.csv')