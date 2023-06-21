'''
Simple Covasim usage
'''

import covasim.covasim as cv


def test_covasim():
    sim = cv.Sim(pars={})  # Configure the simulation
    sim.run()  # Steps required to produce the behavior

    sim.plot(to_plot='seir')
