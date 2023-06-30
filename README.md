# Equation Learning for COVASIM with added Adaptive Behaviors using Biologically-Informed Neural Networks

All equation learning and BINNs code has been adapted from [[1.]](https://arxiv.org/abs/2005.13073) and [Xin Li](xli86@ncsu.edu).

All code under the folder titled "covasim" is the work of the ***["Institute for Disease Modeling"](https://github.com/InstituteforDiseaseModeling/covasim)***.

### Introduction
Biologically-Informed Neural Networks attempt to tackle the *model specification* problem by using multilayer perceptrons as nonlinear functional parameters as surrogate models in order to minmize *a priori* assumptions about the form of the differential equation(s) as well as expand the number of possible solutions to such a model. Physics-Informed Neural Networks, first introduced in [2018](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125), primarily tackle the issue of inferring a surface of solutions to a system of differential equations. This inherently requires prior knowledge and assumptions of the governing dynamics. One of the most apparent assumptions, which is mentioned in [[3.]](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125), is the form of each of the parameters in the governing system of equations. Generally, these parameters may be linear, nonlinear, or constant functions. If they are nonlinear, the space of possible functions is too large to accurately solve for using methods of sparse regression. Hence, using a universal function approximator (such as an MLP) allows us to learn these nonlinear components and then infer underlying relationships and functions *a posteriori*.

#### References

1. Lagergren, J. H., Nardini, J. T., Baker, R. E., Simpson, M. J., & Flores, K. B. (2020). Biologically-informed neural networks guide mechanistic modeling from sparse experimental data. PLOS Computational Biology, 16(12). https://doi.org/10.1371/journal.pcbi.1008462 

2. Kerr, C. C., Stuart, R. M., Mistry, D., Abeysuriya, R. G., Rosenfeld, K., Hart, G. R., Núñez, R. C., Cohen, J. A., Selvaraj, P., Hagedorn, B., George, L., Jastrzębski, M., Izzo, A., Fowler, G., Palmer, A., Delport, D., Scott, N., Kelly, S., Bennette, C. S., … Klein, D. J. (2020). Covasim: An Agent-Based Model of COVID-19 Dynamics and Interventions. https://doi.org/10.1101/2020.05.10.20097469
