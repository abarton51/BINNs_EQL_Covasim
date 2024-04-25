<h1 align="center">Incorporating Adaptive Human Behavior into Epidemiological Models using Equation Learning</h1>

![](https://github.com/abarton51/BINNs_EQL_Covasim/blob/main/Figures/github/binnschematic_github.png?raw=true)

<h2 align="center">Code Repository for Data Generation, Equation Learning, BINNs, Visualization, and Analysis</h2>

Code repository for undergraduate research project titled:

> Incorporating Adaptive Human Behavior into Epidemiological Models using Equation Learning
>
> [Austin T. Barton](github.com/abarton51), Jordan Klein, [Kevin B. Flores*](kbflores@ncsu.edu)
>
>  \* Corresponding author.
>
> Abstract: Mathematical and computational models have both proven to be valuable tools for forecasting and studying epidemics. Covasim is an open-source agent-based model (ABM) developed to simulate the transmission of COVID-19. Human behaviors, such as masking, have been shown to depend on the state of the epidemic and can significantly impact disease spread. While Covasim's base model does not include adaptive elements, we extended it to incorporate adaptive masking behavior to investigate its effect on the system's behavior, complexity, dynamics of certain parameters, and one's ability to obtain an ordinary differential equation (ODE) approximation from it. Using an existing compartmental model, we processed the data generated from this extended ABM through Biologically-Informed Neural Networks (BINNs) and sparse regression methods to obtain an ODE approximation. The extended ABM and equation learning pipeline we developed is open-source to provide a quantitative framework for incorporating adaptive behaviors into other similar computational models.

All code under the folder titled "covasim" is the work of the ***["Institute for Disease Modeling"](https://github.com/InstituteforDiseaseModeling/covasim)***. We use covasim version 3.1.3 for our implementations.
Equation learning and BINNs code has been adapted from [[1.]](https://arxiv.org/abs/2005.13073) and [Xin Li](xli86@ncsu.edu).

<!--
***
## Introduction
&emsp; Covasim is an agent-based simulator created by the Institute for Disease Modeling in response to the COVID-19 pandemic and the apparent need for resources on simulating, studying, modeling, and predicting disease spread. The creators of Covasim created custom features for users to interact with and change the dynamics of the simulator. In the paper, "Covasim: An Agent-Based Model of COVID-19 Dynamics and Interventions" [[2.]](#references), the simulations provided as examples use a variety of features to model agent interactions and probabilities of infection, tracing, quarantining, etc. However, adaptive behaviors of the agents have not been thoroughly studied or modeled. In order to accurately model the spread of disease in a human population, we must take into account the adaptive and maladaptive behaviors of agents in response to the current state of the disease as well as other attributes of their own setting.

&emsp; We introduce a masking adaptive behavior in order to more accurately model the simulation of COVID-19 spread. We then use an existing compartmental model consisting of 9 different states to mathematically model the spread of disease in the simulation and utilize equation learning methods in order to infer possibly nonlinear parameter components of the system of equations. We hypothesized that introducing such adaptive behavior(s) increases the nonlinearity and complexity of the learned equations and therefore, using sophisticated techniques in order to accurately learn components such as contact rate, tracing rate, and quarantined rate, is vital to the accuracy and interpretability of the learned system of equations. We used Biologically-Informed Neural Networks (BINNs) in order to do this. For data generated with masking as an adaptive behavior, we include the proportion of population masking as an input into the contact rate parameter. Additionally, we experimented with a modified BINN that bypasses the need for learning the solutions to the equations by leveraging multiple simulations to obtain sufficiently accurate estimates of the solutions in order to focus on learning the parameter networks and decrease the complexity of the multi-objective optimization problem at hand.

&emsp; Biologically-Informed Neural Networks attempt to tackle the *model specification* problem by using neural networks as surrogate models to approximate unknown and possibly nonlinear parameters in order to minimize *a priori* assumptions about the form of the differential equation(s) as well as expand the number of possible solutions to such a model. The architecture is split into two key parts. The first is the Biologically-Informed Neural Network that approximates the solutions to the governing dynamical system. The second part consists of the parameter networks that take as inputs the approximated solutions from the Biologically-Informed part. These parameter networks approximate parameters that provide solutions of our governing dynamical system close to the approximated solutions. Physics-Informed Neural Networks, first introduced in [2018](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125), primarily tackle the issue of inferring a surface of solutions to a system of differential equations. This inherently requires prior knowledge and assumptions of the governing dynamics. One of the most apparent assumptions, which is directly addressed by [[1.]](#references) and mentioned in [[3.]](#references), is the form of each of the parameters in the governing system of equations. Generally, these parameters may be linear, nonlinear, or constant functions. If they are nonlinear, the space of possible functions is too large to accurately solve for without making large assumptions using methods of sparse regression. Hence, using a universal function approximator (such as an MLP) allows us to learn these nonlinear components and then infer underlying relationships and functions *a posteriori*.
-->
***
## Directory
### Data
Folder that contains all the data and compartmental plots from Covasim generated data.

### Figures
Folder that contains important figures, schematics, images, etc. for the REU project.

### models
Folder that contains all the trained models, their outputted compartments, loss, evaluation, and learned parameter curves.
-  **debugging** - Folder that contains models for the purpose of debugging and exploring results.
-  **denoised** - Folder that contains trained BINNs models that takes the average compartments over multiple simulations and numerically approximated time derivatives as inputs.
- **mask** - Folder that contained trained full BINNs models with average masking as an input into the contact rate function.
### Modules
Folder that acts as a library containing all the models, model wrappers, helper functions, data storage and loading, etc.
- **Activations** - Folder that contains custom made activation functions.
  - SoftPlusReLU.py:
File for the custom made SoftPlusReLU activation function.
- **Loaders** - Folder that contains data loaders and formatters.
  - DataFormatter.py:
File for loading, formatting, and interacting with saved data.
- **Models** - Folder containing the neural network models created using PyTorch.
  - BuildBINNs.py:
File that contains the code for the BINNs models. There are 3 separate models. One being the model created by Xin Li and the others being the models adapted by Austin Barton from Xin Li that bypasses the surface fitting portion of BINNs to focus on learning the parameters.
  - BuildMLP.py: 
File that interacts with PyTorch to create a basic multi-layer perceptron (MLP).
- **Utils** - Folder containing numerous utility files.
  - GetLowestGPU.py
  - Gradient.py
  - Imports.py
  - ModelWrapper.py
  - PDESolver.py
  - TimeRemaining.py
### Notebooks
Folder that contains all the code for training, evaluating, and plotting learned parameter curves.
- **data_generation** - Folder that contains files related to interacting with Covasim, generating data, and storing it.
  - drums_data_gen_multi.py:
File that generates data from multiple simulations.
  - DRUMS_data_gen.ipynb:
Notebook that interacts with data generator files.
- **figs** - Folder that contains figures and plots relating to the data such as Covasim generated plots and numerically approximated time derivatives.
  - **drums** - DRUMS REU figures.
  - **xin_figs** - Xin Li's figures.
### covasim
Folder that contains all code for Covasim version 3.1.3 created by the ["Institute for Disease Modeling"](https://github.com/InstituteforDiseaseModeling/covasim). This is the simulator our code interacts with to generate data.

***
## Contributions
**Contributors**: Austin T. Barton, Jordan Klein, Kevin B. Flores, Patrick M. Haughey, Jonathan Greer, Xin Li.

***
## References

[1.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008462) Lagergren, J. H., Nardini, J. T., Baker, R. E., Simpson, M. J., & Flores, K. B. (2020). Biologically-informed neural networks guide mechanistic modeling from sparse experimental data. PLOS Computational Biology, 16(12).

[2.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009149) Kerr, C. C., Stuart, R. M., Mistry, D., Abeysuriya, R. G., Rosenfeld, K., Hart, G. R., Núñez, R. C., Cohen, J. A., Selvaraj, P., Hagedorn, B., George, L., Jastrzębski, M., Izzo, A., Fowler, G., Palmer, A., Delport, D., Scott, N., Kelly, S., Bennette, C. S., … Klein, D. J. (2020). Covasim: An Agent-Based Model of COVID-19 Dynamics and Interventions.

[3.](https://www.sciencedirect.com/science/article/pii/S0021999118307125) Raissi, M., Perdikaris, P., &amp; Karniadakis, G. E. (2018, November 3). Physics-informed Neural Networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics.

[4.](https://royalsocietypublishing.org/doi/10.1098/rsif.2020.0987) Nardini, J. T., Baker, R. E., Simpson, M. J., & Flores, K. B. (2021, March 17). Learning differential equation models from stochastic agent-based model ... The Royal Society Publishing.
