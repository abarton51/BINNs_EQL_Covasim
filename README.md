<h1 align="center">Incorporating Adaptive Human Behavior into Epidemiological Models Using Equation Learning</h1>

<!---
-->
![](https://github.com/abarton51/BINNs_EQL_Covasim/blob/main/Figures/drums_binn_schematic3_github.png?raw=true)

<h2 align="center">DRUMS REU GitHub Repository</h2>
<h3 align="center">Austin Barton, Jonathan Greer, Jordan Klein, Patrick Haughey; Mentors: Kevin Flores</h3>
***
All equation learning and BINNs code has been adapted from [[1.]](https://arxiv.org/abs/2005.13073) and [Xin Li](xli86@ncsu.edu).

All code under the folder titled "covasim" is the work of the ***["Institute for Disease Modeling"](https://github.com/InstituteforDiseaseModeling/covasim)***. We use covasim version 3.1.3 for our implementations.
***
## Abstract
&emsp; Mathematical models have been shown to be valuable tools for forecasting and evaluating public health interventions during epidemics such as COVID-19. Covasim is an open-source agent-based model (ABM) that was recently developed to simulate the transmission of COVID-19. Covasim has been validated with real-world data and utilized for simulating the potential effect of public health interventions.  Covasim’s base model does not implement adaptive behaviors; however, by leveraging Covasim’s flexibility, we can utilize its resources to generate data for scenarios where human behavior can adapt based on the current state of the model. Human behaviors, such as compliance to masking guidelines, have been shown to be variable between individuals and variable in time, i.e., individuals adapt their masking behavior in response to their perceived risk of infection as infection rates fluctuate. We extended the Covasim model to incorporate adaptive masking behavior to investigate its effect on Covasim’s predicted forecast. We ran the data generated from this model through Biologically-Informed Neural Networks (BINNs) to learn an ordinary differential equation (ODE) approximation of this model. We performed sensitivity analysis on the learned ODE to determine the parameters with the largest influence on how masking behavior affected infection prevalence. The extended Covasim and BINNs computational pipeline we developed will be open-sourced to provide a quantitative framework for incorporating adaptive human behavior into forecasting future epidemics.
***
## Introduction
&emsp; Covasim is an agent-based simulator created by the Institute for Disease Modeling in response to the COVID-19 pandemic and the apparent need for resources on simulating, studying, modeling, and predicting disease spread. The creators of Covasim created custom features for users to interact with and change the dynamics of the simulator. In the paper, "Covasim: An Agent-Based Model of COVID-19 Dynamics and Interventions" [[2.]](#references), the simulations provided as examples use a variety of features to model agent interactions and probabilities of infection, tracing, quarantining, etc. However, adaptive behaviors of the agents have not been thoroughly studied or modeled. In order to accurately model the spread of disease in a human population, we must take into account the adaptive and maladaptive behaviors of agents in response to the current state of the disease as well as other attributes of their own setting.

&emsp; We introduce a masking adaptive behavior in order to more accurately model the simulation of COVID-19 spread. We then use an existing compartmental model consisting of 9 different states to mathematically model the spread of disease in the simulation and utilize equation learning methods in order to infer possibly nonlinear parameter components of the system of equations. We hypothesized that introducing such adaptive behavior(s) increases the nonlinearity and complexity of the learned equations and therefore, using sophisticated techniques in order to accurately learn components such as contact rate, tracing rate, and quarantined rate, is vital to the accuracy and interpretability of the learned system of equations. We used Biologically-Informed Neural Networks (BINNs) in order to do this. For data generated with masking as an adaptive behavior, we included the average proportion of population masking as an input into the contact rate parameter. Additionally, we experimented with a modified BINN that bypasses the need for learning the solutions to the equations by leveraging multiple simulations to obtain sufficiently accurate estimates of the solutions in order to focus on learning the parameter networks and decrease the complexity of the multi-objective optimization problem at hand.

&emsp; Biologically-Informed Neural Networks attempt to tackle the *model specification* problem by using multilayer perceptrons as surrogate models to approximate unknown and possibly nonlinear parameters in order to minmize *a priori* assumptions about the form of the differential equation(s) as well as expand the number of possible solutions to such a model. Physics-Informed Neural Networks, first introduced in [2018](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125), primarily tackle the issue of inferring a surface of solutions to a system of differential equations. This inherently requires prior knowledge and assumptions of the governing dynamics. One of the most apparent assumptions, which is directly addressed by [[1.]](#references) and mentioned in [[3.]](#references), is the form of each of the parameters in the governing system of equations. Generally, these parameters may be linear, nonlinear, or constant functions. If they are nonlinear, the space of possible functions is too large to accurately solve for without making large assumptions using methods of sparse regression. Hence, using a universal function approximator (such as an MLP) allows us to learn these nonlinear components and then infer underlying relationships and functions *a posteriori*.

***
## Directory
### covasim
Folder that contains all code for Covasim version 3.1.3 created by the ["Institute for Disease Modeling"](https://github.com/InstituteforDiseaseModeling/covasim). This is the simulator our code interacts with to generate data.

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

***
## References

[1.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008462) Lagergren, J. H., Nardini, J. T., Baker, R. E., Simpson, M. J., & Flores, K. B. (2020). Biologically-informed neural networks guide mechanistic modeling from sparse experimental data. PLOS Computational Biology, 16(12). https://doi.org/10.1371/journal.pcbi.1008462 

[2.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009149) Kerr, C. C., Stuart, R. M., Mistry, D., Abeysuriya, R. G., Rosenfeld, K., Hart, G. R., Núñez, R. C., Cohen, J. A., Selvaraj, P., Hagedorn, B., George, L., Jastrzębski, M., Izzo, A., Fowler, G., Palmer, A., Delport, D., Scott, N., Kelly, S., Bennette, C. S., … Klein, D. J. (2020). Covasim: An Agent-Based Model of COVID-19 Dynamics and Interventions. https://doi.org/10.1101/2020.05.10.20097469

[3.](https://www.sciencedirect.com/science/article/pii/S0021999118307125) Raissi, M., Perdikaris, P., &amp; Karniadakis, G. E. (2018, November 3). Physics-informed Neural Networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics. https://www.sciencedirect.com/science/article/pii/S0021999118307125
