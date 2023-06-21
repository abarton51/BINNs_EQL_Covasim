
## python environment install with conda
```
# create new environment from file
conda create -n "myenv" --file requirements.txt python=3.9.0
# install sciris
pip install sciris
# install pytorch
conda install pytorch==1.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

## Information of each folder:
```
/covasim contains the source code for the agent-based simulation model Covasim
/Notebooks contains the source code for BINN
/tests contains the source code for data generation
```

## Sequence of execution for BINN
```
BINNCovasimTraining.py for model training

# after model training, remember to replace my_dir with the path for the trained model

BINNCovasimParameterNN_dynamic.py for learning symbolic functions for each parameter neural network

BINNCovasimEvaluation_dynamic.py for evaluation and comparison for ODE models
```
