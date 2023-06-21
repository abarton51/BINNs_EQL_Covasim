
import sys

import joblib

sys.path.append('../')

from Modules.Utils.Imports import *
from Modules.Utils.ModelWrapper import ModelWrapper
from Modules.Models.BuildBINNs import BINNCovasim

import Modules.Loaders.DataFormatter as DF
import datetime

from utils import plot_loss_convergence, get_case_name
import matplotlib
matplotlib.use('Agg')

device = torch.device(GetLowestGPU(pick_from=[0,1,2,3]))
# torch.manual_seed(9099058152467048838)
# np.random.seed(1232914967)

path = '../Data/covasim_data/'
population = 200000
test_prob = 0.1
trace_prob = 0.3
keep_d = True
retrain = False
dynamic=True
chi_type = 'piecewise'
case_name = get_case_name(population, test_prob, trace_prob, keep_d, dynamic=dynamic, chi_type=chi_type)
# yita_lb, yita_ub = 0.2, 0.4

params = DF.load_covasim_data(path, population, test_prob, trace_prob, case_name,plot=True)

def to_torch(ndarray):
    arr = torch.tensor(ndarray, dtype=torch.float)
    arr.requires_grad_(True)
    arr = arr.to(device)
    return arr

def to_numpy(x):
    return x.detach().cpu().numpy()

# split into train/val and convert to torch
data = params['data']
data = (data / params['population']).to_numpy()
params.pop('data')
N = len(data)
split = int(0.8*N)
p = np.random.permutation(N)
x_train = to_torch(p[:split][:, None]/(N-1))
y_train = to_torch(data[p[:split]])
x_val = to_torch(p[split:][:, None]/(N-1))
y_val = to_torch(data[p[split:]])

# generate save path
mydir = os.path.join('../models/covasim', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(mydir)

tracing_array = params['tracing_array']
# initialize model
binn = BINNCovasim(params, N - 1, tracing_array, keep_d=keep_d, chi_type=chi_type)
binn.to(device)

# compile
parameters = binn.parameters()
opt = torch.optim.Adam(parameters, lr=1e-3)
os.makedirs(os.path.join(mydir, case_name))
model = ModelWrapper(
    model=binn,
    optimizer=opt,
    loss=binn.loss,
    augmentation=None,
    # scheduler= scheduler,
    save_name=os.path.join(mydir, case_name) )
model.str_name = 'STEAYDQRF'

# save the range information before training
ranges = [binn.yita_lb, binn.yita_ub, binn.beta_lb, binn.beta_ub, binn.tau_lb, binn.tau_ub]
file_name = '_'.join([str(m) for m in ranges])
joblib.dump(None, os.path.join(mydir, file_name)) # model.save_folder
# if retrain
if retrain:
    model.load(model.save_name + '_best_val_model', device=device)
    model.model.train()
    model.save_name += '_retrain'
epochs = int(10e4)
batch_size = 128
rel_save_thresh = 0.05

# train jointly
model.fit(
    x=x_train,
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=None,
    verbose=1,
    validation_data=[x_val, y_val],
    early_stopping=40000,
    rel_save_thresh=rel_save_thresh)


# # fitting performance on training data
# y_train_pred = to_numpy(model.predict(x_train))


# load training errors
total_train_losses = model.train_loss_list
total_val_losses = model.val_loss_list

plot_loss_convergence(total_train_losses, total_val_losses, rel_save_thresh, model.save_folder)
