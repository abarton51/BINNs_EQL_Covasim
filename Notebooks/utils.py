import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os

from sklearn.linear_model import LassoCV, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import lasso_path
from itertools import cycle
import covasim as cv

def import_new_variants(sim, start_day, n_imports, rel_beta, wild_imm, rel_death_prob=1, rescale=True):

    my_var = cv.variant(variant={'rel_beta': rel_beta, 'rel_death_prob': rel_death_prob},
                        label='New variant', days=start_day, n_imports=n_imports, rescale=rescale) # , 'immunity': {'wild': wild_imm}}
    sim.update_pars(variants=sim.pars['variants'] + [my_var])
    # sim.initialize(reset=True)
    # sim = assign_app_users(sim, pct_dct, trace_type)
    return sim

def AIC_OLS(pred, true, kappa):
    residuals = pred - true
    # residuals *= np.abs(pred).clip(1,np.inf)**(-0.2)
    N = pred.size
    return N * np.log(np.mean(residuals**2)) + 2 * kappa

def RSS(pred, true):
    residuals = pred - true
    return np.sum(residuals**2, axis=0)

def get_case_name(population, test_prob, trace_prob, keep_d, dynamic=False, chi_type=None):
    file_name = '_'.join([str(population), str(test_prob), str(trace_prob)])
    if not keep_d:
        file_name += '_' + 'noD'
    if dynamic:
        file_name += '_' + 'dynamic'
    if chi_type:
        file_name += '_' + chi_type
    return file_name

def plot_loss_convergence(total_train_losses, total_val_losses, rel_save_thresh, file_path):
    # find where errors decreased
    train_idx, train_loss, val_idx, val_loss = [], [], [], []
    best_train, best_val = 1e12, 1e12
    for i in range(len(total_train_losses)-1):
        rel_diff = (best_train - total_train_losses[i])
        rel_diff /= best_train
        if rel_diff > rel_save_thresh:
            best_train = total_train_losses[i]
            train_idx.append(i)
            train_loss.append(best_train)
        rel_diff = (best_val - total_val_losses[i])
        rel_diff /= best_val
        if rel_diff > rel_save_thresh:
            best_val = total_val_losses[i]
            val_idx.append(i)
            val_loss.append(best_val)
    idx = np.argmin(val_loss)

    # plot
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1, 2, 1)
    plt.semilogy(total_train_losses, 'b')
    plt.semilogy(total_val_losses, 'r')
    plt.semilogy(val_idx[idx], val_loss[idx], 'ko')
    plt.legend(['train mse', 'val mse', 'best val'])
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.title('Train/Val errors')
    plt.grid()
    ax = fig.add_subplot(1, 2, 2)
    plt.semilogy(train_idx, train_loss, 'b.-')
    plt.semilogy(val_idx, val_loss, 'r.-')
    plt.legend(['train mse', 'val mse'])
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.title('Train/Val improvements')
    plt.grid()
    plt.savefig(os.path.join(file_path, 'train_loss.png'))
    # plt.show()

def lasso_parameter_fitting(data_x, data_y, parameter_name, save_path, case_name, fit_intercept, term_names):
    N = len(data_x)
    split = int(0.8 * N)
    p = np.random.permutation(N)
    x_train = data_x[p[:split]]
    y_train = data_y[p[:split]]
    x_val = data_x[p[split:]]
    y_val = data_y[p[split:]]

    # lasso CV
    start_time = time.time()
    model = make_pipeline(LassoCV(alphas=np.logspace(-6, -2), cv=5, max_iter=10000, fit_intercept=fit_intercept)).fit(
        data_x, data_y)
    fit_time = time.time() - start_time
    ymin, ymax = 2300, 3800
    lasso = model[-1]
    plt.figure()
    plt.semilogx(lasso.alphas_, lasso.mse_path_, linestyle=":")
    plt.plot(
        lasso.alphas_,
        lasso.mse_path_.mean(axis=-1),
        color="black",
        label="Average across the folds",
        linewidth=2,
    )
    plt.axvline(lasso.alpha_, linestyle="--", color="black", label="alpha: CV estimate")

    # plt.ylim(ymin, ymax)
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Mean square error")
    plt.legend()
    _ = plt.title(
        f"Mean square error on each fold: coordinate descent (train time: {fit_time:.2f}s)"
    )
    plt.savefig(os.path.join(save_path, case_name + '_lassoCV_' + parameter_name + '.png'))
    plt.close()

    eps = 5e-3  # the smaller it is the longer is the path

    # lasso path
    print("Computing regularization path using the lasso...")
    alphas_lasso, coefs_lasso, _ = lasso_path(x_train, y_train.reshape(-1), alphas=np.logspace(-6, -2),
                                              max_iter=10000, eps=eps)
    plt.figure()
    colors = cycle(["b", "r", "g", "c"])
    # names = ['S', r'$S^2$', 'A', 'Y']
    neg_log_alphas_lasso = alphas_lasso  # -np.log10(alphas_lasso)
    for coef_l, c, name in zip(coefs_lasso, colors, term_names):
        l1 = plt.semilogx(neg_log_alphas_lasso, coef_l, c=c, label=name)
    plt.axvline(lasso.alpha_, linestyle="--", color="black", label="alpha: CV estimate")
    plt.xlabel(r"$\alpha$")
    plt.ylabel("coefficients")
    plt.title("Lasso Paths")
    plt.legend(loc="lower left")
    plt.axis("tight")
    plt.savefig(os.path.join(save_path, case_name + '_lassoPath_' + parameter_name + '.png'))
    plt.close()

    final_lasso = Lasso(alpha=lasso.alpha_, max_iter=10000, fit_intercept=fit_intercept)
    final_lasso.fit(x_train, y_train)
    # final_lasso.coef_
    if fit_intercept:
        coefs = np.concatenate((final_lasso.intercept_, final_lasso.coef_))
    else:
        coefs = final_lasso.coef_
    pd.DataFrame(coefs).to_csv(os.path.join(save_path, case_name + '_regression_coef_' + parameter_name + '.csv'))
