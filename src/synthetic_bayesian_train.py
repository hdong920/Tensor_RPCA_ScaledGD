import optuna 
import torch
import numpy as np
import random
import tensorly as tl
import json

from rpca import rpca
from training_functions import generate_problem

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

optuna.logging.set_verbosity(optuna.logging.WARNING)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tl.set_backend('pytorch')

samples = 20 # number of samples
n_trials = 500 # iterations of baseline to run
shape = [100, 100, 100] # Tensor shape
T = 100 # number of iterative updates of RPCA

# Grid of scenarios to experiement
kappas = [5]
alphas = [0.2, 0.3] 
ranks = [10, 20]

epsilon = 1e-7

out_dir = '../outputs/synthetic'


l1_loss = lambda Y, X_t: (Y-X_t).norm(p=1) / (Y.norm()**2)
X_loss = lambda X_star, X_t: (X_t - X_star).norm()**2 / (X_star.norm()**2)
reconstruction_loss = lambda Y, X_t, S_t: (Y - X_t - S_t).norm()**2 / (Y.norm()**2)

def objective(trial, Y, r):
    eta = trial.suggest_float('eta', 0, 3)
    z0 = trial.suggest_float('z0', 0.001, 0.05)
    z1 = trial.suggest_float('z1', 0.001, 0.05)
    decay = trial.suggest_float('decay', 0.001, 0.999)
    try:
        X_t, S_t = rpca(Y, [r, r, r], z0, z1, eta, decay, T, epsilon, device)
        return l1_loss(Y, X_t)
    except:
        return float('inf')

data_generator = lambda ranks, shape, alpha, kappa: generate_problem(ranks, shape, alpha, kappa, device)

for kap in kappas:
    for a in alphas:
        for r in ranks:
            metrics_path = f'{out_dir}/bayesian_kappa{kap}_alpha{int(10 * a)}_rank{r}.json'
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                start_i = max([int(i) for i in metrics.keys()]) + 1
            except:
                metrics = dict()
                start_i = 0
            print(f'Sampling {kap}, {a}, {r}')
            for s in range(start_i, start_i + samples):
                print(s)
                metrics[s] = dict()
                Y, X_star = data_generator(ranks=(r, r, r), shape=shape, alpha=a, kappa=kap)
                order = len(Y.shape)
                objective_wrapper = lambda trial: objective(trial, Y, r)
                study = optuna.create_study()
                study.optimize(objective_wrapper, n_trials=n_trials, gc_after_trial=True)
                

                metrics[s]['eta'] = study.best_params['eta']
                metrics[s]['z0'] = study.best_params['z0']
                metrics[s]['z1'] = study.best_params['z1']
                metrics[s]['decay'] = study.best_params['decay']

                X_t, S_t = rpca(Y, [r, r, r], metrics[s]['z0'], metrics[s]['z1'], metrics[s]['eta'], metrics[s]['decay'], T, epsilon, device)
                metrics[s]['loss'] = l1_loss(Y, X_t).item()
                metrics[s]['X_loss'] = X_loss(X_star, X_t).item()
                metrics[s]['reconstruction_loss'] = reconstruction_loss(Y, X_t, S_t).item()

                with open(metrics_path, 'w') as fp:
                    json.dump(metrics, fp)

