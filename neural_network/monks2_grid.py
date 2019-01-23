from lr_schedulers import StepDecayScheduler
from grid_search import GridSearch
import pickle
import numpy as np


# MONK PROBLEMS
train_set = np.genfromtxt("../monks/monks2-train.txt", delimiter=" ", dtype="int")[:,1:-1]
train_targets = np.genfromtxt("../monks/monks2-train.txt", delimiter=" ", dtype="int")[:,:1]

param_grid = {
    'layers': [(5, 1), (10, 1), (15, 1),
               (5, 5, 1), (10, 5, 1), (10, 10, 1), (15, 5, 1), (15, 10, 1), (15, 15, 1),
               (5, 5, 5, 1), (10, 5, 5, 1), (10, 10, 5, 1), (10, 10, 10, 1),
               (15, 5, 5, 1), (15, 10, 5, 1), (15, 10, 10, 1), (15, 15, 5, 1),
               (15, 15, 10, 1), (15, 15, 15, 1)],
    'activation': ['sigmoid'],
    'l2_lambda': [0, 1e-3, 1e-4, 1e-5, 1e-6],
    'lr': [0.1, 0.15, 0.2, 0.5],
    'epoch': [5000],
    'patience': [200],
    'test_size': [0.3],
    'batch_size': [4, 8, 16, len(train_set)],

    'momentum': [0, 0.7, 0.8, 0.9],
    'nesterov': [False, True],
    'lr_sched': [StepDecayScheduler(drop=0.2, epochs_drop=5), StepDecayScheduler(drop=0.4, epochs_drop=15),
                 StepDecayScheduler(drop=0.6, epochs_drop=20), StepDecayScheduler(drop=0.9, epochs_drop=35)]
}

gs = GridSearch(tuning_params=param_grid, restarts=30)
results = gs.fit(train_set, train_targets)
with open('monks_2_results.pkl', 'wb') as output:
    pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)
