from lr_schedulers import StepDecayScheduler
from grid_search import GridSearch
import pickle
import numpy as np


train_set = np.genfromtxt("../cup/ML-CUP18-TR.csv", delimiter=",")[:, 1:-2]
train_targets = np.genfromtxt("../cup/ML-CUP18-TR.csv", delimiter=",")[:, -2:]

param_grid = {
    'layers': [(5, 5, 2), (10, 5, 2), (10, 10, 2), (15, 5, 2), (15, 10, 2), (15, 15, 2),
               (5, 5, 5, 2), (10, 5, 5, 2), (10, 10, 5, 2), (10, 10, 10, 2),
               (15, 5, 5, 2), (15, 10, 5, 2), (15, 10, 10, 2), (15, 15, 5, 2),
               (15, 15, 10, 2), (15, 15, 15, 2)],
    'activation': ['sigmoid'],
    'l2_lambda': [0, 1e-3, 1e-4, 1e-5, 1e-6],
    'lr': [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
    'epoch': [5000],
    'patience': [200],
    'test_size': [0.3],
    'batch_size': [8, 16, 32, len(train_set)],
    'momentum': [0, 0.7, 0.8, 0.9],
    'nesterov': [False, True],
    'lr_sched': [StepDecayScheduler(drop=0.2, epochs_drop=5), StepDecayScheduler(drop=0.5, epochs_drop=20),
                 StepDecayScheduler(drop=0.9, epochs_drop=35)]
}

gs = GridSearch(task='Regression', tuning_params=param_grid, restarts=30, random_search=1500)
results = gs.fit(train_set, train_targets)
with open('../cup/cup_random_results.pkl', 'wb') as output:
    pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)
