from neural_network import NeuralNetwork
from sgd import SGD
from quickprop import Quickprop
from rprop import Rprop
from lr_schedulers import ExponentialDecayScheduler, TimeBasedScheduler, StepDecayScheduler
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
import random

class GridSearch:
    """
    GridSearch implementation
        Attributes:
        tuning_params(dict) - parameters to tune
        estimator(neural_network) - the neural network to fit with the tuned parameters
    """
    def __init__(self, tuning_params, estimator):
        self.tuning_params=tuning_params
        self.estimator=estimator

    def fit(self, dataset, targets):
        """
        Fit the estimator with the given dataset and targets
        :param dataset: data on which the model will fit
        :param targets: ground_truth values of the given data
        """
        best_accuracy=0
        grid = ParameterGrid(self.tuning_params)

        for params in grid:
            self.estimator.compile(optimizer=SGD(lr_init=params['lr'], momentum=params['momentum'], nesterov=params['nesterov'], lr_sched=StepDecayScheduler(epochs_drop=15)))
            self.estimator.fit(dataset, targets, batch_size=params['batch_size'], test_size=params['test_size'], epochs=params['epoch'], patience=params['patience'], save_stats=True)

            accuracy=np.amax(np.load("vl_accuracy.npy"))
            if(accuracy > best_accuracy):
                best_accuracy=accuracy
                best_param={x:params[x] for x in list(params.keys())}
                
        return best_accuracy, best_param

class RandomSearch:
    """
    RandomSearch implementation
        Attributes:
        tuning_params(dict) - parameters to tune
        estimator(neural_network) - the neural network to fit with the tuned parameters
        iterations(int) - how many iteration
    """
    def __init__(self, tuning_params, estimator, iterations):
        self.tuning_params=tuning_params
        self.estimator=estimator
        self.iterations=iterations
    
    def fit(self, dataset, targets):
        """
        Fit the estimator with the given dataset and targets picking random values
        :param dataset: data on which the model will fit
        :param targets: ground_truth values of the given data
        """
        best_accuracy=0

        for i in range(self.iterations):
            params={}
            for key in self.tuning_params:
                params[key]=random.choice(self.tuning_params[key])

            self.estimator.compile(optimizer=SGD(lr_init=params['lr'], momentum=params['momentum'], nesterov=params['nesterov'], lr_sched=StepDecayScheduler(epochs_drop=15)))
            self.estimator.fit(dataset, targets, batch_size=params['batch_size'], test_size=params['test_size'], epochs=params['epoch'], patience=params['patience'], save_stats=True)

            
            accuracy=np.amax(np.load("vl_accuracy.npy"))
            if(accuracy > best_accuracy):
                best_accuracy=accuracy
                best_param={x:params[x] for x in list(params.keys())}
        
        return best_accuracy, best_param