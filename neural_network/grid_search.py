from neural_network import NeuralNetwork
from sgd import SGD
from quickprop import Quickprop
from rprop import Rprop
from lr_schedulers import ExponentialDecayScheduler, TimeBasedScheduler, StepDecayScheduler
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid

class GridSearch:
    """
    GridSearch implemenation
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
            self.estimator.fit(dataset, targets, batch_size=params['batch_size'], test_size=params['test_size'], epochs=params['epoch'], patience=params['patience'], save_stats=True) #lr=0.05 test_size=0.5 epochs=2000

            accuracy=np.amax(np.load("vl_accuracy.npy"))
            if(accuracy > best_accuracy):
                best_accuracy=accuracy
                best_param={x:params[x] for x in list(params.keys())}
                
        return best_accuracy, best_param