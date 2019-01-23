from neural_network import NeuralNetwork
from sgd import SGD, NesterovError
from losses import loss_aux
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, StratifiedShuffleSplit
import random
import copy
import string
import os
import shutil


class GridSearch:
    """
    GridSearch implementation

        Attributes:
            task(str) - the task to be performed (Classification of Regression)
            loss_name(str) - name of the loss function
            loss(func) - loss function
            grid(dict) - parameters to tune
            folds(float) - determines the cross-validation splitting strategy: if cv > 1 must be an integer which
                defines the number of folds. If 0 < cv < 1 determines the portion of dataset to be held for testing
            random_search(int) - determines the number of random configurations to be tested. If None, a whole grid
                search will be performed.
    """
    def __init__(self, task='Classification', loss='LMS', tuning_params=None, folds=3, restarts=15, random_search=None):
        self.task = task
        self.loss_name = loss
        self.loss = loss_aux(loss)[0]
        self.grid = ParameterGrid(tuning_params)
        if random_search is not None:
            self.random = True
            self.grid = random.choices(self.grid, k=random_search)
        else:
            self.random = False
        self.folds = folds
        self.restarts = restarts

    def fit(self, dataset, targets):
        """
        Fit the estimator with the given dataset and targets
        :param dataset: data on which the model will fit
        :param targets: ground_truth values of the given data
        """
        dir = ''.join(random.choices(string.ascii_lowercase + string.digits, k=16))
        dir = '.tmp'+dir+'/'
        os.mkdir(dir)
        grid = self.grid
        if self.folds is not None:
            if self.folds > 1:
                sf = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=0)
            elif 0 <= self.folds < 1:
                sf = StratifiedShuffleSplit(n_splits=1, test_size=self.folds, random_state=0)
        results = []
        for params in grid:
            try:
                nn = NeuralNetwork()
                for layer in params['layers']:
                    if params['layers'].index(layer) == 0:
                        nn.add_layer('dense', layer, params['activation'], dataset.shape[1])
                    else:
                        nn.add_layer('dense', layer, params['activation'])

                conf_acc = []
                conf_loss = []
                for train_index, test_index in sf.split(dataset, targets):
                    X_train, X_test = dataset[train_index], dataset[test_index]
                    Y_train, Y_test = targets[train_index], targets[test_index]
                    nested_best = None
                    nested_best_acc = -1
                    nested_best_loss = float("+inf")
                    for i in range(self.restarts):
                        nn.compile(task=self.task,
                                   loss=self.loss_name,
                                   l2_lambda=params['l2_lambda'],
                                   optimizer=SGD(lr_init=params['lr'],
                                                 momentum=params['momentum'],
                                                 nesterov=params['nesterov'],
                                                 lr_sched=params['lr_sched']))

                        curr_model = nn.fit(X_train, Y_train,
                                            batch_size=params['batch_size'],
                                            test_size=params['test_size'],
                                            epochs=params['epoch'],
                                            patience=params['patience'],
                                            save_stats=dir+'tmp_gs')

                        loss_array = np.load(dir+   'tmp_gs_vl_loss.npy')
                        if self.task == 'Classification':
                            acc_array = np.load(dir+'tmp_gs_vl_accuracy.npy')
                            index = np.argmax(acc_array)
                            acc = acc_array[index]
                        else:
                            index = np.argmin(loss_array)
                        loss = loss_array[index]
                        if (self.task == 'Classification' and acc > nested_best_acc) or \
                                (self.task == 'Regression' and loss < nested_best_loss):
                            if self.task == 'Classification':
                                nested_best_acc = acc
                            nested_best_loss = loss
                            nested_best = copy.deepcopy(curr_model)
                            if (self.task == 'Classification' and nested_best_acc == 1) or \
                                    (self.task == 'Regression' and nested_best_loss == 0):
                                break

                    Y_pred = nested_best.predict(X_test)
                    if self.task == 'Classification':
                        conf_acc.append(accuracy_score(Y_test, Y_pred))
                        mean_acc = np.mean(conf_acc)
                    conf_loss.append(np.mean(self.loss(Y_test, Y_pred)))
                mean_loss = np.mean(conf_loss)

                if self.task == 'Classification':
                    results.append((mean_acc, mean_loss, {x: params[x] for x in list(params.keys())}))
                elif self.task == 'Regression':
                    results.append((mean_loss, {x: params[x] for x in list(params.keys())}))

            except NesterovError:
                continue
        shutil.rmtree(dir)
        return results
