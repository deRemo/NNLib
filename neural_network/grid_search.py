from neural_network import NeuralNetwork
from sgd import SGD, NesterovError
from lr_schedulers import StepDecayScheduler
from losses import loss_aux
import numpy as np
import metrics
from sklearn.model_selection import ParameterGrid, StratifiedKFold, StratifiedShuffleSplit
import random
import copy
import string
import os
import shutil
import pickle

def build_by_params(task, params, input_size):
    """
    Builds a neural network by passing the parameter grid
    :task - the task to be performed (Classification of Regression)
    :grid - parameters to tune
    :input_size - the input size of the input layer
    """
    if params is None or task is None or input_size is None:
        print("You need to pass a valid parameter grid, task or input size")
        return -1
    
    for param in params:
        if type(param) is list:
            print("Don't pass lists, use tuples instead when specifying the lr_sched or the layers")
            return -1

    nn = NeuralNetwork()
    for i in range(len(params['layers'])):
        if i == 0:
            nn.add_layer('dense', params['layers'][i], params['activation'], input_size)
        else:
            if i == len(params['layers']) - 1 and task == 'Regression':
                nn.add_layer('dense', params['layers'][i], 'linear')
            else:
                nn.add_layer('dense', params['layers'][i], params['activation'])

    nn.compile(task=task,
                dropout=params['dropout'],
                l2_lambda=params['l2_lambda'],
                optimizer=SGD(lr_init=params['lr'],
                momentum=params['momentum'],
                nesterov=params['nesterov'],
                lr_sched=StepDecayScheduler(drop=params['lr_sched'][0],
                epochs_drop=params['lr_sched'][1])))
    return nn

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
    def __init__(self, task='Classification', loss='LMS', tuning_params=None, folds=3, restarts=15, random_search=None,
                 metric=None, statistics=None):
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
        if metric is None:
            if task == 'Classification':
                self.metric = 'accuracy'
            elif task == 'Regression':
                self.metric = 'loss'
        else:
            self.metric = metric
        if statistics is None:
            if task == 'Classification':
                self.statistics = ['loss', 'accuracy']
            elif task == 'Regression':
                self.statistics = ['loss']
        else:
            self.statistics = statistics

    def fit(self, dataset, targets, checkpoints=None):
        if self.folds is not None and self.folds > 0:
            return self._fit_split_(dataset, targets, checkpoints)
        else:
            return self._fit_no_split(dataset, targets, checkpoints)

    def _fit_split_(self, dataset, targets, checkpoints):
        """
                Fit the estimator with the given dataset and targets
                :param dataset: data on which the model will fit
                :param targets: ground_truth values of the given data
                :param checkpoints: name of the file on which to save current results
                """
        dir = ''.join(random.choices(string.ascii_lowercase + string.digits, k=16))
        dir = '.tmp' + dir + '/'
        os.mkdir(dir)
        grid = self.grid
        if self.folds is not None or self.folds != 0:
            if self.task == 'Classification':
                if self.folds > 1:
                    sf = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=0)
                elif 0 <= self.folds < 1:
                    sf = StratifiedShuffleSplit(n_splits=1, test_size=self.folds, random_state=0)
            elif self.task == 'Regression':
                folds, dataset, targets = self.split_regression(dataset, targets)
        results = []
        for params in grid:
            try:
                nn = NeuralNetwork()
                for i in range(len(params['layers'])):
                    if i == 0:
                        nn.add_layer('dense', params['layers'][i], params['activation'], dataset.shape[1])
                    else:
                        if i == len(params['layers']) - 1 and self.task == 'Regression':
                            nn.add_layer('dense', params['layers'][i], 'linear')
                        else:
                            nn.add_layer('dense', params['layers'][i], params['activation'])
                curr_res = {'params': params,
                            'metric_stats': [],
                            'test_stats': [],
                            'vl_stats': [],
                            'tr_stats': []}

                if self.task == 'Classification':
                    folds = sf.split(dataset, targets)
                for train_index, test_index in folds:
                    X_train, X_test = dataset[train_index], dataset[test_index]
                    Y_train, Y_test = targets[train_index], targets[test_index]
                    nested_best = None
                    nested_best_metric = None
                    nested_tr_pred = None
                    nested_vl_pred = None
                    for i in range(self.restarts):
                        nn.compile(task=self.task,
                                   loss=self.loss_name,
                                   l2_lambda=params['l2_lambda'],
                                   dropout=params['dropout'],
                                   optimizer=SGD(lr_init=params['lr'],
                                                 momentum=params['momentum'],
                                                 nesterov=params['nesterov'],
                                                 lr_sched=StepDecayScheduler(drop=params['lr_sched'][0],
                                                                             epochs_drop=params['lr_sched'][1])))

                        curr_model, curr_metric, best_epoch = nn.fit(X_train, Y_train,
                                                                     batch_size=params['batch_size'],
                                                                     test_size=params['test_size'],
                                                                     epochs=params['epoch'],
                                                                     patience=params['patience'],
                                                                     save_pred=dir + 'tmp_gs',
                                                                     save_model=None)

                        nested_best_metric = metrics.metric_improve(self.metric, nested_best_metric, curr_metric)
                        if nested_best_metric[1]:
                            nested_tr_pred = np.load(dir + 'tmp_gs_tr_predictions.npy')[best_epoch]
                            nested_vl_pred = np.load(dir + 'tmp_gs_vl_predictions.npy')[best_epoch]
                            nested_best = copy.deepcopy(curr_model)
                            if nested_best_metric[2]:
                                break

                    Y_pred = nested_best.predict(X_test)
                    if self.metric == 'loss':
                        curr_metric = np.sum(self.loss(Y_test, Y_pred), axis=0) / len(Y_test)
                    else:
                        curr_metric = metrics.metric_computation(self.metric, Y_test, Y_pred)

                    curr_res['metric_stats'].append(curr_metric)
                    tr_stats = []
                    vl_stats = []
                    test_stats = []
                    for stat in self.statistics:
                        if stat == 'loss':

                            tr_stats.append(np.mean(self.loss(nested_tr_pred[:, :targets.shape[1]],
                                                              nested_tr_pred[:, targets.shape[1]:])))
                            vl_stats.append(np.mean(self.loss(nested_vl_pred[:, :targets.shape[1]],
                                                              nested_vl_pred[:, targets.shape[1]:])))
                            test_stats.append(np.mean(self.loss(Y_test, Y_pred)))
                        else:
                            tr_stats.append(metrics.metric_computation(stat,
                                                                       nested_tr_pred[:, :targets.shape[1]],
                                                                       nested_tr_pred[:, targets.shape[1]:]))
                            vl_stats.append(metrics.metric_computation(stat,
                                                                       nested_vl_pred[:, :targets.shape[1]],
                                                                       nested_vl_pred[:, targets.shape[1]:]))
                            test_stats.append(metrics.metric_computation(stat, Y_test, Y_pred))
                    curr_res['tr_stats'].append(tr_stats)
                    curr_res['vl_stats'].append(vl_stats)
                    curr_res['test_stats'].append(test_stats)

                results.append(curr_res)
                if checkpoints is not None:
                    with open(checkpoints + '.pkl', 'wb') as output:
                        pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)

            except NesterovError:
                continue
        shutil.rmtree(dir)
        return results

    def _fit_no_split(self, dataset, targets, checkpoints):
        """
                Fit the estimator with the given dataset and targets
                :param dataset: data on which the model will fit
                :param targets: ground_truth values of the given data
                :param checkpoints: name of the file on which to save current results
                """
        dir = ''.join(random.choices(string.ascii_lowercase + string.digits, k=16))
        dir = '.tmp' + dir + '/'
        os.mkdir(dir)
        grid = self.grid
        results = []
        for params in grid:
            try:
                nn = NeuralNetwork()
                for i in range(len(params['layers'])):
                    if i == 0:
                        nn.add_layer('dense', params['layers'][i], params['activation'], dataset.shape[1])
                    else:
                        if i == len(params['layers']) - 1 and self.task == 'Regression':
                            nn.add_layer('dense', params['layers'][i], 'linear')
                        else:
                            nn.add_layer('dense', params['layers'][i], params['activation'])
                curr_res = {'params': params,
                            'vl_stats': [],
                            'tr_stats': []}

                nested_best_metric = None
                nested_tr_pred = None
                nested_vl_pred = None
                for i in range(self.restarts):
                    nn.compile(task=self.task,
                               loss=self.loss_name,
                               l2_lambda=params['l2_lambda'],
                               optimizer=SGD(lr_init=params['lr'],
                                             momentum=params['momentum'],
                                             nesterov=params['nesterov'],
                                             lr_sched=StepDecayScheduler(drop=params['lr_sched'][0],
                                                                         epochs_drop=params['lr_sched'][1])))

                    curr_model, curr_metric, best_epoch = nn.fit(dataset, targets,
                                                                 batch_size=params['batch_size'],
                                                                 test_size=params['test_size'],
                                                                 epochs=params['epoch'],
                                                                 patience=params['patience'],
                                                                 save_pred=dir + 'tmp_gs',
                                                                 save_model=None)

                    nested_best_metric = metrics.metric_improve(self.metric, nested_best_metric, curr_metric)
                    if nested_best_metric[1]:
                        nested_tr_pred = np.load(dir + 'tmp_gs_tr_predictions.npy')[best_epoch]
                        nested_vl_pred = np.load(dir + 'tmp_gs_vl_predictions.npy')[best_epoch]
                        if nested_best_metric[2]:
                            break

                tr_stats = []
                vl_stats = []
                for stat in self.statistics:
                    if stat == 'loss':
                        tr_stats.append(np.mean(self.loss(nested_tr_pred[:, :targets.shape[1]],
                                                          nested_tr_pred[:, targets.shape[1]:])))
                        vl_stats.append(np.mean(self.loss(nested_vl_pred[:, :targets.shape[1]],
                                                          nested_vl_pred[:, targets.shape[1]:])))
                    else:
                        tr_stats.append(metrics.metric_computation(stat,
                                                                   nested_tr_pred[:, :targets.shape[1]],
                                                                   nested_tr_pred[:, targets.shape[1]:]))
                        vl_stats.append(metrics.metric_computation(stat,
                                                                   nested_vl_pred[:, :targets.shape[1]],
                                                                   nested_vl_pred[:, targets.shape[1]:]))
                curr_res['tr_stats'].append(tr_stats)
                curr_res['vl_stats'].append(vl_stats)

                results.append(curr_res)
                if checkpoints is not None:
                    with open(checkpoints + '.pkl', 'wb') as output:
                        pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)

            except NesterovError:
                continue
        shutil.rmtree(dir)
        return results

    def split_regression(self, dataset, targets):
        """
        Splits the dataset for a regression task
        :param dataset: dataset to be splitted
        :param targets: targets of the dataset
        :return: indices of every fold, ordered dataset, ordered targets
        """
        index = np.argsort(targets[:, 0])
        dataset, targets = dataset[index], targets[index]
        indices = [([], []) for _ in range(self.folds)]
        for i in range(0, len(targets), self.folds):
            if i + self.folds < len(targets):
                for j in range(self.folds):
                        for k in range(self.folds):
                            if k == j and k + i < len(targets):
                                indices[j][1].append(i+k)
                            elif k != j:
                                indices[j][0].append(i+k)
        return indices, dataset, targets
        


