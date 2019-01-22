import numpy as np
from layer import Layer
from losses import loss_aux

import pickle
from copy import deepcopy
import random
import math


class NeuralNetwork:
    """
    Models a neural network.

        Attributes:
            task(str) - the task to be performed
            layers(list) - layers of the neural network
            loss(func) - the loss function exploited by the neural network
            optimizer(obj) - optimizer of the neural network (currently SGD only, later bundle method's too)
            l2_lambda(float) - lambda value for L2 regularization
            drop_or_not(np.vectorize) - aux function for the computation of the dropout mask
    """
    def __init__(self):
        self.task = None
        self.loss = None
        self.layers = []
        self.optimizer = None
        self.l2_lambda = None
        self.dropout = None
        self.drop_or_not = None

    def add_layer(self, conn, num_units, activation, input_size=None):
        """
        Adds a layer in the neural network.
        :param conn: if the layer dense or sparse
        :param num_units: number of units of the layer
        :param activation: name of the activation function
        :param input_size: size of the input (necessary for the input layer, inferred for hidden layers)
        """
        if conn == "dense" or conn == "Dense" or conn == "d" or conn == "D":
            if input_size is None:
                input_size = self.layers[-1].num_units
            self.layers.append(Layer(num_units, activation, input_size))

    def compile(self, task='Classification', loss='LMS', optimizer=None, l2_lambda=None, dropout=None):
        """
        Initializes the model for the training.
        :param task: task to be performed by the model ('Classification' or 'Regression')
        :param loss: loss function used by the model
        :param optimizer: optimizer used during the training
        :param l2_lambda: defines the lambda parameter for L2 regularization. None if not regularized.
        :param dropout: list of the probabilities of dropout for each layer
        """
        self.task = task
        self.loss = loss_aux(loss)[0]
        self.optimizer = optimizer
        self.optimizer.init_optimizer(loss=loss, layers=self.layers)
        if l2_lambda is not None:
            self.l2_lambda = l2_lambda
        if dropout is not None:
            self.dropout = dropout
            self.drop_or_not = np.vectorize(self._drop_or_not_)
        for i, layer in enumerate(self.layers):
            if i < len(self.layers)-1:
                layer.init_weights(self.layers[i+1].num_units)
            else:
                layer.init_weights(None)

    def _feed_forward_(self, batch):
        """
        Prediction with preparation of back propagation aux variables.
        :param batch: np.array of shape [batch_size, n_features]
        :return: predicted value
        """
        current_input = batch
        for i, layer in enumerate(self.layers):
            layer.input = current_input
            tmp_weights, tmp_bias = self.optimizer.aux_params(layer.weights, layer.bias, i)
            layer.net = np.dot(layer.input, tmp_weights) + tmp_bias
            layer.out = layer.f(layer.net)

            current_input = layer.out
        return current_input

    def fit(self, dataset, targets, batch_size=32, buffer_size=None, test_size=0.3, epochs=100, patience=10,
            save_stats=None, save_model='best_nn', verbose=False):
        """
        Fits the neural network on the given dataset with a training-validation cycle (the split is performed by the
        function itself with the proportions provided by test_size).
        The fitting process is performed as follows:
            1) The model fits on the training set by performing backpropagation every buffer_size records: this allows
                to avoid to maintain in memory too many informations about the records without losing parallelism within
                a batch;
            2) After batch_size records (with batch_size >= buffer_size) the model performs the weights update with the
                informations held by backpropagation into the network;
            3) After an epoch of training the model performs the validation on the dedicated portion of the dataset.
                If the loss does not decrease for #patience iterations, the algorithm performs an early stopping.
        :param dataset: data on which the model will fit
        :param targets: ground_truth values of the given data
        :param batch_size: if ==1 perform an online training,  if > 1 and < len(dataset) performs a mini batch training,
                                if == -1 or >= len(dataset) perform a batch training
        :param buffer_size: number of records before calling backpropagation
        :param test_size: portion of dataset to be held for validation
        :param epochs: number of epochs of training
        :param patience: number of epochs without gain before calling early stopping
        :param save_stats: whether or not to save the statistics
        :param save_model: whether or not to save in memory the best model
        :param verbose: whether or not print informations about training process
        :return: the best model
        """
        if buffer_size is None:
            if batch_size < 2048:
                buffer_size = batch_size
            else:
                buffer_size = 2048
        iter_no_gain = 0
        best_loss = float('Inf')
        best_accuracy = float(0)
        best_model = None

        tr_loss = []
        vl_loss = []
        if self.task == 'Classification':
            tr_accuracy = []
            vl_accuracy = []

        train_set, valid_set, train_targets, valid_targets = self._prepare_dataset_fit_(dataset, targets, test_size)
        train_indices = list(range(len(train_set)))
        for i in range(1, epochs+1):
            # -------------------------------------------------------------------- #
            #                            TRAINING                                  #
            # -------------------------------------------------------------------- #
            tr_epoch_loss = np.zeros((len(train_set),))
            tr_prediction = np.zeros((len(train_set),))
            random.shuffle(train_indices)
            train_set, train_targets = train_set[train_indices], train_targets[train_indices]

            j = 0
            curr_batch = 1
            while j < len(train_set):
                train_slice = min(j+buffer_size, (curr_batch*batch_size)-j, len(train_set)-j)
                y_true = train_targets[j:j+train_slice]
                y_pred = self._feed_forward_(train_set[j:j+train_slice, :])
                self.optimizer.process_loss(y_true, y_pred, self.layers)
                tr_epoch_loss[j:j+train_slice] = self.loss(y_true, y_pred)[:, 0]
                if self.task == 'Classification':
                    tr_prediction[j:j+train_slice] = y_pred[:, 0]
                j += train_slice
                if (j != 0 and j % batch_size == 0) or j == len(train_set):
                    self._weights_update_()
                    curr_batch += 1

            self.optimizer.epoch_change(i)

            tr_loss.append(np.sum(tr_epoch_loss, axis=0))
            if self.task == 'Classification':
                correct = len(train_targets[train_targets[:, 0] == tr_prediction.round()])
                accuracy = correct / len(tr_prediction)
                vl_accuracy.append(accuracy)

            # -------------------------------------------------------------------- #
            #                           VALIDATION                                 #
            # -------------------------------------------------------------------- #
            y_true = valid_targets
            y_pred = self.predict(valid_set)
            validation_loss = np.sum(self.loss(y_true, y_pred), axis=0)
            vl_loss.append(validation_loss)
            if self.task == 'Classification':
                correct = len(y_true[y_true == y_pred.round()])
                accuracy = correct/len(y_true)
                vl_accuracy.append(accuracy)

            if (self.task == 'Regression' and validation_loss < best_loss) or \
                    (self.task == 'Classification' and accuracy > best_accuracy):

                if self.task == 'Classification' and accuracy > best_accuracy:
                    best_accuracy = accuracy
                if self.task == 'Regression' and validation_loss < best_loss:
                    best_loss = validation_loss

                best_model = deepcopy(self)
                if save_model is not None:
                    with open(save_model + '.pkl', 'wb') as output:
                        pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

                if verbose:
                    if self.task == 'Regression':
                        print("Loss improved at epoch ", i, ": ", validation_loss / len(valid_targets))
                    elif self.task == 'Classification':
                        print("Accuracy improved at epoch ", i, ": ", best_accuracy)

                if (self.task == 'Regression' and best_loss == 0) or \
                        (self.task == 'Classification' and best_accuracy == 1.0):
                    break
                iter_no_gain = 0
            else:
                iter_no_gain += 1
                if patience is not None and iter_no_gain == patience:
                    break
        if save_stats is not None:
            np.save(save_stats+"_tr_loss.npy", np.array(tr_loss))
            np.save(save_stats+"_vl_loss.npy", np.array(vl_loss))
            if self.task == 'Classification':
                np.save(save_stats+"_tr_accuracy.npy", np.array(tr_accuracy))
                np.save(save_stats+"_vl_accuracy.npy", np.array(vl_accuracy))

        return best_model

    def _weights_update_(self):
        """
        Computes the weight update caring of the adopted regularization techniques.
        """
        for i in range(len(self.layers)):
            layer = self.layers[i]
            w_update, bias_update = self.optimizer.weights_update(i)
            if self.l2_lambda is not None:
                w_update -= 2*self.l2_lambda*layer.weights
            if self.dropout is not None:
                mask = self.drop_or_not(np.random.random(layer.num_units), self.dropout[i])
                w_update *= mask.T
            layer.weights += w_update
            layer.bias += bias_update

    def predict(self, batch):
        """
        Prediction of the input data.
        :param batch: np.array of shape [batch_size, n_features]
        :return: predicted value
        """
        prediction = batch
        for curr_layer in self.layers:
            prediction = np.dot(prediction, curr_layer.weights) + curr_layer.bias  # dot(in,w)+bias
            prediction = curr_layer.f(prediction)

        if self.task == 'Classification':
            return prediction.round()
        else:
            return prediction

    def _prepare_dataset_fit_(self, dataset, targets, test_size):
        """
        Prepares a stratified split of the dataset for both classification and regression tasks.
        :param dataset: the dataset to be splitted
        :param targets: the corresponding targets of the dataset
        :param test_size: the proportion of the test set ((0.0, 1.0))
        :return: tuple (training_set, validation_set, training_targets, validation_targets)
        """
        if len(targets.shape) == 1:
            targets = np.expand_dims(targets, axis=1)
        if self.task == 'Classification':
            tmp_data = np.concatenate([dataset, targets], axis=1)
            train_set, valid_set, train_targets, valid_targets = [], [], [], []
            for value in np.unique(targets):
                curr_split = tmp_data[tmp_data[:, -1] == value]
                np.random.shuffle(curr_split)
                test_split = curr_split[:math.floor(len(curr_split)*test_size)]
                train_split = curr_split[math.floor(len(curr_split)*test_size):]
                train_set += train_split[:, :-1].tolist()
                train_targets += train_split[:, -1].tolist()
                valid_set += test_split[:, :-1].tolist()
                valid_targets += test_split[:, -1].tolist()
            train_set = np.array(train_set)
            train_targets = np.expand_dims(np.array(train_targets), axis=1)
            valid_set = np.array(valid_set)
            valid_targets = np.expand_dims(np.array(valid_targets), axis=1)

        elif self.task == 'Regression':
            index = np.argsort(targets[:, 0])
            step = math.floor(1./test_size)
            dataset, targets = dataset[index], targets[index]
            train_indices = []
            val_indices = []
            for i in range(0, len(targets), step):
                to_add = random.randint(0, step)
                for j in range(step+1):
                    if j == to_add and to_add + i < len(targets):
                        val_indices.append(to_add + i)
                    elif j != to_add and j+i < len(targets):
                        train_indices.append(j+i)
            train_set, valid_set, train_targets, valid_targets = \
                dataset[train_indices], dataset[val_indices], targets[train_indices], targets[val_indices]
        return train_set, valid_set, train_targets, valid_targets

    def _drop_or_not_(self, x, y):
        if x >= y:
            return 1
        else:
            return 0
