import numpy as np
import losses


class GradientBasedOptimizer:
    """
    Template for the optimizers
    """
    def __init__(self):
        self.loss = None
        self.d_loss = None
        self.gradients = []
        pass

    def init_optimizer(self, loss, layers):
        """
        Prepares the additional informations of the optimizer.
        :param loss: loss function to be minimized
        :param layers: layers of the neural network.
        """
        self.loss, self.d_loss = losses.loss_aux(loss)
        self.gradients = [(np.zeros((layer.input_size, layer.num_units)),
                           np.zeros((1, layer.num_units))) for layer in layers]

    def process_loss(self, y_true, y_pred, layers):
        """
        Implementation of backpropagation for the weight update.
        :param y_true: ground_truth value of the labels
        :param y_pred: predicted labels
        :param layers: layers of the neural network
        """
        err = self.d_loss(y_true, y_pred)
        for i in range(len(layers)-1, -1, -1):
            curr_d_weights, curr_d_bias = self.gradients[i]
            layer = layers[i]

            # Output Layer
            if i == len(layers)-1:
                err_signal = err * layer.d_f(layer.net)
                d_weights = np.dot(layer.input.T, err_signal)

            # Hidden Layer
            else:
                err_signal = bp * layer.d_f(layer.net)
                d_weights = np.dot(layer.input.T, err_signal)

            bp = np.dot(err_signal, layer.weights.T)
            d_bias = np.sum(err_signal, axis=0)
            self.gradients[i] = (curr_d_weights+d_weights, curr_d_bias+d_bias)

    def weights_update(self, i):
        pass

    def aux_params(self, weights, bias, i):
        return weights, bias

    def epoch_change(self, epoch=None):
        pass
