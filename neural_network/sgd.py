import numpy as np
from gradient_based_optimizer import GradientBasedOptimizer


class SGD(GradientBasedOptimizer):
    """
    Stochastic Gradient Descent optimizer.
        method(str) - declares that the optimizer is a SGD optimizer
        lr(float) - current value of learning rate
        lr_init(float) - initial value of learning rate
        momentum(float) - momentum's value
        prev_updates(list) - takes trace of the updates at the previous iteration (used only with momentum descent)
        nesterov(bool) - True if Nesterov Accelerated Gradient is performed
        lr_sched(str) -  the technique of the learning rate's update
    """
    def __init__(self, lr_init=0.1, momentum=None, nesterov=False, lr_sched=None):
        super(SGD, self).__init__()
        if nesterov and (momentum is None or momentum == 0):
            raise NesterovError
        self.method = 'SGD'
        self.lr = lr_init
        self.lr_init = lr_init
        self.momentum = momentum
        self.lr_sched = lr_sched
        self.nesterov = nesterov

        if momentum is not None and momentum != 0:
            self.prev_updates = None

    def init_optimizer(self, loss, layers):
        super(SGD, self).init_optimizer(loss, layers)
        if self.momentum is not None:
            self.prev_updates = [(np.zeros((layer.input_size, layer.num_units)),
                                  np.zeros((1, layer.num_units))) for layer in layers]

    def weights_update(self, i):
        """
        Function for the weight update by SGD.
        :param i: index of the layer to update
        :returns: the value for the weights and bias update by SGD
        """
        w_grad, b_grad = self.gradients[i]
        if self.momentum is None or self.momentum == 0:
            return self.lr * w_grad, self.lr * b_grad
        else:
            prev_w_up, prev_b_up = self.prev_updates[i]
            w_update = self.momentum * prev_w_up + self.lr * w_grad
            b_update = self.momentum * prev_b_up + self.lr * b_grad
            self.gradients[i] = (np.zeros(w_grad.shape), np.zeros(b_grad.shape))
            self.prev_updates[i] = (w_update, b_update)
            return w_update, b_update

    def aux_params(self, weights, bias, i):
        """
        Aux function for temporary modifications of a layer's parameters during the feed forward.
        :param weights: current weights of the layer
        :param bias: current bias of the layer
        :param i: index of the layer
        """
        if self.nesterov:
            prev_g_weights, prev_g_bias = self.prev_updates[i]
            tmp_weights = weights + self.momentum*prev_g_weights
            tmp_bias = bias + self.momentum*prev_g_bias
            return tmp_weights, tmp_bias
        else:
            return weights, bias

    def epoch_change(self, epoch=None):
        """
        Updates the learning rate with the technique described by lr_sched
        :param epoch: current epoch's number
        """
        if self.lr_sched is not None:
            self.lr = self.lr_sched.lr_update((self.lr, self.lr_init), epoch)


class NesterovError(Exception):
    def __init__(self):
        # Call the base class constructor with the parameters it needs
        super().__init__("Cannot use Nesterov's Accelerated Gradient with 0 momentum's value.")
