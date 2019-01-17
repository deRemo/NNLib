import numpy as np


class SGD:
    """
    Stochastic Gradient Descent optimizer.
        method(str) - declares that the optimizer is a SGD optimizer
        lr(float) - current value of learning rate
        lr_init(float) - initial value of learning rate
        momentum(float) - momentum's value
        prev_gradients(list) - takes trace of the gradients at the previous iteration (used only with momentum descent)
        nesterov(bool) - True if Nesterov's momentum is used in the weight update
        layer_deriv(list) - takes trace of the layer's activation derivative (used only with NAG)
        lr_sched(str) -  the technique of the learning rate's update
    """
    def __init__(self, lr_init=0.1, momentum=None, nesterov=False, lr_sched=None):
        self.method = 'SGD'
        self.lr = lr_init
        self.lr_init = lr_init
        self.momentum = momentum
        self.lr_sched = lr_sched
        self.nesterov = nesterov

        if momentum != 0:
            self.prev_gradients = None
        if nesterov:
            self.layer_deriv = None

    def weights_update(self, w_grad, bias_grad):
        """
        Function for the weight update by SGD.
        :param w_grad: gradient of a layer's weights
        :param bias_grad: gradient of a layer's bias
        :returns: the value for the weight update by SGD
        """
        if self.momentum is None or self.momentum == 0:
            return self.lr * w_grad, self.lr * bias_grad
        else:
            prev_w_grad, prev_b_grad = self.prev_gradients.pop(0)
            w_update = self.momentum * prev_w_grad + self.lr * w_grad
            bias_update = self.momentum * prev_b_grad + self.lr * bias_grad
            self.prev_gradients.append((w_update, bias_update))
            return w_update, bias_update

    def aux_params(self, weights, bias, i):
        """
        Aux function for temporary modifications of a layer's parameters during the feed forward.
        :param weights: current weights of the layer
        :param bias: current bias of the layer
        :param i: index of the layer
        """
        if self.nesterov:
            prev_g_weights, prev_g_bias = self.prev_gradients[i]
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
        if self.momentum != 0:
            for i in range(len(self.prev_gradients)):
                self.prev_gradients[i] = (np.zeros(self.prev_gradients[i][0].shape),
                                          np.zeros(self.prev_gradients[i][1].shape))
