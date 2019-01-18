import numpy as np
from optimizer import Optimizer
from copy import deepcopy


class Rprop(Optimizer):

    def __init__(self, delta_0=0.1, lr_low=0.5, lr_high=1.2, delta_max=50, backtracking=False):
        super(Rprop, self).__init__()
        self.method = 'RPROP'
        self.backtracking = backtracking
        self.lr_low = lr_low
        self.lr_high = lr_high
        self.delta_min = 1e-6
        self.delta_0 = delta_0
        self.delta_max = delta_max
        self.prev_steps = None
        self.prev_gradients = None
        if self.backtracking:
            self.prev_updates = None

    def init_optimizer(self, layers):
        """
        Prepares the additional informations of the optimizer.
        :param layers: layers of the neural network.
        """
        self.prev_steps = [(np.full(shape=layer.d_weights.shape, fill_value=self.delta_0),
                           (np.full(shape=layer.d_bias.shape, fill_value=self.delta_0))) for layer in layers]
        self.prev_gradients = [(np.zeros(layer.d_weights.shape),
                               (np.zeros(layer.d_bias.shape))) for layer in layers]
        if self.backtracking:
            self.prev_updates = [(np.zeros(layer.d_weights.shape),
                                 (np.zeros(layer.d_bias.shape))) for layer in layers]

    def weights_update(self, w_grad, b_grad):
        """
        Function for the weight update by RPROP.
        :param w_grad: weights' current gradient
        :param b_grad: bias current gradient
        :return: the value for the weights and bias update by Resilient Propagation
        """
        if self.backtracking:
            return self._backtracking_rprop_(w_grad, b_grad)
        else:
            return self._no_backtracking_rprop_(w_grad, b_grad)

    def _backtracking_rprop_(self, w_grad, b_grad):
        prev_w_step, prev_b_step = self.prev_steps.pop(0)
        prev_w_grad, prev_b_grad = self.prev_gradients.pop(0)
        prev_w_update, prev_b_update = self.prev_updates.pop(0)

        prod_w_grad = prev_w_grad * w_grad
        prod_b_grad = prev_b_grad * b_grad
        curr_w_step = deepcopy(prev_w_step)
        curr_b_step = deepcopy(prev_b_step)
        w_update = np.zeros(w_grad.shape)
        b_update = np.zeros(b_grad.shape)

        for j in range(w_grad.shape[1]):  # For each neuron j
            for i in range(w_grad.shape[0]):  # For each weight i of neuron j
                if prod_w_grad[i, j] >= 0:
                    if prod_w_grad[i, j] > 0:  # Descending step ---> boost stepsize (if ==0 we use the same stepsize)
                        curr_w_step[i, j] = min(prev_w_step[i, j] * self.lr_high, self.delta_max)
                    if w_grad[i, j] == 0:
                        coeff = +1
                    else:
                        coeff = -np.sign(w_grad[i, j])
                    w_update[i, j] = coeff * curr_w_step[i, j]

                else:   # Ascending step ---> backtrack
                    curr_w_step[i, j] = max(prev_w_step[i, j] * self.lr_low, self.delta_min)
                    w_update[i, j] = -prev_w_update[i, j]
                    w_grad[i, j] = 0

            # Updating bias of neuron j
            if prod_b_grad[0, j] >= 0:
                if prod_b_grad[0, j] > 0:
                    curr_b_step[0, j] = min(prev_b_step[0, j] * self.lr_high, self.delta_max)
                if b_grad[0, j] == 0:
                    coeff = +1
                else:
                    coeff = -np.sign(b_grad[0, j])
                b_update[0, j] = coeff * curr_b_step[0, j]

            else:
                curr_b_step[0, j] = max(prev_b_step[0, j] * self.lr_low, self.delta_min)
                b_update[0, j] = -prev_b_update[0, j]
                b_grad[0, j] = 0

        self.prev_steps.append((curr_w_step, curr_b_step))
        self.prev_gradients.append((w_grad, b_grad))
        self.prev_updates.append((w_update, b_update))
        return w_update, b_update

    def _no_backtracking_rprop_(self, w_grad, b_grad):
        prev_w_step, prev_b_step = self.prev_steps.pop(0)
        prev_w_grad, prev_b_grad = self.prev_gradients.pop(0)

        prod_w_grad = prev_w_grad * w_grad
        prod_b_grad = prev_b_grad * b_grad
        curr_w_step = deepcopy(prev_w_step)
        curr_b_step = deepcopy(prev_b_step)
        w_update = np.zeros(w_grad.shape)
        b_update = np.zeros(b_grad.shape)

        for j in range(w_grad.shape[1]):  # For each neuron j
            for i in range(w_grad.shape[0]):  # For each weight i of neuron j
                if prod_w_grad[i, j] > 0:
                    curr_w_step[i, j] = min(prev_w_step[i, j] * self.lr_high, self.delta_max)
                elif prod_w_grad[i, j] < 0:
                    curr_w_step[i, j] = max(prev_w_step[i, j] * self.lr_low, self.delta_min)
                if w_grad[i, j] == 0:
                    coeff = +1
                else:
                    coeff = -np.sign(w_grad[i, j])
                w_update[i, j] = coeff * curr_w_step[i, j]

            # Updating bias of neuron j
            if prod_b_grad[0, j] > 0:
                curr_b_step[0, j] = min(prev_b_step[0, j] * self.lr_high, self.delta_max)
            elif prod_b_grad[0, j] < 0:
                curr_b_step[0, j] = max(prev_b_step[0, j] * self.lr_low, self.delta_min)
            if b_grad[0, j] == 0:
                coeff = +1
            else:
                coeff = -np.sign(b_grad[0, j])
            b_update[0, j] = coeff * curr_b_step[0, j]

        self.prev_steps.append((curr_w_step, curr_b_step))
        self.prev_gradients.append((w_grad, b_grad))

        return w_update, b_update
