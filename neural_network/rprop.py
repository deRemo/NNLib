import numpy as np
from gradient_based_optimizer import GradientBasedOptimizer


class Rprop(GradientBasedOptimizer):

    def __init__(self, delta_0=0.1, lr_low=0.5, lr_high=1.2, delta_max=50):
        super(Rprop, self).__init__()
        self.method = 'RPROP'
        self.lr_low = lr_low
        self.lr_high = lr_high
        self.delta_min = 1e-12
        self.delta_0 = delta_0
        self.delta_max = delta_max
        self.prev_steps = None
        self.prev_gradients = None
        self.prev_updates = None
        self.curr_loss = float(0)
        self.last_loss = float('+inf')

    def process_loss(self, y_true, y_pred, layers):
        super(Rprop, self).process_loss(y_true, y_pred, layers)
        self.curr_loss += np.sum(self.loss(y_true, y_pred), axis=0)

    def init_optimizer(self, loss, layers):
        super(Rprop, self).init_optimizer(loss, layers)
        self.prev_steps = [(np.full(shape=(layer.input_size, layer.num_units), fill_value=self.delta_0),
                           (np.full(shape=(1, layer.num_units), fill_value=self.delta_0))) for layer in layers]
        self.prev_gradients = [(np.zeros((layer.input_size, layer.num_units)),
                               (np.zeros((1, layer.num_units)))) for layer in layers]
        self.prev_updates = [(np.zeros((layer.input_size, layer.num_units)),
                             (np.zeros((1, layer.num_units)))) for layer in layers]

    def weights_update(self, i):
        """
        Function for the weight update by RPROP.
        """
        w_grad, b_grad = self.gradients[i]
        curr_w_step, curr_b_step = self.prev_steps[i]
        prev_w_grad, prev_b_grad = self.prev_gradients[i]
        prev_w_update, prev_b_update = self.prev_updates[i]

        prod_w_grad = prev_w_grad * w_grad
        prod_b_grad = prev_b_grad * b_grad
        w_update = np.zeros(w_grad.shape)
        b_update = np.zeros(b_grad.shape)

        for j in range(w_grad.shape[1]):  # For each neuron j
            for k in range(w_grad.shape[0]):  # For each weight i of neuron j
                if prod_w_grad[k, j] > 0:
                    curr_w_step[k, j] = min(curr_w_step[k, j] * self.lr_high, self.delta_max)
                    w_update[k, j] = -np.sign(w_grad[k, j]) * curr_w_step[k, j]

                elif prod_w_grad[k, j] < 0:
                    curr_w_step[k, j] = max(curr_w_step[k, j] * self.lr_low, self.delta_min)
                    if self.curr_loss > self.last_loss:
                        w_update[k, j] = -prev_w_update[k, j]
                    w_grad[k, j] = 0

                else:
                    w_update[k, j] = -np.sign(w_grad[k, j]) * curr_w_step[k, j]

            # Updating bias of neuron j
            if prod_b_grad[0, j] > 0:
                curr_b_step[0, j] = min(curr_b_step[0, j] * self.lr_high, self.delta_max)
                b_update[0, j] = -np.sign(b_grad[0, j]) * curr_b_step[0, j]

            elif prod_b_grad[0, j] < 0:
                curr_b_step[0, j] = max(curr_b_step[0, j] * self.lr_low, self.delta_min)
                if self.curr_loss > self.last_loss:
                    b_update[0, j] = -prev_b_update[0, j]
                b_grad[0, j] = 0

            else:
                b_update[0, j] = -np.sign(b_grad[0, j]) * curr_b_step[0, j]

        self.gradients[i] = (np.zeros(w_grad.shape), np.zeros(b_grad.shape))
        self.prev_updates[i] = (w_update, b_update)
        self.prev_gradients[i] = (w_grad, b_grad)
        self.last_loss = self.curr_loss
        self.curr_loss = 0
        return w_update, b_update
