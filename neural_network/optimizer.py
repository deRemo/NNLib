

class Optimizer:
    """
    Template for the optimizers
    """
    def __init__(self):
        pass

    def init_optimizer(self, layers):
        pass

    def weights_update(self, w_grad, bias_grad):
        pass

    def aux_params(self, weights, bias, i):
        return weights, bias

    def epoch_change(self, epoch=None):
        pass
