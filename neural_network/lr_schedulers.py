import math


class TimeBasedScheduler:
    def __init__(self, decay=1e-6):
        self.decay = decay

    def lr_update(self, pair_lr, epoch):
        return pair_lr[0] * (1. / (1. + self.decay * epoch))


class StepDecayScheduler:
    def __init__(self, drop=0.5, epochs_drop=10.0):
        self.drop = drop
        self.epochs_drop = epochs_drop

    def lr_update(self, pair_lr, epoch):
        return pair_lr[1] * math.pow(self.drop, math.floor(epoch / self.epochs_drop))


class ExponentialDecayScheduler:
    def __init__(self, k=0.1):
        self.k = k

    def lr_update(self, pair_lr, epoch):
        return pair_lr[1] * math.exp(-self.k * epoch)
