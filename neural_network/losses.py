import numpy as np


# Handler function to make easier the choice's logic
def loss_aux(f):
    """
    Aux function to return loss function and its derivative
    :param f: loss function to be returned
    :return: (loss, loss_derivative)
    """
    loss_dict = {
        'LMS': (lms, d_lms),
        # 'categorical_crossentropy': (lambda x: tanh(x), lambda x: d_tanh(x)),
    }
    return loss_dict[f]


def lms(y_true, y_pred):
    lms_err = y_true - y_pred
    lms_err *= lms_err
    if y_true.shape[1] > 1:
        lms_err = np.sum(lms_err, axis=1)
    return lms_err/2


def d_lms(y_true, y_pred):
    return y_true-y_pred


# DA IMPLEMENTARE
def categorical_crossentropy(y_true, y_pred):
    pass


# DA IMPLEMENTARE
def d_categorical_crossentropy(y_true, y_pred):
    pass
