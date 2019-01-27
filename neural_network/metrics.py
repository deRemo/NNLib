

def metric_improve(metric, curr_best_metric, curr_metric):
    """
    Determines whether or not the metric has got an improvement.
    :param metric: the type of metric
    :param curr_best_metric: current best value of the metric
    :param curr_metric: current value of the metric
    :return: tuple(metric value, improvement(bool), reached best value for the metric (bool))
    """
    if metric == 'accuracy':
        if curr_best_metric is None or curr_metric > curr_best_metric[0]:
            return curr_metric, True, curr_metric == 1.0
        else:
            return curr_best_metric[0], False, curr_best_metric[2]
    elif metric == 'loss':
        if curr_best_metric is None or curr_metric < curr_best_metric[0]:
            return curr_metric, True, curr_metric == 0
        else:
            return curr_best_metric[0], False, curr_best_metric[2]


def metric_computation(metric, y_true, y_pred):
    """
    Computes the defined metric
    :param metric: metric to be computed
    :param y_true: ground_truth values
    :param y_pred: predicted values
    :return: metric value
    """

    if metric == 'accuracy':
        y_pred = y_pred.round()
        return len(y_true[y_true == y_pred]) / len(y_true)
