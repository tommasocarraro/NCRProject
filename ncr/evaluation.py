"""Module containing utility functions to evaluate recommendation engines.
"""
import torch
from functools import partial
import inspect
import numpy as np
from .metrics import Metrics

__all__ = ['ValidFunc']

class ValidFunc(object):
    """Wrapper class for validation functions.
    When a validation function is passed to the method train() of a model, it must have a specific signature,
    that has three parameters: model, test_loader and metric_list.
    This class has to be used to adapt any evaluation function to this signature by partially initializing
    potential additional arguments.
    Parameters
    ----------
    func : :obj:`function`
        Evaluation function that has to be wrapped. The evaluation function must match the signature
        ``func(model, test_loader, metric_list, **kwargs)``.
    Attributes
    ----------
    func_name : :obj:`str`
        The name of the evalutaion function.
    function : :obj:`function`
        The wrapped evaluation function.
    """
    def __init__(self, func, **kwargs):
        self.func_name = func.__name__
        self.function = partial(func, **kwargs)

        args = inspect.getfullargspec(self.function).args
        assert args == ["model", "test_loader", "metric_list"],\
            "A (partial) validation function must have the following kwargs: model, test_loader and\
            metric_list"

    def __call__(self, model, test_loader, metric):
        return self.function(model, test_loader, [metric])[metric]

    def __str__(self):
        kwdefargs = inspect.getfullargspec(self.function).kwonlydefaults
        return "ValidFunc(fun='%s', params=%s)" %(self.func_name, kwdefargs)

    def __repr__(self):
        return str(self)


def evaluate(model, test_loader, metric_list):
    """Evaluate the given model.
    The ``model`` evaluation is performed with all the provided metrics in ``metric_list``.
    The test set is loaded through the provided DataSampler.
    Parameters
    ----------
    model : the model to evaluate.
    test_loader : the DataSampler for the test set.
    metric_list : list of :obj:`str`
        The list of metrics to compute. Metrics are indicated by strings formed in the
        following way:
        ``metric_name`` @ ``k``
        where ``metric_name`` must correspond to one of the
        method names without the suffix '_at_k', and ``k`` is the corresponding parameter of
        the method and it must be an integer value. For example: ``ndcg@10`` is a valid metric
        name and it corresponds to the method
        :func:`ndcg_at_k` with ``k=10``.
    Returns
    -------
    :obj:`dict` of :obj:`numpy.array`
        Dictionary with the results for each metric in ``metric_list``. Keys are string
        representing the metric, while the value is an array with the value of the metric
        computed on the users.
    """
    results = {m:[] for m in metric_list}
    for batch_idx, batch_data in enumerate(test_loader):
        positive_pred, negative_pred = model.predict(batch_data)
        # we concatenate the positive prediction to the negative predictions
        # in each row of the final tensor we will have the positive prediction in the first column
        # and the 100 negative predictions in the last 100 columns
        positive_pred = positive_pred.view(positive_pred.size(0), 1)
        pred_scores = torch.cat((positive_pred, negative_pred), dim=1)
        # now, we need to construct the ground truth tensor
        ground_truth = np.zeros(pred_scores.size())
        ground_truth[:, 0] = 1  # the positive item is always in the first column of pred_scores, as we said before
        pred_scores = pred_scores.cpu().numpy()
        res = Metrics.compute(pred_scores, ground_truth, metric_list)
        for m in res:
            results[m].append(res[m])

    for m in results:
        results[m] = np.concatenate(results[m])
    return results