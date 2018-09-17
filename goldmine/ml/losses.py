import torch
from torch.nn.modules.loss import MSELoss


def negative_log_likelihood(log_p_pred, r_pred, t_pred, y_true, r_true, t_true):
    return -torch.mean(log_p_pred)


def score_mse(log_p_pred, r_pred, t_pred, y_true, r_true, t_true):
    return MSELoss()(t_pred, t_true)
