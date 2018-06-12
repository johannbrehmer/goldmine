import torch
from torch.nn.modules.loss import MSELoss


def negative_log_likelihood(model, y_true, r_true, t_true):
    return -torch.mean(model.log_likelihood)

def score_mse(model, y_true, r_true, t_true):
    return MSELoss()(model.score, t_true)
