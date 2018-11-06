import logging
import torch
from torch.nn.modules.loss import BCELoss, MSELoss
import numpy as np


def negative_log_likelihood(log_p_pred, t_pred, t_true, t_checkpoints_pred, t_checkpoints_true):
    return -torch.mean(log_p_pred)


def score_mse(log_p_pred, t_pred, t_true, t_checkpoints_pred, t_checkpoints_true):
    t_pred
    return MSELoss()(t_pred, t_true)


def score_checkpoint_mse(log_p_pred, t_pred, t_true, t_checkpoints_pred, t_checkpoints_true):
    return MSELoss()(t_checkpoints_pred, t_checkpoints_true[1:])