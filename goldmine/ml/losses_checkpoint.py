import logging
import torch
from torch.nn.modules.loss import BCELoss, MSELoss
import numpy as np


def score_checkpoint_mse(log_p_pred, t_x_pred, t_xv_pred, t_xv_checkpoints_pred, t_xz_checkpoints):
    return MSELoss()(t_xv_checkpoints_pred, t_xz_checkpoints[:, 1:])


def score_mse(log_p_pred, t_x_pred, t_xv_pred, t_xv_checkpoints_pred, t_xz_checkpoints):
    return MSELoss()(t_x_pred, t_xv_pred)


def negative_log_likelihood(log_p_pred, t_x_pred, t_xv_pred, t_xv_checkpoints_pred, t_xz_checkpoints):
    return -torch.mean(log_p_pred)
