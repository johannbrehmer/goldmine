import torch


def negative_log_likelihood_loss(model, y_true, y_pred):
    return -torch.mean(model.log_likelihood)