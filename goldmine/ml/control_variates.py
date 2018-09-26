import torch
import numpy as np
import logging


def t_xz_cv_corrected(log_p_pred, log_r_pred, t_pred, y_true, r_true, t_true, coefficients, log_r_clip=10.):
    r_true = torch.clamp(r_true, np.exp(-log_r_clip), np.exp(log_r_clip))
    r_pred = torch.exp(torch.clamp(log_r_pred, -log_r_clip, log_r_clip))

    logging.info('Shapes: %s, %s, %s', t_true.shape, r_true.shape, coefficients_.shape)

    t_xz_corrected = t_true +  (1. / r_true - 1. / r_pred) * coefficients_

    return log_p_pred, log_r_pred, t_pred, y_true, r_true, t_xz_corrected
