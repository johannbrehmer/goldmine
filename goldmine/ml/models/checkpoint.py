import logging
import torch.nn as nn
import torch


class FlowCheckpointScoreModel(nn.Module):

    def __init__(self, global_model, step_model):
        super().__init__()

        self.global_model = global_model
        self.step_model = step_model

    def forward_checkpoints(self, theta, z_checkpoints):
        logging.debug("Evaluating forward_checkpoints():")

        logging.debug("  theta = %s", theta)
        logging.debug("  z_checkpoints = %s", z_checkpoints)

        n_batch, n_steps, n_latent = z_checkpoints.size()

        z_initial = z_checkpoints[:,:-1,:].view(-1,n_latent)
        z_final = z_checkpoints[:,1:,:].view(-1,n_latent)

        logging.debug("  z_initial = %s", z_initial)
        logging.debug("  z_final = %s", z_final)

        t_checkpoints = self.step_model.forward(z_initial, z_final, theta)
        t_checkpoints = t_checkpoints.view(n_batch, n_steps, n_latent)
        logging.debug("  that: %s", t_checkpoints)

        return t_checkpoints

    def forward_global(self, theta, x):
        u, log_likelihood, score = self.global_model(theta, x).log_likelihood_and_score

        return u, log_likelihood, score

    def forward(self, theta, x, z_checkpoints):
        t_checkpoints = self.forward_checkpoints(theta, z_checkpoints)
        u, log_likelihood, score = self.forward_global(theta, x)

        return u, log_likelihood, score, t_checkpoints

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.global_model.to(*args, **kwargs)
        self.step_model.to(*args, **kwargs)

        return self
