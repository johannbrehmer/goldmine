import logging
import torch.nn as nn


class FlowCheckpointScoreModel(nn.Module):

    def __init__(self, global_model, step_model):
        super().__init__()

        self.global_model = global_model
        self.step_model = step_model

    def forward_checkpoints(self, theta, z_checkpoints):
        n_batch, n_steps, n_latent = z_checkpoints.size()
        n_parameters = theta.size()[-1]

        z_initial = z_checkpoints[:, :-1, :].contiguous().view(-1, n_latent)
        z_final = z_checkpoints[:, 1:, :].contiguous().view(-1, n_latent)

        theta_step = theta.clone()
        theta_step.unsqueeze(1)  # (n_batch, 1, n_parameters)
        theta_step = theta_step.repeat(1, n_steps - 1, 1)
        theta_step = theta_step.contiguous().view(-1, n_parameters)

        t_checkpoints = self.step_model.forward(z_initial, z_final, theta_step)
        t_checkpoints = t_checkpoints.view(n_batch, n_steps - 1, n_parameters).contiguous()

        return t_checkpoints

    def forward_global(self, theta, x):
        u, log_likelihood, score = self.global_model.log_likelihood_and_score(theta, x)

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
