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

        t_checkpoints = []
        for z_initial, z_final in zip(z_checkpoints[:, -1], z_checkpoints[:, 1:]):
            logging.debug("  z step: %s -> %s", z_initial, z_final)
            t_checkpoints.append(
                self.step_model.forward(z_initial, z_final, theta).unsqueeze(1)
            )
            logging.debug("  that for step: %s", t_checkpoints[-1])
        t_checkpoints = torch.cat(t_checkpoints, 1)  # Shape (n_batch, n_checkpoints, n_params)
        logging.debug("  Concatenated that: %s", t_checkpoints)

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
