import torch.nn as nn


class FlowCheckpointScoreModel(nn.Module):

    def __init__(self, global_model, step_model):
        super().__init__()

        self.global_model = global_model
        self.step_model = step_model

    def forward(self, theta, x, z_checkpoints):
        # Make step model forward
        t_checkpoints = []
        for z_initial, z_final in zip(z_checkpoints[:, -1], z_checkpoints[:, 1:]):
            t_checkpoints.append(
                self.step_model.forward(z_initial, z_final, theta)
            )

        # Call global model forward
        u, log_likelihood, score = self.global_model(theta, x).log_likelihood_and_score

        # Return
        return u, log_likelihood, score, t_checkpoints

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.global_model.to(*args, **kwargs)
        self.step_model.to(*args, **kwargs)

        return self
