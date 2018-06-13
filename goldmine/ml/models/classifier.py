import numpy as np
import numpy.random as rng
import logging

import torch
import torch.nn as nn
from torch import tensor
from torch.autograd import grad

# TODO

class Classifier(nn.Module):

    def __init__(self, n_hidden_layers, n_hidden_layer_size):
        super().__init__()

        # save input arguments
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_layer_size = n_hidden_layer_size

    def forward(self, x):
        pass
