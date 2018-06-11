import math
import numpy as np
import numpy.random as rng
from scipy.stats import beta, norm
from sklearn.utils import shuffle

import torch
from torch import Tensor
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim


from goldmine.inference.base import Inference
from goldmine.ml.models.maf import ConditionalMaskedAutoregressiveFlow


class MAFInference(Inference):

    """ Base class for inference methods. """

    def __init__(self):
        super().__init__()

        self.dtype = np.float32

        self.maf = None

    def fit(self, theta=None, x=None, y=None, r_xz=None, t_xz=None):
        raise NotImplementedError()

    def save(self, filename):
        raise NotImplementedError()

    def load(self, filename):
        raise NotImplementedError()

    def predict_density(self, x=None, theta=None):
        raise NotImplementedError()

    def predict_ratio(self, x=None, theta=None, theta1=None):
        raise NotImplementedError()

    def predict_score(self, x=None, theta=None):
        raise NotImplementedError()

