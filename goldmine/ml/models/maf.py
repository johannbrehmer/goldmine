import numpy as np
import numpy.random as rng
import logging

import torch
import torch.nn as nn
from torch import tensor
from torch.autograd import grad

from .made import GaussianMADE, ConditionalGaussianMADE
from .batch_norm import BatchNorm


class MaskedAutoregressiveFlow(nn.Module):
    """
    Implements a Masked Autoregressive Flow, which is a stack of mades such that the random numbers which drive made i
    are generated by made i-1. The first made is driven by standard gaussian noise. In the current implementation, all
    mades are of the same type. If there is only one made in the stack, then it's equivalent to a single made.
    """

    def __init__(self, n_inputs, n_hiddens, n_mades, batch_norm=True,
                 input_order='sequential', mode='sequential', alpha=0.1):

        super(MaskedAutoregressiveFlow, self).__init__()

        # save input arguments
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.n_mades = n_mades
        self.batch_norm = batch_norm
        self.mode = mode
        self.alpha = alpha

        # log p
        self.log_likelihood = None

        # Build MADEs
        self.mades = nn.ModuleList()
        for i in range(n_mades):
            made = GaussianMADE(n_inputs, n_hiddens, input_order, mode)
            self.mades.append(made)
            if input_order != 'random':
                input_order = made.input_order[::-1]

        # Batch normalizatino
        self.bns = None
        if self.batch_norm:
            self.bns = nn.ModuleList()
            for i in range(n_mades):
                bn = BatchNorm(n_inputs, alpha=self.alpha)
                self.bns.append(bn)

    def forward(self, x, fix_batch_norm=False):

        """ Transforms x into u = f^-1(x) """

        # Change batch norm means only while training
        if not self.training:
            fix_batch_norm = True

        logdet_dudx = 0.0
        u = x

        for i, made in enumerate(self.mades):
            # inverse autoregressive transform
            u = made(u)
            logdet_dudx += 0.5 * torch.sum(made.logp, dim=1)

            # batch normalization
            if self.batch_norm:
                bn = self.bns[i]
                u = bn(u, fixed_params=fix_batch_norm)
                logdet_dudx -= 0.5 * torch.sum(torch.log(bn.var))

        # log likelihood
        const = float(-0.5 * self.n_inputs * np.log(2 * np.pi))
        self.log_likelihood = const - 0.5 * torch.sum(u ** 2, dim=1) + logdet_dudx

        return u

    def log_p(self, x):

        """ Calculates log p(x) """

        u = self.forward(x)

        return self.log_likelihood

    def gen(self, n_samples=1, u=None):
        """
        Generate samples, by propagating random numbers through each made.
        :param n_samples: number of samples
        :param u: random numbers to use in generating samples; if None, new random numbers are drawn
        :return: samples
        """

        x = tensor(rng.randn(n_samples, self.n_inputs)) if u is None else u

        if self.batch_norm:
            mades = [made for made in self.mades]
            bns = [bn for bn in self.bns]

            for i, (made, bn) in enumerate(zip(mades[::-1], bns[::-1])):
                x = bn.inverse(x)
                x = made.gen(n_samples, x)
        else:
            mades = [made for made in self.mades]
            for made in mades[::-1]:
                x = made.gen(n_samples, x)

        return x


class ConditionalMaskedAutoregressiveFlow(nn.Module):
    """
    Implements a Masked Autoregressive Flow, which is a stack of mades such that the random numbers which drive made i
    are generated by made i-1. The first made is driven by standard gaussian noise. In the current implementation, all
    mades are of the same type. If there is only one made in the stack, then it's equivalent to a single made.
    """

    def __init__(self, n_conditionals, n_inputs, n_hiddens, n_mades, batch_norm=True,
                 input_order='sequential', mode='sequential', alpha=0.1):

        super(ConditionalMaskedAutoregressiveFlow, self).__init__()

        # save input arguments
        self.n_conditionals = n_conditionals
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.n_mades = n_mades
        self.batch_norm = batch_norm
        self.mode = mode
        self.alpha = alpha

        # log p and score
        self.log_likelihood = None
        self.score = None

        # Build MADEs
        self.mades = nn.ModuleList()
        for i in range(n_mades):
            made = ConditionalGaussianMADE(n_conditionals, n_inputs, n_hiddens, input_order, mode)
            self.mades.append(made)
            if input_order != 'random':
                input_order = made.input_order[::-1]

        # Batch normalizatino
        self.bns = None
        if self.batch_norm:
            self.bns = nn.ModuleList()
            for i in range(n_mades):
                bn = BatchNorm(n_inputs, alpha=self.alpha)
                self.bns.append(bn)

    def forward(self, theta, x, fix_batch_norm=None, track_score=True):

        """ Transforms x into u = f^-1(x) """

        # Change batch norm means only while training
        if fix_batch_norm is None:
            fix_batch_norm = not self.training

        # Track gradient wrt theta
        if track_score and not theta.requires_grad:  # Can this happen?
            theta.requires_grad = True

        logdet_dudx = 0.0
        u = x

        for i, made in enumerate(self.mades):
            # inverse autoregressive transform
            u = made(theta, u)
            logdet_dudx += 0.5 * torch.sum(made.logp, dim=1)

            # batch normalization
            if self.batch_norm:
                bn = self.bns[i]
                u = bn(u, fixed_params=fix_batch_norm)
                logdet_dudx -= 0.5 * torch.sum(torch.log(bn.var))

        # log likelihood
        const = float(-0.5 * self.n_inputs * np.log(2 * np.pi))
        self.log_likelihood = const - 0.5 * torch.sum(u ** 2, dim=1) + logdet_dudx

        # Score
        if track_score:
            self.score = grad(self.log_likelihood, theta,
                              grad_outputs=torch.ones_like(self.log_likelihood.data),
                              only_inputs=True, create_graph=True)[0]

        return u

    def predict_log_likelihood(self, theta, x):

        """ Calculates log p(x) """

        _ = self.forward(theta, x)

        return self.log_likelihood

    def predict_score(self, theta, x):

        """ Calculates log p(x) """

        _ = self.forward(theta, x)

        return self.score

    def generate_samples(self, theta, n_samples=1, u=None):
        """
        Generate samples, by propagating random numbers through each made.
        :param n_samples: number of samples
        :param u: random numbers to use in generating samples; if None, new random numbers are drawn
        :return: samples
        """

        x = tensor(rng.randn(n_samples, self.n_inputs)) if u is None else u

        if self.batch_norm:
            mades = [made for made in self.mades]
            bns = [bn for bn in self.bns]

            for i, (made, bn) in enumerate(zip(mades[::-1], bns[::-1])):
                x = bn.inverse(x)
                x = made.generate_samples(theta, n_samples, x)
        else:
            mades = [made for made in self.mades]
            for made in mades[::-1]:
                x = made.generate_samples(theta, n_samples, x)

        return x
