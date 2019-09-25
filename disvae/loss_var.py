"""
Module containing all vae losses.
"""
import abc
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

from .discriminator import Discriminator
from disvae.utils.math import (log_density_gaussian, log_importance_weight_matrix,
                               matrix_log_density_gaussian)


# TO-DO: clean n_data and device
def get_loss_f(loss_name, **kwargs_parse):
    """Return the correct loss function given the argparse arguments."""
	return VarLoss()

class VarLoss(abc.ABC):
    def __init__(self, record_loss_every=50):
        self.n_train_steps = 0
        self.record_loss_every = record_loss_every

    def __call__(self, data, recon_data, var, is_train, storer, **kwargs):
		


		if storer is not None:
			storer['loss'].append(loss.item())

		return loss

    def _pre_call(self, is_train, storer):
        if is_train:
            self.n_train_steps += 1

        if not is_train or self.n_train_steps % self.record_loss_every == 1:
            storer = storer
        else:
            storer = None
        return storer
