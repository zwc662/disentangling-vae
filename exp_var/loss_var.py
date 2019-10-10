"""
Module containing all vae losses.
"""
import sys
sys.path.append('/export/u1/homes/weichao/Workspace/disentangling-vae/disvae/models/')

import abc
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

from discriminator import Discriminator
from disvae.utils.math import (log_density_gaussian, log_importance_weight_matrix,
							   matrix_log_density_gaussian)


# TO-DO: clean n_data and device
def get_loss_f(**kwargs_parse):
	"""Return the correct loss function given the argparse arguments."""
	return VarLoss(**kwargs_parse)

class VarLoss(abc.ABC):
	def __init__(self, record_loss_every=50, reg = 1.0):
		self.n_train_steps = 0
		self.record_loss_every = record_loss_every
		self.reg = reg

	def __call__(self, data, recon_data, var, is_train, storer, **kwargs):
		loss = torch.sum(torch.sum((data - recon_data)**2, dim = (2, 3)).unsqueeze(1)/(var**2)) + self.reg * torch.sum(torch.log(var**2))

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
