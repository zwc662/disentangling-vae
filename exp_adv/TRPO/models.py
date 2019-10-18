import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
	fan_in = layer.weight.data.size()[0]
	lim = 1. / np.sqrt(fan_in)
	return (-lim, lim)


class Policy(nn.Module):
	def __init__(self, state_size, action_size, seed = 1, fc1_units=400, fc2_units=300):
		super(Policy, self).__init__()
		r"""Encoder of the model proposed in [1].

		Parameters
		----------
		img_size : tuple of ints
			Size of images. E.g. (1, 32, 32) or (3, 64, 64).

		latent_dim : int
			Dimensionality of latent output.

		Model Architecture (transposed for decoder)
		------------
		- 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
		- 2 fully connected layers (each of 256 units)
		- Latent distribution:
			- 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

		References:
			[1] Burgess, Christopher P., et al. "Understanding disentangling in
			$\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
		"""

		# Layer parameters
		hid_channels = 32
		kernel_size = 4
		hidden_dim = 256
		self.action_size = action_size
		self.img_size = (1, 32, 32)
		# Shape required to start transpose convs
		self.reshape = (hid_channels, kernel_size, kernel_size)
		n_chan = self.img_size[0]

		# Convolutional layers
		cnn_kwargs = dict(stride=2, padding=1)
		self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
		self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
		self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

		# If input image is 64x64 do fourth convolution
		if self.img_size[1] == self.img_size[2] == 64:
			self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

		# Fully connected layers
		self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
		self.lin2 = nn.Linear(hidden_dim, hidden_dim)

		# Fully connected layers for mean and variance
		self.mu_logvar_gen = nn.Linear(hidden_dim, np.product(self.action_size) * 2)

	def forward(self, x):
		batch_size = x.size(0)

		x = x.view((batch_size, ) + self.img_size)
		# Convolutional layers with ReLu activations
		x = torch.relu(self.conv1(x))
		x = torch.relu(self.conv2(x))
		x = torch.relu(self.conv3(x))
		if self.img_size[1] == self.img_size[2] == 64:
			x = torch.relu(self.conv_64(x))

		# Fully connected layers with ReLu activations
		x = x.view((batch_size, -1))
		x = torch.relu(self.lin1(x))
		x = torch.relu(self.lin2(x))

		# Fully connected layer for log variance and mean
		# Log std-dev in paper (bear in mind)
		mu_logvar = self.mu_logvar_gen(x)
		mu, logvar = mu_logvar.view(-1, self.action_size, 2).unbind(-1)

		return mu, logvar, torch.exp(logvar)



class Value(nn.Module):
	def __init__(self, state_size, fcs1_units=400, fc2_units=300):
		super(Value, self).__init__()
		self.conv1 = nn.Conv2d(1, 20, 5, 1)
		self.conv2 = nn.Conv2d(20, 50, 5, 1)
		self.fc1 = nn.Linear(5*5*50, 500)
		self.fc2 = nn.Linear(500, 10)
		self.fc3 = nn.Linear(10, 1)

	def forward(self, state):
		x = state.view(-1, 1, 32, 32)
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, 5*5*50)
		
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		
		return x

