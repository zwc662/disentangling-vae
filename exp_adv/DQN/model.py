import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
	fan_in = layer.weight.data.size()[0]
	lim = 1. / np.sqrt(fan_in)
	return (-lim, lim)

class QNetwork_(nn.Module):
	"""QNetwork (Policy) Model."""

	def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
		"""Initialize parameters and build model.
		Params
		======
			state_size (int): Dimension of each state
			action_size (int): Dimension of each action
			seed (int): Random seed
			fc1_units (int): Number of nodes in first hidden layer
			fc2_units (int): Number of nodes in second hidden layer
		"""
		super(QNetwork, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.fc1 = nn.Linear(state_size, fc1_units)
		self.fc2 = nn.Linear(fc1_units, fc2_units)
		self.fc3 = nn.Linear(fc2_units, action_size)
		self.reset_parameters()

	def reset_parameters(self):
		self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
		self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
		self.fc3.weight.data.uniform_(-3e-3, 3e-3)

	def forward(self, state):
		"""Build an actor (policy) network that maps states -> actions."""
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		return F.tanh(self.fc3(x))

class QNetwork(nn.Module):
	def __init__(self, state_size, action_size, seed = 1., fc1_units=400, fc2_units=300):
		super(QNetwork, self).__init__()
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
		self.img_size = state_size
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
		self.mu_logvar_gen = nn.Linear(hidden_dim, self.action_size * 2)

	def forward(self, x):
		batch_size = x.size(0)

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

		return mu

class __QNetwork(nn.Module):
	def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
		super(QNetwork, self).__init__()
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
		self.img_size = state_size
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
		self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))

		# Convolutional layers
		cnn_kwargs = dict(stride=2, padding=1)
		# If input image is 64x64 do fourth convolution
		if self.img_size[1] == self.img_size[2] == 64:
			self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

		self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
		self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
		self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs)
		

	def forward(self, x):
		batch_size = x.size(0)

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
		x = torch.relu(self.lin3(x))
		x = x.view(batch_size, *self.reshape)

		# Convolutional layers with ReLu activations
		if self.img_size[1] == self.img_size[2] == 64:
			x = torch.relu(self.convT_64(x))
		x = torch.relu(self.convT1(x))
		x = torch.relu(self.convT2(x))
		# Sigmoid activation for final conv layer
		x = torch.sigmoid(self.convT3(x))
		x = x.view(batch_size, self.action_size)
		x = F.softmax(x)
		x = x.view(batch_size, self.action_size)
		return x
