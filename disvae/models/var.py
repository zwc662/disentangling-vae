import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np


class VAR(nn.Module):
	def __init__(self, img_size):
		super(VAR, self).__init__()

		if list(img_size[1:]) not in [[32, 32], [64, 64]]:
			raise RuntimeError("{} sized images not supported. Only (None, 32, 32) and (None, 64, 64) supported. Build your own architecture or reshape images!".format(img_size))

		self.img_size = img_size

		hid_channels = 32
		kernel_size = 4
		hidden_dim = 256
		self.latent_dim = 1
		self.model_type = 'Burgess'
			
		output_size = np.product(self.img_size[0:3])
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
		"""
		self.lin1 = nn.Linear(np.product(self.reshape), np.product(self.img_size[0:3]))
		self.lin2 = nn.Linear(np.product(self.img_size[0:3]), np.product(self.img_size[0:3]))
		self.lin3 = nn.Linear(np.product(self.img_size[0:3]), 1)
		"""
		self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
		self.lin2 = nn.Linear(hidden_dim, hidden_dim)
		self.lin3 = nn.Linear(hidden_dim, 1)


	def forward(self, x):
		"""
		Forward pass of model.

		Parameters
		----------
		x : torch.Tensor
			Batch of data. Shape (batch_size, n_chan, height, width)
		"""
		batch_size = x.size(0)
		
		x = torch.relu(self.conv1(x))
		x = torch.relu(self.conv2(x))
		x = torch.relu(self.conv3(x))
		if self.img_size[1] == self.img_size[2] == 64:
			x = torch.relu(self.conv_64(x))
		
		x = x.view((batch_size, -1))
		x = torch.relu(self.lin1(x))
		x = torch.relu(self.lin2(x))
		x = self.lin3(x)
			
		return x
