import os
import sys
sys.path.append('/export/u1/homes/weichao/Workspace/disentangling-vae/')

from disvae.utils.modelIO import save_model, load_model, load_metadata
from disvae.models.losses import LOSSES, RECON_DIST
from disvae.models.vae import MODELS
from utils.datasets import get_dataloaders, get_img_size, DATASETS, get_background
from utils.mnist_classifier import Net as MNIST_Net


import six
import abc
from scipy.optimize import approx_fprime
from skimage.io import imread
import torch
import torchvision
import torchvision.transforms.functional as TF
import numpy as np

from datetime import datetime
from PIL import Image


BASE = "../"
CONFIG_FILE = BASE + "hyperparam.ini"
RES_DIR = BASE + "exp_adv/"

		


class VAE_Dynamics():
	def __init__(self, vae):
		""" Load VAE models """
		exp_dir = os.path.join(RES_DIR, vae) 
		self.model = load_model(exp_dir, is_gpu = False)
		self.device = 'cpu'
		self.model.eval()
		self.cur_data = None

	def _state_size(self):
		return self.model.img_size

	def _action_size(self):
		return self.model.latent_dim

	def _reset(self, data):
		""" Load data """
		self.cur_data = data

	def _step(self, action):
		with torch.no_grad():
			means, logvars = self.model.encoder(self.cur_data.unsqueeze(0).cpu())
			latent_var = means[0]
			latent_var += torch.tensor(action).cpu()
			nxt_data = self.model.decoder(latent_var.unsqueeze(0)).cpu()[0]
		return nxt_data


class Dynamics(VAE_Dynamics):
	def __init__(self, dataset, vae, cls, target):
		super(Dynamics, self).__init__(vae)
		self.dataset = dataset
		self.classifier = MNIST_Net()
		self.classifier.load_state_dict(\
			torch.load(open(os.path.join(RES_DIR, cls, 'model.pt'), 'rb')), strict = False)
		
		""" Target logits """
		self.target_logits = np.zeros([10])
		self.target_logits[target] = 1.0
	
		self.initial_data = None
		self.initial_state = None
		self.cur_state = None
		self.cur_step = None
	
	def action_size(self):
		return self._action_size()
	
	def state_size(self):
		return np.product(self._state_size())
	
	def reset(self):
		self.cur_step = 0
		timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S-%f")
		os.mkdir(os.path.join(RES_DIR, timestampStr))
		
		data_loader = get_dataloaders(self.dataset, batch_size=1)
		while True:
			(initial_data, targets) = next(iter(data_loader))
			self.initial_data = initial_data[0]
			initial_target = targets[0]
			if initial_target != np.argmax(self.target_logits):
				break
		self._reset(self.initial_data)	
		cur_image = self.render()
		cur_image.save(os.path.join(RES_DIR, timestampStr, '0.png'))
		
		self.initial_state = self.initial_data.flatten().cpu().numpy()
		self.cur_state = self.initial_state
		return self.initial_state
		
	def render(self):
		img = self.cur_data.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
		toImage = torchvision.transforms.ToPILImage()
		img = toImage(img)	
		#img.show()
		return img

	def reward(self):
		with torch.no_grad():
			pred_logits = self.classifier(self.cur_data.unsqueeze(0))[0].cpu().numpy()
		rew_logits = - np.linalg.norm(self.target_logits - pred_logits, ord = 2)	
		rew_diff = - np.linalg.norm(np.abs(self.cur_state - self.initial_state), ord = 2)

		if np.linalg.norm(self.target_logits - pred_logits, ord = 2) == 0:
			done = True	
		else:
			done = False

		return rew_logits + rew_diff, done

	def step(self, action):
		self.cur_step += 1
		nxt_data = self._step(action)
		self.cur_data = nxt_data
		reward, done = self.reward()
		self.cur_state = self.cur_data.flatten().cpu().numpy()
		
		if done:
			cur_image.save(os.path.join(RES_DIR, timestampStr, self.cur_step + '.png'))
	
		return self.cur_state, reward, done, None

