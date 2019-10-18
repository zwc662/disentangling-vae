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


BASE = '/export/u1/homes/weichao/Workspace/disentangling-vae/'
ROOT_DIR = BASE + "exp_adv/"
RES_DIR = ROOT_DIR + "/results/"

		


class VAE_Dynamics():
	def __init__(self, vae):
		""" Load VAE models """
		exp_dir = os.path.join(ROOT_DIR, vae) 
		self.model = load_model(exp_dir, is_gpu = False)
		self.device = 'cpu'
		self.model.eval()
		self.cur_state = None

	def _state_size(self):
		return self.model.img_size

	def _action_size(self):
		return self.model.latent_dim

	def _reset(self, data):
		""" Load data """
		self.cur_state = data

	def _step(self, action):
		with torch.no_grad():
			means, logvars = self.model.encoder(self.cur_state.unsqueeze(0).cpu())
			latent_var = means[0]
			latent_var += torch.tensor(action.flatten()).cpu()
			nxt_state = self.model.decoder(latent_var.unsqueeze(0)).cpu()[0]
		return nxt_state

class Basic_Dynamics():
	def __init__(self, vae):
		self.cur_state = None

	def _state_size(self):
		return torch.empty(1, 32, 32).size()

	def _action_size(self):
		return torch.empty(1, 32, 32).size()

	def _reset(self, data):
		""" Load data """
		self.cur_state = data

	def _step(self, action):
		nxt_state = self.cur_state + torch.tensor(action)
		nxt_state = nxt_state.clamp(0., 1.)
		return nxt_state


class Dynamics(Basic_Dynamics):
	def __init__(self, dataset, vae, cls, target = 11):
		super(Dynamics, self).__init__(vae)
		self.dataset = dataset
		self.classifier = MNIST_Net()
		self.classifier.load_state_dict(\
			torch.load(open(os.path.join(ROOT_DIR, cls, 'model.pt'), 'rb')), strict = False)
		
		""" Target logits """
		if target <= 9:
			self.target_logits = np.zeros([10])
			self.target_logits[target] = 1.0
	
		self.initial_state = None
		self.initial_target = None
		self.initial_logits = None
		self.cur_state = None
		self.cur_step = None


		timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S-%f")
		self.result_dir = os.path.join(RES_DIR, timestampStr)
		os.mkdir(self.result_dir)
		
		data_loader = get_dataloaders(self.dataset, batch_size=1)
		while True:
			(initial_state, targets) = next(iter(data_loader))
			self.initial_state = initial_state[0]
			self.initial_logits = self.classifier(self.initial_state.unsqueeze(0)).detach().cpu().numpy()
			self.initial_target = targets[0]

			if self.initial_target != target:
				break
		self._reset(self.initial_state)	
		cur_image = self.render()
		cur_image.save(os.path.join(self.result_dir, '0.png'))
	
	def action_size(self):
		return self._action_size()
	
	def state_size(self):
		#return np.product(self._state_size())
		return self._state_size()
	
	def reset(self):
		self.cur_step = 0
		self._reset(self.initial_state)
		return self.cur_state.cpu().numpy()
		
	def render(self):
		img = self.cur_state.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
		toImage = torchvision.transforms.ToPILImage()
		img = toImage(img)	
		#img.show()
		return img

	def reward(self, action = None):
		with torch.no_grad():
			pred_logits = self.classifier(self.cur_state.unsqueeze(0)).cpu().numpy()
		#rew_logits = - np.linalg.norm(self.target_logits - pred_logits, ord = 2)	
		rew_action = - np.linalg.norm(action.flatten())
		if np.argmax(self.target_logits) == np.argmax(pred_logits):
			rew_logits = 100.0
			done = True	
		else:
			rew_logits = 0.0
			done = False

		return rew_logits + rew_action, done

	def step(self, action):
		self.cur_step += 1
		nxt_state = self._step(action)
		self.cur_state = nxt_state
		reward, done = self.reward(action)
		cur_image = self.render()
		if done:
			cur_image.save(os.path.join(self.result_dir, 'Done_' + self.cur_step + '.png'))
	
		return self.cur_state.cpu().numpy(), reward, done, None

	def close(self):
		self.cur_state = None

