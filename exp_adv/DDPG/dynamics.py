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
RES_DIR = ROOT_DIR + "/DDPG/results/"

		


class VAE_Dynamics():
	def __init__(self, vae = 'VAE_mnist'):
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
			means, logvars = self.model.encoder(self.cur_state)
			latent_var = means.cpu()[0]
			latent_var += torch.tensor(action.flatten()).cpu()
			nxt_state = self.model.decoder(latent_var.unsqueeze(0)).cpu()
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


class Dynamics(VAE_Dynamics):
	def __init__(self, dataset, vae, cls, target = 10):
		super(Dynamics, self).__init__(vae)
		self.dataset = dataset
		self.classifier = MNIST_Net()
		self.classifier.load_state_dict(\
			torch.load(open(os.path.join(ROOT_DIR, cls, 'model.pt'), 'rb')), strict = False)
		
		""" Target logits """
		self.target_logits = np.zeros([10])
		self.initial_state = None
		self.initial_target = None
		#self.initial_logits = None
		self.cur_state = None
		self.cur_step = None
		self.target = target


		timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S-%f")
		self.result_dir = os.path.join(RES_DIR, timestampStr)
		os.mkdir(self.result_dir)
		
		data_loader = get_dataloaders(self.dataset, batch_size=1)
		if True:
			(initial_state, initial_target) = next(iter(data_loader))
			self.initial_state = initial_state
			self.initial_logits = self.classifier(self.initial_state).detach().cpu().numpy()
			self.initial_target = initial_target[0]

			self.initial_logits_dummy = np.zeros([10])
			self.initial_logits_dummy[self.initial_target] = 1.0

			if target is not None:
				if target > 9 or self.initial_target == target:
					self.target_logits[self.initial_target] = 1.0
					self.target_logits = 1./9. * (1.0 - self.target_logits)
			else:
				self.target_logits[target] = 1.0
				
				
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
		cur_state = torch.zeros_like(self.cur_state)
		cur_state.copy_(self.cur_state)
		img = cur_state.squeeze(0).mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
		toImage = torchvision.transforms.ToPILImage()
		img = toImage(img)	
		return img

	def reward(self, action = None):
		rew_logits = 0.
		rew_action = 0
		with torch.no_grad():
			pred_logits = np.exp(self.classifier(self.cur_state)[0].cpu().numpy())
			
		if self.target_logits is not None:
			#rew_logits = - np.linalg.norm(self.target_logits - np.exp(pred_logits), ord = 2)
			rew_logits = np.sum(np.log(pred_logits) * (self.target_logits - self.initial_logits_dummy))\
							#np.log(1 - np.exp(pred_logits)) * (1 - self.target_logits))
			if np.argmax(self.target_logits) == np.argmax(pred_logits):
				rew_logits += 100.0
				done = True	
			else:
				rew_logits += 0.0
				done = False
		else:
			rew_logits = np.linalg.norm(self.initial_logits - pred_logits, ord = 2)
			if np.argmax(self.initial_logits) != np.argmax(pred_logits):
				rew_logits += 100.0
				done = True	
			else:
				rew_logits += 0.0
				done = False

		rew_action = - 0.0 * np.linalg.norm(action.flatten())
		return rew_logits + rew_action, done

	def step(self, action):
		self.cur_step += 1

		'''
		idx = np.argmax(np.abs(action))
		sign = np.sign(np.max(action))
		action *= 0.0
		action[0, 0, int(idx/32), idx%32] += sign * 1./255.
		'''
		action = 1e-2 * action/np.linalg.norm(action, ord = 2)
		nxt_state = self._step(action)
		self.cur_state = nxt_state

		reward, done = self.reward(action)
		if done:
			cur_image = self.render()
			cur_image.save(os.path.join(self.result_dir, 'Done_' + str(self.cur_step) + '.png'))
	
		return self.cur_state.cpu().numpy(), reward, done, None

