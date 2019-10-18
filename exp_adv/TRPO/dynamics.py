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
RES_DIR = ROOT_DIR + "/TRPO/results/"

		


class VAE_Dynamics():
	def __init__(self, vae):
		""" Load VAE models """
		exp_dir = os.path.join(ROOT_DIR, vae) 
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
			latent_var += torch.tensor(action.flatten()).cpu()
			nxt_data = self.model.decoder(latent_var.unsqueeze(0)).cpu()[0]
		return nxt_data

class Basic_Dynamics():
	def __init__(self, vae):
		self.cur_data = None

	def _state_size(self):
		return torch.empty(1, 32, 32).size()

	def _action_size(self):
		return torch.empty(1, 32, 32).size()

	def _reset(self, data):
		""" Load data """
		self.cur_data = data

	def _step(self, action):
		nxt_data = self.cur_data.float() + torch.tensor(action).float()
		nxt_data = nxt_data.clamp(0., 1.)
		return nxt_data


class Dynamics(Basic_Dynamics):
	def __init__(self, dataset, vae, cls, target = 10):
		super(Dynamics, self).__init__(vae)
		self.dataset = dataset
		self.classifier = MNIST_Net().float()
		self.classifier.load_state_dict(\
			torch.load(open(os.path.join(ROOT_DIR, cls, 'model.pt'), 'rb')), strict = False)
		
		""" Target logits """
		self.target_logits = np.zeros([10])
		self.initial_data = None
		self.initial_state = None
		self.initial_target = None
		#self.initial_logits = None
		self.cur_data = None
		self.initial_state = None
		self.cur_step = None


		timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S-%f")
		self.result_dir = os.path.join(RES_DIR, timestampStr)
		os.mkdir(self.result_dir)
		
		data_loader = get_dataloaders(self.dataset, batch_size=1)
		if True:
			(initial_data, initial_target) = next(iter(data_loader))
			self.initial_data = initial_data.squeeze(0)
			self.initial_state = self.initial_data.flatten()
			self.initial_logits = self.classifier(initial_data).detach().cpu().numpy()
			self.initial_target = initial_target.squeeze(0)

			if target is not None:
				if target > 9 or self.initial_target == target:
					self.target_logits[self.initial_target] = 1.0
					self.target_logits = 1./9. * (1.0 - self.target_logits)
			else:
				self.target_logits[target] = 1.0
				
				
		self._reset(self.initial_data)	
		self.cur_state = self.cur_data.flatten()
		cur_image = self.render()
		cur_image.save(os.path.join(self.result_dir, '0.png'))
	
	def action_size(self):
		return self._action_size()
	
	@property
	def action_space(self):
		return [1 * 32 * 32]
	
	def state_size(self):
		return self._state_size()
	
	@property
	def state_space(self):
		return [1 * 32 * 32]
	
	def reset(self):
		self.cur_step = 0
		self._reset(self.initial_data)
		self.cur_state = self.cur_data.flatten().cpu().numpy()
		return self.cur_state
		
	def render(self):
		cur_data = torch.zeros_like(self.cur_data)
		cur_data.copy_(self.cur_data)
		img = cur_data.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
		toImage = torchvision.transforms.ToPILImage()
		img = toImage(img)	
		return img

	def reward(self, action = None):
		rew_logits = 0.
		rew_action = 0
		with torch.no_grad():
			pred_logits = self.classifier(self.cur_data.unsqueeze(0)).squeeze(0).cpu().numpy()
			
		if self.target_logits is not None:
			#rew_logits = - np.linalg.norm(self.target_logits - np.exp(pred_logits), ord = 2)
			rew_logits = np.sum(pred_logits * self.target_logits + \
							np.log(1 - np.exp(pred_logits)) * (1 - self.target_logits))
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

		rew_action = - np.linalg.norm(action.flatten())
		return rew_logits + rew_action, done

	def step(self, action):
		self.cur_step += 1

		cur_action = action.reshape(self.cur_data.size())
		self.cur_data = self._step(cur_action)
		self.cur_state = self.cur_data.flatten().cpu().numpy()

		reward, done = self.reward(cur_action)
		if done:
			cur_image = self.render()
			cur_image.save(os.path.join(self.result_dir, 'Done_' + str(self.cur_step) + '.png'))
	
		return self.cur_state, reward, done, None

