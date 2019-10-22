import os
import sys
sys.path.append('/export/u1/homes/weichao/Workspace/disentangling-vae/')
sys.path.append('../../')
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
BASE = '../../'
ROOT_DIR = BASE + "exp_adv/"
RES_DIR = ROOT_DIR + "/A3C/results/"

		


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
			means, logvars = self.model.encoder(self.cur_state.unsqueeze(0).cpu())
			latent_var = means[0]
			latent_var += torch.tensor(action).float().flatten().cpu()
			nxt_state = self.model.decoder(latent_var.unsqueeze(0)).cpu()[0]
		return nxt_state

class Basic_Dynamics():
	def __init__(self):
		self.cur_state = None

	def _state_size(self):
		return torch.empty(1, 32, 32).size()

	def _action_size(self):
		return np.product((1, 32, 32))

	def _reset(self, data):
		""" Load data """
		self.cur_state = data

	def _step(self, action):
		nxt_state = self.cur_state.float() + torch.tensor(action).float()
		nxt_state = nxt_state.clamp(0., 1.)
		return nxt_state


class Dynamics(VAE_Dynamics):
	def __init__(self, dataset, cls, target = 9):
		super(Dynamics, self).__init__()
		self.dataset = dataset
		self.classifier = MNIST_Net()
		self.classifier.load_state_dict(\
			torch.load(open(os.path.join(ROOT_DIR, cls, 'model.pt'), 'rb'), map_location=torch.device('cpu')),\
			strict = False)
		
		""" Target logits """
		self.target_logits = np.zeros([10])
		self.initial_state = None
		self.initial_target = None
		#self.initial_logits = None
		self.target = None
		self.cur_state = None
		self.cur_step = None


		timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S-%f")
		self.result_dir = os.path.join(RES_DIR, timestampStr)
		os.mkdir(self.result_dir)
		
		data_loader = get_dataloaders(self.dataset, batch_size=1)
		while True:
			(initial_state, initial_target) = next(iter(data_loader))
			self.initial_state = initial_state[0].float()
			self.initial_logits = self.classifier(initial_state)[0].detach().cpu().numpy()
			self.initial_target = initial_target[0]
	
			self.initial_logits_dummy = np.zeros([10])
			self.initial_logits_dummy[self.initial_target] = 1.0

			if target is None or target > 9 or self.initial_target == target:
				continue
				self.target_logits[self.initial_target] = 1.0
				self.target_logits = 1./9. * (1.0 - self.target_logits)
				self.target = np.delete(np.arange(10), self.initial_target, 0)
			elif target is not None:
				self.target_logits[target] = 1.0
				self.target = target
				break
				
				
				
		self._reset(self.initial_state)	
		cur_image = self.render()
		cur_image.save(os.path.join(self.result_dir, '0.png'))
	
	def action_size(self):
		return self._action_size()


	@property	
	def observation_space(self):
		return np.zeros([1, 32, 32])
	
	@property
	def action_space(self):
		return 10 
	
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
		img = cur_state.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
		toImage = torchvision.transforms.ToPILImage()
		img = toImage(img)	
		return img

	def reward(self, action = None):
		rew_logits = 0.
		rew_action = 0
		with torch.no_grad():
			cur_logits = np.exp(self.classifier(self.cur_state.unsqueeze(0))[0].cpu().numpy())
			
		if self.target_logits is not None:
			#rew_logits = - np.linalg.norm(self.target_logits - np.exp(pred_logits), ord = 2)
			rew_logits = np.sum(np.log(cur_logits) * (self.target_logits - self.initial_logits_dummy) + \
							0)#np.log(1 - cur_logits) * (1 - self.target_logits))
			#rew_logits = np.exp(1e1 * cur_logits[self.target]) - 1.
			if np.argmax(self.target_logits) == np.argmax(cur_logits):
				rew_logits += 100.0
				done = True	
			else:
				rew_logits += 0.0
				done = False
		else:
			rew_logits = np.linalg.norm(self.initial_logits - cur_logits, ord = 2)
			if np.argmax(self.initial_logits) != np.argmax(cur_logits):
				rew_logits += 100.0
				done = True	
			else:
				rew_logits += 0.0
				done = False

		rew_action = - 0. #np.linalg.norm(action.flatten())
		return rew_logits + rew_action, done

	def step(self, action):
		
		#action = action + np.zeros([10])
		#action = 1./255. * action/np.linalg.norm(action, ord = 2)
		self.cur_step += 1

		#action = np.reshape(action, self.cur_state.shape)
		action = action.flatten()
		action = 0.01 * action/np.linalg.norm(action, ord = 2)
		nxt_state = self._step(action)
		self.cur_state = nxt_state

		reward, done = self.reward(action)
		if done:
			cur_image = self.render()
			cur_image.save(os.path.join(self.result_dir, 'Done_' + str(self.cur_step) + '.png'))
	
		return self.cur_state.cpu().numpy(), reward, done, None
	
	def seed(self, seed):
		pass
