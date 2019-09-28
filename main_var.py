import argparse
import logging
import sys
import os
from configparser import ConfigParser

from torch import optim
import torch
from PIL import Image
import scipy.misc
import matplotlib

import numpy as np

from disvae.utils.modelIO import save_model, load_model, load_metadata
from disvae.models.vae import MODELS
from disvae.models.losses import LOSSES, RECON_DIST
from disvae.training_var import Trainer
from disvae.models.loss_var import get_loss_f
from disvae.models.var import VAR
from utils.datasets import get_dataloaders, get_img_size, DATASETS
from utils.helpers import (create_safe_directory, get_device, set_seed, get_n_param,
						   get_config_section, update_namespace_, FormatterNoDuplicate)
from utils.visualize import GifTraversalsTraining

CONFIG_FILE = "hyperparam.ini"
RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom', "debug", "best_celeba", "best_dsprites"]
EXPERIMENTS = ADDITIONAL_EXP + ["{}_{}".format(loss, data)
								for loss in LOSSES
								for data in DATASETS]


def parse_arguments(args_to_parse):
	"""Parse the command line arguments.

	Parameters
	----------
	args_to_parse: list of str
		Arguments to parse (splitted on whitespaces).
	"""
	default_config = get_config_section([CONFIG_FILE], "Custom")

	description = "PyTorch implementation and evaluation of disentangled Variational AutoEncoders and metrics."
	parser = argparse.ArgumentParser(description=description,
									 formatter_class=FormatterNoDuplicate)

	# General options
	general = parser.add_argument_group('General options')
	general.add_argument('name', type=str,
						 help="Name of the model for storing and loading purposes.")
	general.add_argument('-L', '--log-level', help="Logging levels.",
						 default=default_config['log_level'], choices=LOG_LEVELS)
	general.add_argument('--no-progress-bar', action='store_true',
						 default=default_config['no_progress_bar'],
						 help='Disables progress bar.')
	general.add_argument('--no-cuda', action='store_true',
						 default=default_config['no_cuda'],
						 help='Disables CUDA training, even when have one.')
	general.add_argument('-s', '--seed', type=int, default=default_config['seed'],
						 help='Random seed. Can be `None` for stochastic behavior.')

	# Learning options
	training = parser.add_argument_group('Training specific options')
	training.add_argument('--checkpoint-every',
						  type=int, default=default_config['checkpoint_every'],
						  help='Save a checkpoint of the trained model every n epoch.')
	training.add_argument('-d', '--dataset', help="Path to training data.",
						  default=default_config['dataset'], choices=DATASETS)
	training.add_argument('-x', '--experiment',
						  default=default_config['experiment'], choices=EXPERIMENTS,
						  help='Predefined experiments to run. If not `custom` this will overwrite some other arguments.')
	training.add_argument('-e', '--epochs', type=int,
						  default=default_config['epochs'],
						  help='Maximum number of epochs to run for.')
	training.add_argument('-b', '--batch-size', type=int,
						  default=default_config['batch_size'],
						  help='Batch size for training.')
	training.add_argument('--lr', type=float, default=default_config['lr'],
						  help='Learning rate.')

	# Model Options
	model = parser.add_argument_group('Model specfic options')
	model.add_argument('-m', '--model-type',
					   default=default_config['model'], choices=MODELS,
					   help='Type of encoder and decoder to use.')
	model.add_argument('-z', '--latent-dim', type=int,
					   default=default_config['latent_dim'],
					   help='Dimension of the latent variable.')
	model.add_argument('-l', '--loss',
					   default=default_config['loss'], choices=LOSSES,
					   help="Type of VAE loss function to use.")
	model.add_argument('-r', '--rec-dist', default=default_config['rec_dist'],
					   choices=RECON_DIST,
					   help="Form of the likelihood ot use for each pixel.")
	model.add_argument('-a', '--reg-anneal', type=float,
					   default=default_config['reg_anneal'],
					   help="Number of annealing steps where gradually adding the regularisation. What is annealed is specific to each loss.")

	# Loss Specific Options
	betaH = parser.add_argument_group('BetaH specific parameters')
	betaH.add_argument('--betaH-B', type=float,
					   default=default_config['betaH_B'],
					   help="Weight of the KL (beta in the paper).")

	betaB = parser.add_argument_group('BetaB specific parameters')
	betaB.add_argument('--betaB-initC', type=float,
					   default=default_config['betaB_initC'],
					   help="Starting annealed capacity.")
	betaB.add_argument('--betaB-finC', type=float,
					   default=default_config['betaB_finC'],
					   help="Final annealed capacity.")
	betaB.add_argument('--betaB-G', type=float,
					   default=default_config['betaB_G'],
					   help="Weight of the KL divergence term (gamma in the paper).")

	factor = parser.add_argument_group('factor VAE specific parameters')
	factor.add_argument('--factor-G', type=float,
						default=default_config['factor_G'],
						help="Weight of the TC term (gamma in the paper).")
	factor.add_argument('--lr-disc', type=float,
						default=default_config['lr_disc'],
						help='Learning rate of the discriminator.')

	btcvae = parser.add_argument_group('beta-tcvae specific parameters')
	btcvae.add_argument('--btcvae-A', type=float,
						default=default_config['btcvae_A'],
						help="Weight of the MI term (alpha in the paper).")
	btcvae.add_argument('--btcvae-G', type=float,
						default=default_config['btcvae_G'],
						help="Weight of the dim-wise KL term (gamma in the paper).")
	btcvae.add_argument('--btcvae-B', type=float,
						default=default_config['btcvae_B'],
						help="Weight of the TC term (beta in the paper).")

	# Learning options
	evaluation = parser.add_argument_group('Evaluation specific options')
	evaluation.add_argument('--is-eval-only', action='store_true',
							default=default_config['is_eval_only'],
							help='Whether to only evaluate using precomputed model `name`.')
	evaluation.add_argument('--is-metrics', action='store_true',
							default=default_config['is_metrics'],
							help="Whether to compute the disentangled metrcics. Currently only possible with `dsprites` as it is the only dataset with known true factors of variations.")
	evaluation.add_argument('--no-test', action='store_true',
							default=default_config['no_test'],
							help="Whether not to compute the test losses.`")
	evaluation.add_argument('--eval-batchsize', type=int,
							default=default_config['eval_batchsize'],
							help='Batch size for evaluation.')

	args = parser.parse_args(args_to_parse)
	if args.experiment != 'custom':
		if args.experiment not in ADDITIONAL_EXP:
			# update all common sections first
			model, dataset = args.experiment.split("_")
			common_data = get_config_section([CONFIG_FILE], "Common_{}".format(dataset))
			update_namespace_(args, common_data)
			common_model = get_config_section([CONFIG_FILE], "Common_{}".format(model))
			update_namespace_(args, common_model)

		try:
			experiments_config = get_config_section([CONFIG_FILE], args.experiment)
			update_namespace_(args, experiments_config)
		except KeyError as e:
			if args.experiment in ADDITIONAL_EXP:
				raise e  # only reraise if didn't use common section

	return args

def main(args):
	formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
									"%H:%M:%S")
	logger = logging.getLogger(__name__)
	logger.setLevel(args.log_level.upper())
	stream = logging.StreamHandler()
	stream.setLevel(args.log_level.upper())
	stream.setFormatter(formatter)
	logger.addHandler(stream)

	set_seed(args.seed)
	device = get_device(is_gpu = not args.no_cuda)
	exp_dir = os.path.join(RES_DIR, args.name)
	logger.info("Root directory for loading experiments: {}".format(exp_dir))

	# Prepare VAE model
	model_vae_dir = exp_dir
	model_vae = load_model(model_vae_dir, is_gpu = False)
	model_vae.eval()  # don't sample from latent: use mean

	# Prepare VAR model
	args.img_size = get_img_size(args.dataset)
	model_var_dir = os.path.join(exp_dir, 'var')
	create_safe_directory(model_var_dir, logger=logger)
	model_var = VAR(args.img_size)	
	logger.info('Num parameters in model: {}'.format(get_n_param(model_var)))

	# Prepare dataset
	meta_data = load_metadata(exp_dir)
	dataset = meta_data['dataset']
	train_loader = get_dataloaders(dataset, batch_size=args.batch_size, logger=logger)
	logger.info("Train {} with {} samples".format(args.dataset, len(train_loader.dataset)))

	# Train
	optimizer = optim.Adam(model_var.parameters(), lr = args.lr)
	model_var = model_var.to(device)
	loss_f = get_loss_f()
	gif_visualizer = GifTraversalsTraining(model_vae, dataset, model_vae_dir)
	trainer = Trainer(model_var, model_vae, optimizer, loss_f,
						device = device,
						logger = logger,
						save_dir = model_var_dir,
						is_progress_bar = not args.no_progress_bar,
						)
	trainer(train_loader, epochs = args.epochs, checkpoint_every = args.checkpoint_every,)

	
def test(args):
	formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
									"%H:%M:%S")
	logger = logging.getLogger(__name__)
	logger.setLevel(args.log_level.upper())
	stream = logging.StreamHandler()
	stream.setLevel(args.log_level.upper())
	stream.setFormatter(formatter)
	logger.addHandler(stream)

	device = get_device(is_gpu = not args.no_cuda)
	exp_dir = os.path.join(RES_DIR, args.name)
	logger.info("Root directory for loading experiments: {}".format(exp_dir))

	# Prepare VAE model
	model_vae = load_model(exp_dir, is_gpu = False)

	args.img_size = get_img_size(args.dataset)
	model_var_dir = os.path.join(exp_dir, 'var/model-400.pt')
	model_var = VAR(args.img_size)	
	model_var.load_state_dict(torch.load(model_var_dir), strict = False)
	img = np.random.random([3, 64, 64])
	x = torch.tensor(img, requires_grad = True).unsqueeze(0).float()
	eps = 1e-5
	x_ = torch.zeros(x.size())
	diff = torch.sum((x - x_)**2)
	epoch = 0
	while epoch <= args.epochs * 3 and diff >= eps:
		epoch += 1
		x_.copy_(x)
		y = model_var(x)	
		grad_x = torch.autograd.grad(y[0], x, retain_graph = True)[0].detach()
		x = x - 0.005 * grad_x	
		diff = torch.sum((x - x_)**2)
	
	x_recon, _, _ = model_vae(x)	
	store_img(x.detach(), x_recon.detach(), os.path.join(exp_dir, 'var'), name = 'syn')
	
	# Prepare dataset
	meta_data = load_metadata(exp_dir)
	dataset = meta_data['dataset']
	max_var = -float('inf')
	max_data_var = None
	max_loss = -float('inf')
	max_data_loss = None 
	test_loader = get_dataloaders(dataset, batch_size=1, logger=logger)
	loss_f = get_loss_f()
	for i, (data, _) in enumerate(test_loader):
		data_var = model_var(data)
		data_recon, _, _ = model_vae(data)
		data_loss = loss_f(data, data_recon, data_var, is_train = False, storer = None)

		if data_var > max_var:
			max_var = data_var
			max_data_var = data
			max_data_recon_var = data_recon
		if data_loss > max_loss:
			max_loss = data_loss
			max_data_loss = data
			max_data_recon_loss = data_recon

	store_img(max_data_var, max_data_recon_var, os.path.join(exp_dir, 'var'), name = 'var')
	store_img(max_data_loss, max_data_recon_loss, os.path.join(exp_dir, 'var'), name = 'loss') 	

def store_img(data, data_recon, model_var_dir, name = 'var'):
	img = data.squeeze(0).cpu().numpy()
	img = (np.clip(img, 0., 1.) * 255).astype(np.uint8).transpose(1, 2, 0)
	matplotlib.image.imsave(os.path.join(model_var_dir, name + '.png'), img)
	
	img_recon = data_recon.squeeze(0).detach().cpu().numpy()
	img_recon = (np.clip(img_recon, 0., 1.) * 255).astype(np.uint8).transpose(1, 2, 0)
	matplotlib.image.imsave(os.path.join(model_var_dir, name + '_recon.png'), img_recon)

if __name__ == '__main__':
	args = parse_arguments(sys.argv[1:])
	main(args) 
	test(args)

