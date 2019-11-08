import argparse
import logging

import os
from configparser import ConfigParser

from torch import optim
import torch
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid

from PIL import Image, ImageDraw
import scipy.misc
import matplotlib

import numpy as np

from var import VAR
from training_var import Trainer
from loss_var import get_loss_f

import sys
sys.path.append('/export/u1/homes/weichao/Workspace/disentangling-vae/')

from disvae.utils.modelIO import save_model, load_model, load_metadata
from disvae.models.losses import LOSSES, RECON_DIST
from disvae.models.vae import MODELS
from utils.datasets import get_dataloaders, get_img_size, DATASETS, get_background
from utils.helpers import (create_safe_directory, get_device, set_seed, get_n_param,
						   get_config_section, update_namespace_, FormatterNoDuplicate)
from utils.visualize import GifTraversalsTraining
from utils.viz_helpers import add_labels
from utils.mnist_classifier import Net as MNIST_Net

BASE = "../"
CONFIG_FILE = BASE + "hyperparam.ini"
RES_DIR = BASE + "results"
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
	general.add_argument('-c', '--classifier', type = str,
						 help='Name of the classifeir')
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
	training.add_argument('-g', '--gamma', type = float, default = 1., help = 'gamma for var')

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





def train_calib(args):
	formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
	
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

	# Prepare Classifier
	model_cls = MNIST_Net() 
	model_cls.load_state_dict(torch.load(open('./utils/mnist_cnn.pt', 'rb')), strict = False)
	model_cls.eval() 

	# Prepare Calibration model
	model_calib = CALIB_Net()
	
	# Prepare dataset
	meta_data = load_metadata(exp_dir)
	dataset = meta_data['dataset']
	train_loader = get_dataloaders(dataset, batch_size=args.batch_size, logger=logger)
	logger.info("Train {} with {} samples".format(args.dataset, len(train_loader.dataset)))

	# Train
	optimizer = optim.Adam(model_calib.parameters(), lr = args.lr)
	model_calib = model_calib.to(device)
	loss_f = get_loss_f(reg = args.gamma)
	gif_visualizer = GifTraversalsTraining(model_vae, dataset, model_vae_dir)
	trainer = Trainer(model_var, model_vae, optimizer, loss_f,
						device = device,
						logger = logger,
						save_dir = model_var_dir,
						is_progress_bar = not args.no_progress_bar,
						)
	args.epoch = 50
	args.checkpoint_every = 10
	print("Total number of epochs: {}".format(args.epochs))
	trainer(train_loader, epochs = args.epochs, checkpoint_every = args.checkpoint_every) 

	
	
	

	
if __name__ == '__main__':
