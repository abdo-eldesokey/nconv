########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################


import os
import sys
import importlib
import json
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from dataloader.KittiDepthDataloader import KittiDepthDataloader
from modules.losses import ConfLossDecay, SmoothL1Loss, MSELoss

# Fix CUDNN error for non-contiguous inputs
import torch.backends.cudnn as cudnn
cudnn.enabled = True
cudnn.benchmark = True


# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-mode', action='store', dest='mode', help='"eval" or "train" mode')
parser.add_argument('-exp', action='store', dest='exp', help='Experiment name as in workspace directory')
parser.add_argument('-chkpt', action='store', dest='chkpt', default=-1, type=int, nargs='?', help='Checkpoint number to load')
parser.add_argument('-set',  action='store', dest='set', default='selval', type=str, nargs='?', help='Which set to evaluate on "val", "selval" or "test"')
args = parser.parse_args()


# Path to the workspace directory 
training_ws_path = 'workspace/'
exp = args.exp
exp_dir = os.path.join(training_ws_path, exp)

# Add the experiment's folder to python path
sys.path.append(exp_dir)

# Read parameters file 
with open(os.path.join(exp_dir, 'params.json'), 'r') as fp:
    params = json.load(fp)

# Use GPU or not
device = torch.device("cuda:"+str(params['gpu_id']) if torch.cuda.is_available() else "cpu")
    
# Dataloader for KittiDepth
dataloaders, dataset_sizes = KittiDepthDataloader(params)

# Import the network file       
f = importlib.import_module('network_'+exp)
model = f.CNN(pos_fn = params['enforce_pos_weights']).to(device)

# Import the trainer 
t = importlib.import_module('trainers.'+params['trainer'])

if args.mode == 'train':
    mode = 'train' # train    eval
    sets = ['train', 'selval'] #  train  selval   
elif args.mode == 'eval':
    mode = 'eval' # train    eval
    sets = [args.set] #  train  selval   
     

with torch.cuda.device(params['gpu_id']):

    # Objective function 
    objective = locals()[params['loss']]()


    # Optimize only parameters that requires_grad
    parameters = filter(lambda p: p.requires_grad, model.parameters())


    # The optimizer
    optimizer = getattr(optim, params['optimizer'])(parameters, lr=params['lr'], 
                                                    weight_decay=params['weight_decay'])
    

    # Decay LR by a factor of 0.1 every exp_dir7 epochs
    lr_decay = lr_scheduler.StepLR(optimizer, step_size=params['lr_decay_step'], gamma=params['lr_decay'])


    mytrainer = t.KittiDepthTrainer(model, params, optimizer, objective, lr_decay, dataloaders, dataset_sizes,
                 workspace_dir = exp_dir, sets=sets, use_load_checkpoint = args.chkpt  )

    if mode == 'train':
        # train the network
        net = mytrainer.train(params['num_epochs'])
    else:
        net = mytrainer.evaluate()

