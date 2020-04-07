"""
Main run script to execute experiments and analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import transformers as hug
import os
from tqdm import tqdm, trange
import argparse

# =============== Self Defined ===============
import io            # module for handling import/export of data
import utils         # utility functions for training and and evaluating
import model         # module to define model architecture
import meta_learning # module for meta-learning (OML)
import cont_learning # module for continual learning
import analyze       # module for analyzing results

args = argparse.ArgumentParser()
args.add_argument('--data', type=str, default='/data',
                  help='directory storing all data')
args.add_argument('--save_dir', type=str, default='/results',
                    help='directory to save results')
args.add_argument('--meta_epochs', type=int, default=100,
                  help='number of epochs for meta-learning')
args.add_argument('--meta_epochs', type=int, default=100,
                  help='number of epochs for meta-learning')
args.add_argument('--fine_tune_epochs', type=int, default=100000,
                  help='number of epochs for fine-tuning')
args.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
args.add_argument('--batch_size', type=int, default=20,
                    help='batch size')
#args.add_argument('--dropout', type=float, default=0.2,
#                    help='dropout applied to layers (0 = no dropout)')
args.add_argument('--seed', type=int, default=1111,
                    help='random seed')
args.add_argument('--model', type=str, default='BERT',
                    help='name of RLN network. default is BERT')

# Set devise to CPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device is {}".format(device))

# create io object to import data

# define models

# continual learning for baseline BERT

# meta-learning to get Meta-BERT

# continual learning for Meta-BERT

# analyze results from continual learning steps

# save results with io


"""
model = model.model(...)
model_meta = model.model(...)
optimizer = opt.Adam(model.parameters, arguments)
optimizer_rln = opt.Adam(model_meta.representation.parameters, arguments)
optimizer_pln = opt.Adam(model_meta.classification.parameters, arguments)
loss = nn.NLLLoss(arguments)
"""