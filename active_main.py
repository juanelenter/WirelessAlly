
import numpy as np
import sys
import gzip
import gc
import os
import argparse
from dataset import get_dataset, get_handler
import models
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
import torch
import pdb
from strategy import Strategy
import random
from ally import ALLYSampling


def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(opts.seed)

# parameters
NUM_INIT_LB = opts.nStart
NUM_QUERY = opts.nQuery
NUM_ROUND = int((opts.nEnd - NUM_INIT_LB)/ opts.nQuery)

X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME, opts.path)
opts.dim = np.shape(X_tr)[1:]
handler = get_handler(opts.data)

# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)

# generate initial labeled pool
idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_tmp = np.arange(n_pool)
np.random.shuffle(idxs_tmp)
idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

ally = ALLYSampling(X_tr, Y_tr, idxs_lb, net, handler, args, opts.cluster, opts.epsilon, opts.nPrimal, opts.lambdaTestSize)

print(DATA_NAME, flush=True)
print(type(ally).__name__, flush=True)

# Initialize active learning strategy
ally.train()
P = ally.predict(X_te, Y_te)
probs = ally.predict_prob(X_te, Y_te)

loss = np.zeros(NUM_ROUND+1)

loss[0] = F.cross_entropy(probs, Y_te).item()
print(f"\n\nNumber of samples = {sum(idxs_lb)} ------> Loss: {loss[0]} \n\n", flush=True)

sampled = []

for rd in range(1, NUM_ROUND+1):
    print('Round {}'.format(rd), flush=True)
    torch.cuda.empty_cache()
    gc.collect()

    # Query
    output = ally.query(NUM_QUERY)
    q_idxs = output
    sampled += list(q_idxs)
    idxs_lb[q_idxs] = True

    # Update
    ally.update(idxs_lb)
    ally.train()

    # Evaluate round accuracy
    P = ally.predict(X_te, Y_te)
    probs = ally.predict_prob(X_te, Y_te)
    loss[rd] = F.cross_entropy(probs, Y_te).item()
    print(f"\n\nNumber of samples = {sum(idxs_lb)} ------> testing loss: {loss[rd]} \n\n", flush=True)
    if sum(~ally.idxs_lb) < opts.nQuery: 
        sys.exit('Too few remaining samples to query')
