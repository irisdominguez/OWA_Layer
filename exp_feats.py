#!/usr/bin/env python
# coding: utf-8

bs = 64

import os
USE_GPUS = '2'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = USE_GPUS
N_GPUS = len(USE_GPUS.split(','))

from fastai.metrics import error_rate
import sys
import copy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch.optim

from order_metrics import *
from model import *
from utils import *
from evaluation import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

log_folder = 'res'
log_suffix = 't5_feats'
log_file = os.path.join(log_folder, f'{log_suffix}.txt')
matrix_path = os.path.join('matrixes', log_suffix)

data = load_dataset("cifar100", bs=bs)

lr = 1e-4

epochs = 80
its = 40

for f in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    config = {
        'type': 'layer',
        'data': data,
        'model': create_small_model,
        'matrixpath': matrix_path,
        'lr': lr,
        'epochs': epochs,
        'its': its,
        'pos': {3: f},
        'order': total_variation_image_batch,
        'aggregate': 'imagenette',
        'constrainmode': 'full_owa',
        'init_denominator': 128,
        'metric': top_k_accuracy
    }
    log_results(run_config(config), log_file)


