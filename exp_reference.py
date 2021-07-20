#!/usr/bin/env python
# coding: utf-8

bs = 64

import os
USE_GPUS = '0'
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
log_suffix = 't5_reference'
log_file = os.path.join(log_folder, f'{log_suffix}.txt')
matrix_path = os.path.join('matrixes', log_suffix)

data = load_dataset("cifar100", bs=bs)

config = {
    'type': 'reference',
    'data': data,
    'model': create_small_model,
    'matrixpath': matrix_path
}

lr = 1e-4

epochs = 80
its = 40

config = {
    'type': 'reference',
    'data': data,
    'model': create_small_model,
    'matrixpath': matrix_path,
    'lr': lr,
    'epochs': epochs,
    'its': its,
    'metric': top_k_accuracy
}
log_results(run_config(config), log_file)






