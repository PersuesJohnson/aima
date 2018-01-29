#!/usr/bin/env python

from __future__ import print_function
import argparse
import random
import time
import os
import numpy as np
from optparse import OptionParser

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR as LR_Policy

import torch.nn as nn
import cat_and_dog_model as mnist_model
from dataloader import img_Dataset as mnist_Dataset
from tools.config_tools import Config
from tools import utils

import matplotlib as mpl
import pickle
from eval import test

mpl.use('Agg')

from matplotlib import pyplot as plt

parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="training configuration",
                  default="./configs/test_config.yaml")

(opts, args) = parser.parse_args()
assert isinstance(opts, object)
opt = Config(opts.config)
#print(opt)

if opt.checkpoint_folder is None:
    opt.checkpoint_folder = 'checkpoints'

# make dir
if not os.path.exists(opt.checkpoint_folder):
    os.system('mkdir {0}'.format(opt.checkpoint_folder))




def main():
    global opt
    train_dataset = mnist_Dataset(num_of_cross=0,cross=1)
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with \"cuda: True\"")
        torch.manual_seed(opt.manualSeed)
    else:
        if int(opt.ngpu) == 1:
            print('so we use 1 gpu to training')
            print('setting gpu on gpuid {0}'.format(opt.gpu_id))

            if opt.cuda:
                os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
                torch.cuda.manual_seed(opt.manualSeed)
                cudnn.benchmark = True
    #loss_rec = np.load('acc_train.npy')
    #acc_rec = np.load('acc.npy')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                                   shuffle=True, num_workers=int(opt.workers))

    # create model
    model = mnist_model.cat_and_dog_resnet()

    if opt.init_model != '':
        print('loading pretrained model from {0}'.format(opt.init_model))
        model.load_state_dict(torch.load(opt.init_model))
    if opt.cuda:
        print('shift model and criterion to GPU .. ')
        model = model.cuda()
        # criterion = criterion.cuda()
    acc = test(model,opt,0,Training =False,cross=1)


if __name__ == '__main__':
    main()

