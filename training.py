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
                  default="./configs/train_config.yaml")

(opts, args) = parser.parse_args()
assert isinstance(opts, object)
opt = Config(opts.config)
print(opt)

if opt.checkpoint_folder is None:
    opt.checkpoint_folder = 'checkpoints'

# make dir
if not os.path.exists(opt.checkpoint_folder):
    os.system('mkdir {0}'.format(opt.checkpoint_folder))



# training function for metric learning
def train(train_loader, model, criterion, optimizer, epoch, opt,eponum):
    """
    train for one epoch on the training set
    """
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()

    # training mode
    model.train()
    model.float()
    loss_rec = []
    end = time.time()
    for i, (data, label) in enumerate(train_loader):
        #model.train()
        #model.float()
        # shuffling the index orders
        bz = label.size()[0]
        orders = np.arange(bz).astype('int32')
        shuffle_orders = orders.copy()
        np.random.shuffle(shuffle_orders)

        # creating a new data with the shuffled indices
        data = data[torch.from_numpy(shuffle_orders).long()].clone()
        label = label[torch.from_numpy(shuffle_orders).long()].clone()

        # concat the vfeat and afeat respectively

        # generating the labels
        # 1. the labels for the shuffled feats

        # 2. the labels for the original feats
        #label = label.astype(np.int64)
        #label = torch.from_numpy(label)
        label = label.view(label.size(0))
        #one_hot = torch.zeros(np.shape(label)[0], 10).scatter_(1, label, 1)
        #one_hot = one_hot.type(torch.LongTensor)
        one_hot = torch.LongTensor(label)
        # transpose the feats
        # vfeat0 = vfeat0.transpose(2, 1)
        # afeat0 = afeat0.transpose(2, 1)


        # put the data into Variable
        data_var = Variable(data)
        target_var = Variable(one_hot)

        # if you have gpu, then shift data to GPU
        if opt.cuda:
            data_var = data_var.cuda()
            target_var = target_var.cuda()

        # forward, backward optimize
        sim = model(data_var)  # inference simialrity
        #print(sim)
        #print(target_var)
        loss = criterion(sim, target_var)  # compute contrastive loss

        loss_rec.append(list(loss.data)[0])

        ##############################
        # update loss in the loss meter
        ##############################
        losses.update(loss.data[0], label.size(0))

        ##############################
        # compute gradient and do sgd
        ##############################
        optimizer.zero_grad()
        loss.backward()

        ##############################
        # gradient clip stuff
        ##############################
        # utils.clip_gradient(optimizer, opt.gradient_clip)

        # update parameters
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % opt.print_freq == 0:
            log_str = 'Fold:[{3}] Epoch: [{0}][{1}/{2}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                eponum, i, len(train_loader), epoch, batch_time=batch_time, loss=losses)
            print(log_str)


def main():
    global opt
    loss_rec = np.zeros((opt.folds,100))
    acc_rec = np.zeros((opt.folds,100))
    #loss_rec = np.load('acc_train.npy')
    #acc_rec = np.load('acc.npy')
    for iteration in range(opt.folds):
        train_dataset = mnist_Dataset(num_of_cross=iteration)

        print('number of train samples is: {0}'.format(len(train_dataset)))
        print('finished loading data')

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
        print('Random Seed: {0}'.format(opt.manualSeed))
        # train data loader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                                   shuffle=True, num_workers=int(opt.workers))

        # create model
        model = mnist_model.cat_and_dog_resnet()

        if opt.init_model != '':
            print('loading pretrained model from {0}'.format(opt.init_model))
            model.load_state_dict(torch.load(opt.init_model))

        # Contrastive Loss
        #criterion = mnist_model.StableBCELoss()
        criterion = nn.CrossEntropyLoss()

        if opt.cuda:
            print('shift model and criterion to GPU .. ')
            model = model.cuda()
            criterion = criterion.cuda()

        # optimizer
        # optimizer = optim.SGD(model.parameters(), lr=opt.lr,
        #                      momentum=opt.momentum,
        #                      weight_decay=opt.weight_decay)

        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
        # optimizer = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, momentum=opt.momentum)
        # optimizer = optim.Adadelta(params=model.parameters(), lr=opt.lr)
        # adjust learning rate every lr_decay_epoch
        lambda_lr = lambda epoch: opt.lr_decay ** ((epoch + 1) // opt.lr_decay_epoch)  # poly policy
        scheduler = LR_Policy(optimizer, lambda_lr)

        resume_epoch = 0
        acc = test(model, opt, iteration)
        acc_rec[iteration][0] = acc
        acc = test(model,opt,iteration,Training = True)
        loss_rec[iteration][0] = acc
        for epoch in range(resume_epoch, opt.max_epochs):
            #################################
            # train for one epoch
            #################################
            #accuracy = test(model, opt, epoch)
            train(train_loader, model, criterion, optimizer, iteration, opt,epoch)
            scheduler.step()

            ##################################
            # save checkpoints
            ##################################

            # save model every 10 epochs
            accuracy = test(model, opt, iteration)
            acc_rec[iteration][epoch+1] = accuracy
            np.save('acc.npy',acc_rec)
            accuracy = test(model, opt, iteration,Training = True)
            loss_rec[iteration][epoch+1] = accuracy
            np.save('acc_train.npy',loss_rec)

            if ((epoch + 1) % opt.epoch_save) == 0:
                path_checkpoint = '{0}/{1}_{3}_epoch{2}.pth'.format(opt.checkpoint_folder, opt.prefix, epoch + 1,iteration)
                utils.save_checkpoint(model.state_dict(), path_checkpoint)


if __name__ == '__main__':
    main()
