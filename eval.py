#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
import os
from optparse import OptionParser
from tools.config_tools import Config

#----------------------------------- loading paramters -------------------------------------------#

#--------------------------------------------------------------------------------------------------#

#------------------ environment variable should be set before import torch  -----------------------#
#if opt.cuda:
#    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
#    print('setting gpu on gpuid {0}'.format(opt.gpu_id))
#--------------------------------------------------------------------------------------------------#

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np

import cat_and_dog_model
from dataloader import img_Dataset
from tools import utils


# reminding the cuda option


# test function for metric learning
def test(model, opt, test_batch,Training = False,cross=10):
    """
    train for one epoch on the training set
    """
    # evaluation mode: only useful for the models with batchnorm or dropout
    model.eval()
    model.float()
    evaluation_set = img_Dataset(num_of_cross=test_batch,Training = Training,cross=cross)
    evaluation_loader = torch.utils.data.DataLoader(evaluation_set, batch_size=opt.eval_batchSize,
                                     shuffle=False, num_workers=int(opt.workers))
    right = 0       # correct sample number
    sample_num = len(evaluation_set)   # total sample number

    #------------------------------------ important parameters -----------------------------------------------#
    # bz_sim: the batch similarity between two visual and auditory feature batches
    # slice_sim: the slice similarity between the visual feature batch and the whole auditory feature sets
    # sim_mat: the total simmilarity matrix between the visual and auditory feature datasets
    #-----------------------------------------------------------------------------------------------------#

    for i, ( eval_sample, eval_label,file_name) in enumerate(evaluation_loader):
        #print(np.shape(eval_sample),np.shape(eval_label))
        eval_sample = Variable(eval_sample)
        if opt.cuda:
            eval_sample = eval_sample.cuda()
        predicted_label_one_hot = model(eval_sample)
        predicted_label_one_hot = predicted_label_one_hot.cpu()
        predicted_label_one_hot = predicted_label_one_hot.data.numpy()
        #print(np.shape(predicted_label_one_hot))
        predicted_label = np.argsort(predicted_label_one_hot)[:,0]
        #print(np.shape(predicted_label))
        eval_label = eval_label.numpy()
        #print(eval_label == predicted_label)
        right += np.sum(predicted_label == eval_label)
        final_label_pred = 'finish.txt'
        for i in len(file_name):
            with open(final_label_pred, 'a') as file:
                afile_tmp = file_name[i].split('.')
                file.write(afile_tmp[0] + ' ')
                print('predict label is ',predicted_label[i])
                file.write(str(predicted_label[i])+'\n')
    print("---------------------------------------------------------------")
    if Training:
        print("Training accuracy for training set is ",right/sample_num,' with ',sample_num, 'samples.')

    else:
        print("Test accuracy for test set is ",right/sample_num,' with ',sample_num, 'samples.')
    print("---------------------------------------------------------------")
    return right/sample_num


def main():
    global opt
    parser = OptionParser()
    parser.add_option('--config',
                      type=str,
                      help="evaluation configuration",
                      default="./configs/test_config.yaml")

    (opts, args) = parser.parse_args()
    assert isinstance(opts, object)
    opt = Config(opts.config)
    #print(opt)
    if torch.cuda.is_available():
        if not opt.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with \"cuda: True\"")
    else:
        cudnn.benchmark = True

    # loading test dataset
    test_audio_dataset = dset(opt.data_dir, opt.audio_flist, which_feat='afeat')
    print('number of test samples is: {0}'.format(len(test_video_dataset)))
    print('finished loading data')
    # test data loader
    
    test_audio_loader = torch.utils.data.DataLoader(test_audio_dataset, batch_size=opt.batchSize,
                                     shuffle=False, num_workers=int(opt.workers))
    # create model
    model = mnist_model.mnistModel()
    if opt.cuda:
        print('shift model to GPU .. ')
        model = model.cuda()

    test(test_video_loader, test_audio_loader, model, opt)


if __name__ == '__main__':
    main()
