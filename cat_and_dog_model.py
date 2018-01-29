import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision.models as models


class Inception(nn.Module):
    def __init__(self,in_channels,out_channels):  
        super(Inception, self).__init__()
        out_channels = int(out_channels/4)
        self.branch1x1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                              kernel_size=(1,1), padding=0,stride=1)  
    
        self.branch3x3_1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                              kernel_size=(3,3), padding=1,stride=1)    
        self.branch3x3_2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,
                              kernel_size=(3,3), padding=1,stride=1)  
        self.branch5x5_1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                              kernel_size=(5,5), padding=2,stride=1)    
        self.branch5x5_2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,
                              kernel_size=(5,5), padding=2,stride=1)    
    
        self.branchpool_1 = nn.MaxPool2d(kernel_size=(3,3),stride=1,padding=1)
        self.branchpool_2 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                              kernel_size=(3,3), padding=1,stride=1)  
    
    def forward(self,data):
        out_1x1 = self.branch1x1(data)
        out_3x3 = self.branch3x3_1(data)
        out_3x3 = self.branch3x3_2(out_3x3)
        out_5x5 = self.branch5x5_1(data)
        out_5x5 = self.branch5x5_2(out_5x5)
        out_branch = self.branchpool_1(data)
        out_branch = self.branchpool_2(out_branch) 
        x = torch.cat((out_1x1,out_3x3,out_5x5,out_branch),dim=1)
        return x      


class cat_and_dog_resnet(nn.Module):
    def __init__(self):
        super(cat_and_dog_resnet, self).__init__()
        self.resnet101 = models.resnet101(pretrained = True)
        self.resnet101.fc = torch.nn.Linear(2048,2)
    
    def forward(self,data):
        print(self.resnet101(data))
        return self.resnet101(data)
        

class StableBCELoss(nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, inputdata, target):
        #neg_abs = - input.abs()
        dist = inputdata-target
        loss = torch.pow(dist,2)
        #loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()
