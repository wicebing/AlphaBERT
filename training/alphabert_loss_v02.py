import pandas as pd
import numpy as np
#import cv2
import math 
import glob
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Alphabert_loss(nn.Module):
    def __init__(self,device):
        super(Alphabert_loss,self).__init__()
        self.device = device
        self.criterion = nn.BCELoss().to(device)
        
    def forward(self,predict,target,length):
        p_ = []
        t_ = []
        for i, l in enumerate(length):
            p_.append(predict[i][:l].unsqueeze(0))
            t_.append(target[i][:l].unsqueeze(0))
        
        p = torch.cat(p_,dim=1).contiguous()
        t =torch.cat(t_,dim=1).contiguous()
        
        total_loss = self.criterion(p,t)
        
        return total_loss

class Alphabert_satge1_loss(nn.Module):
    def __init__(self,device):
        super(Alphabert_satge1_loss,self).__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
        
    def forward(self,predict,target,length,err_cloze):
        p_ = []
        t_ = []
        for i, l in enumerate(length):
            lmask = err_cloze < l
            lmask = err_cloze[lmask]
            p_.append(predict[i][lmask])
            t_.append(target[i][lmask].unsqueeze(0))
        
        p = torch.cat(p_,dim=0).contiguous()
        t =torch.cat(t_,dim=1)
        t = t.view(-1).contiguous()
        
        total_loss = self.criterion(p,t)      
#        print(p.shape, p)
#        print(t.shape, t)
#        print(length)
#        print(len(err_cloze),err_cloze)       
        return total_loss
    
if __name__ == '__main__':
    torch.manual_seed(seed = 1)
    predict = torch.rand(size = [5,10])
    target = torch.randint(low=0, high =2, size = [5,10]).float()
    length = torch.tensor([5,7,8,4,10])
    
    cri = Alphabert_loss('cpu')