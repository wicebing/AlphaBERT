import os
import time
import unicodedata
import random
import string
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
#import apex
#from apex import amp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed

#import gruModel
import alphabert_dataset_v02
import dg_alphabets
#import bert_unet_001
import alphabert_loss_v02
from alphabert_utils import *

from transformers import tokenization_bert, modeling_bert, configuration_bert


class alphabetPooler(nn.Module):
    def __init__(self):
        super(alphabetPooler, self).__init__()
        self.avgpool = nn.Sequential(nn.Linear(768,256),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(256,64),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(64,2)
                                     )
#        self.activation = nn.Sigmoid()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.avgpool(hidden_states)
        pooled_output = pooled_output.view(-1,2).contiguous()
#        pooled_output = self.activation(pooled_output)
        return pooled_output



class bert_baseModel(nn.Module):
    def __init__(self, init_weight= None, bioBERT=False):
        super(bert_baseModel, self).__init__()    
        if bioBERT:
            config = configuration_bert.BertConfig()
            config.vocab_size = 28996
            weight_path = '../biobert_v1.0_pubmed_pmc/biobert_model.ckpt.index'
            self.bert_model = modeling_bert.BertModel.from_pretrained(weight_path,
                                                                      from_tf=True,
                                                                      config=config)
        else:
            pretrained_weights = 'bert-base-cased'
            self.bert_model = modeling_bert.BertModel.from_pretrained(pretrained_weights)
        self.pool = alphabetPooler()
        
    def forward(self, input_ids,attention_mask, position_ids=None):    

        x = self.bert_model(input_ids,attention_mask)
        x = self.pool(x[0])
        
        return x

# 904405

if __name__ == '__main__':
#    test()
    config = {'hidden_size': 64,
              'max_position_embeddings':1350,
              'eps': 1e-12,
              'input_size': 84,
              'vocab_size':84,
              'hidden_dropout_prob': 0.1,
              'num_attention_heads': 16, 
              'attention_probs_dropout_prob': 0.2,
              'intermediate_size': 64,
              'num_hidden_layers': 8,
              }
    torch.manual_seed(seed = 0)
    x = torch.randint(0,69,[3,20])
    x = torch.cat([x.float(),torch.zeros([3,5]).float()],dim=1)
    m = torch.ones([3,20])
    m = torch.cat([m.float(),torch.zeros([3,5]).float()],dim=1)
    l = [25,21,18]
    
    emb = rnn_embedding(config)
    enc = ExtractiveRNN(config)    
    xx = emb(x)
    xc = enc(xx,l)
    
    lstm = LSTM_baseModel(config=config)
    xl = lstm(x,l)
    
