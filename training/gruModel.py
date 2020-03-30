import os
import unicodedata
import string
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset

from alphabert_utils import *


class ExtractiveRNN(nn.Module):
    def __init__(self, config):
        super(ExtractiveRNN, self).__init__()
        self.hidden_size = config['hidden_size']

        self.rnn = nn.LSTM(config['hidden_size'],
                          config['hidden_size'],
                          9,
                          batch_first=True,
                          bidirectional=True,
                          dropout=0.1)
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)
    
    def forward(self, x, x_lengths, hidden=None):      
        pack_x = nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)
        output, hidden = self.rnn(pack_x, hidden)
        
        x = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return x



class alphabetPooler(nn.Module):
    def __init__(self, config):
        super(alphabetPooler, self).__init__()
        self.avgpool = nn.Sequential(nn.Linear(2*config['hidden_size'],2*config['hidden_size']),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(2*config['hidden_size'],2*config['hidden_size']),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(2*config['hidden_size'],1)
                                     )
        self.activation = nn.Sigmoid()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.avgpool(hidden_states)
        pooled_output = pooled_output.view(hidden_states.shape[:2])
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertLayerNorm(nn.Module):
    def __init__(self, config):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(config['hidden_size']))
        self.bias = nn.Parameter(torch.zeros(config['hidden_size']))
        self.variance_epsilon = config['eps']

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
    
class rnn_embedding(nn.Module):
    def __init__(self, config, init_weight= None):
        super(rnn_embedding, self).__init__()
        if init_weight is None:
            self.LSTMembedding = nn.Embedding(config['input_size'], config['hidden_size'])
        else:
            self.LSTMembedding = nn.Embedding.from_pretrained(init_weight,freeze=False)
        self.LayerNorm = BertLayerNorm(config)
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, x):
        embeddings = self.LSTMembedding(x.long())
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class LSTM_baseModel(nn.Module):
    def __init__(self, config, init_weight= None):
        super(LSTM_baseModel, self).__init__()    

        self.emb = rnn_embedding(config, init_weight=init_weight)
        self.lstm = ExtractiveRNN(config)
        self.pool = alphabetPooler(config)
        
    def forward(self, x, x_lengths):    
        x = self.emb(x)
        x = self.lstm(x, x_lengths)
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
    
