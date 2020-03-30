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

#from gensim.models import Word2Vec
import alphabert_dataset_v03 as alphabert_dataset
import dg_alphabets
import alphaBERT_model
import alphabert_loss_v02
from alphabert_utils import clean_up_v204_ft, split2words, rouge12l, make_statistics,save_checkpoint,load_checkpoint,clean_up,clean_up_ensemble
from alphabert_utils import count_parameters, plot_all_ROC
from alphabert_dadaloader import alphabet_loaders
from transformers import tokenization_bert, modeling_bert, configuration_bert

batch_size = 1
device = 'cpu'
parallel = True

#device_ids = list(range(rank * n, (rank + 1) * n))

filepath = './data'

all_expect_alphabets = [' ', '#', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.',
 '/', ':', ';', '<', '=', '>', '?', '@', '0', '1', '2', '3', '4', '5', '6', '7', '8',  
 '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
 'y', 'z', '^', '~', '`', '$', '!', '[', ']', '{', '}', '|', '#0#', '#1#','#2#', '#3#',
 '#4#','#5#','#6#','#7#',]


tokenize_alphabets = dg_alphabets.Diagnosis_alphabets()
for a in all_expect_alphabets:
    tokenize_alphabets.addalphabet(a)

config = {'hidden_size': 64,
          'max_position_embeddings':1350,
          'eps': 1e-12,
          'input_size': tokenize_alphabets.n_alphabets, 
          'vocab_size': tokenize_alphabets.n_alphabets, 
          'hidden_dropout_prob': 0.1,
          'num_attention_heads': 16, 
          'attention_probs_dropout_prob': 0.1,
          'intermediate_size': 256,
          'num_hidden_layers': 16,
          }

pretrained_weights = 'bert-base-cased'
bert_tokenizer = tokenization_bert.BertTokenizer.from_pretrained(pretrained_weights)

all_expect_alphabets_lstm = [' ', '#', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.',
 '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?',
 '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '^', 'a', 'b', 'c', 'd', 'e', 'f',
 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
 'x', 'y', 'z', '~']

tokenize_alphabets_lstm = dg_alphabets.Diagnosis_alphabets()
for a in all_expect_alphabets_lstm:
    tokenize_alphabets_lstm.addalphabet(a)


class _loaders():
    def __init__(self, datapath, config, tokenize_alphabets, num_workers=4, batch_size=2):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.config = config
        self.tokenize_alphabets = tokenize_alphabets
        self.filepath = datapath
        self.clamp_size = config['max_position_embeddings']
       
    def get_data_np(self):
        filename1a = os.path.join(self.filepath,'finetune_train.csv')
        data_a = pd.read_csv(filename1a, header=None, encoding='utf-8')
        data_train = data_a

        filename1b = os.path.join(self.filepath,'finetune_valtest.csv')
        data_val = pd.read_csv(filename1b, header=None, encoding='utf-8')

        data_test = data_val.sample(frac=0.5,random_state=1)
        data_val = data_val.drop(data_test.index)
        D_L_train = np.array(data_train)[:,[1,3]]
        D_L_val = np.array(data_val)[:,[1,3]]
        D_L_test = np.array(data_test)[:,[1,3]]
   
        filename2 = os.path.join(self.filepath,'codes.csv')
        data2 = pd.read_csv(filename2, header=None)
        icd_wd = np.array(data2[4])
   
        filename3 = os.path.join(self.filepath,'C2018_OUTPUT_CD.txt')
        data3 = pd.read_csv(filename3,encoding = 'utf8',sep='\t')
        ntuh_wd = np.array(data3['TYPECONTENT'])
        stage1_wd = np.concatenate([icd_wd,ntuh_wd],axis=0)
   
        filename4 = os.path.join(self.filepath,'valtest_18488.csv')
        data4= pd.read_csv(filename4,encoding = 'utf8')
        stage1_test = np.array(data4['Diagnosis'])

        filename1cyy = os.path.join(self.filepath,'finetune_test_cyy.csv')
        data_test_cyy = pd.read_csv(filename1cyy, header=None, encoding='utf-8')
        D_L_test_cyy = np.array(data_test_cyy)[:,[1,3]]
        
        filename1lin= os.path.join(self.filepath,'finetune_test_lin.csv')
        data_test_lin = pd.read_csv(filename1lin, header=None, encoding='utf-8')
        D_L_test_lin = np.array(data_test_lin)[:,[1,3]]

        data_test[2] = 0
        data_test_cyy[2] = 1
        data_test_lin[2] = 2
        
        data_sql = pd.concat([data_test,data_test_cyy,data_test_lin])
        data_sql[0] = range(len(data_sql))

        D_L_test_sql = np.array(data_sql)[:,[1,3]]
        
        data_np = {'D_L_train':D_L_train,
                   'D_L_val':D_L_val,
                   'D_L_test':D_L_test,
                   'stage1_wd':stage1_wd,
                   'stage1_test':stage1_test,
                   'D_L_test_cyy':D_L_test_cyy,
                   'D_L_test_lin':D_L_test_lin
                   }
        return data_np,data_val,data_sql, D_L_test_sql


def ROC0(filepath = './data'):
    global config, tokenize_alphabets, batch_size
    alphaloader2= _loaders(datapath=filepath, 
                              config=config, 
                              tokenize_alphabets=tokenize_alphabets,
                              num_workers=4,
                              batch_size=batch_size)
    data_np,a, aa, b= alphaloader2.get_data_np()
    
    
    _data = alphabert_dataset.D2Lntuh_leading(data_np['D_L_val'],
                                                 tokenize_alphabets,
                                                 1350,
                                                 train=False)
    
    D_loader = DataLoader(_data,
                            batch_size=1,
                            shuffle=False,
                            num_workers=2,
                            collate_fn=alphabert_dataset.collate_fn)
    
    leading_token_idx = tokenize_alphabets.alphabet2idx['|']
    padding_token_idx = tokenize_alphabets.alphabet2idx[' ']
    out_pred_res = []
    
    out_a = []
    out_b = []
    out_c = []
    out_d = []
    
    for batch_idx, sample in enumerate(D_loader):
        print(batch_idx)
        src = sample['src_token']
        trg = sample['trg']
        att_mask = sample['mask_padding']
        origin_len = sample['origin_seq_length']    
    
        referecne = []
    
        isselect_pred = False
        isselect_trg = False
        
        for i, src_ in enumerate(src):
            for j, wp in enumerate(src_):
                if wp == leading_token_idx:
                    if trg[i][j]  > 0:
                        if isselect_trg:
                            referecne.append(padding_token_idx)
                        isselect_trg = True
                    else:
                        if isselect_trg:
                            referecne.append(padding_token_idx)
                        isselect_trg = False                                
                else:
                    if isselect_trg:
                        if wp != leading_token_idx:
                            referecne.append(wp.item())                    
    
            referecne = tokenize_alphabets.convert_idx2str(referecne)
        
            referecne_list = referecne.split()
    
            out_pred_res.append((referecne))
            out_a.append(('000a '+referecne))
            out_b.append(('000b '+referecne))
            out_c.append(('000c '+referecne))
            out_d.append(('000d '+referecne))
     
    npa = np.array(a)    
    out_np_res = np.array(out_pred_res).reshape(-1,1)   
    out_pd_res = pd.DataFrame(out_pred_res)
    
    out_np_a = np.array(out_a).reshape(-1,1)  
    out_np_b = np.array(out_b).reshape(-1,1)  
    out_np_c = np.array(out_c).reshape(-1,1)  
    out_np_d = np.array(out_d).reshape(-1,1)  
    
    
    nppp = np.concatenate([npa,out_np_res,out_np_a,out_np_b,out_np_c,out_np_d],axis=1)
    
    aaaa = pd.DataFrame(nppp)
    aaaa.columns=['index','oriDD','dr','labels','ref','BERT','bioBERT','lstm','ours']
    
    aaaa.to_csv('DD_ROC.csv',index=None)

class test_loaders_BERT():
    def __init__(self, datapath, config, tokenize_alphabets, num_workers=4, batch_size=2):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.config = config
        self.tokenize_alphabets = tokenize_alphabets
        self.filepath = datapath
        self.clamp_size = config['max_position_embeddings']
       
    def get_data_np(self):
        filename1a = os.path.join(self.filepath,'DD_ROC.csv')
        data_a = pd.read_csv(filename1a, encoding='utf-8')
        D_L_test = np.array(data_a)[:,[1,3]]
        return D_L_test,data_a

def ROC12(bert_type = 'W_biobert.pth'):    
    filepath = '../checkpoint_exe'
    # device = 'cpu'
    tloader= test_loaders_BERT(datapath=filepath, 
                              config=config, 
                              tokenize_alphabets=tokenize_alphabets,
                              num_workers=4,
                              batch_size=batch_size)
    
    data_np, data_pd= tloader.get_data_np()
    
    
    _data = alphabert_dataset.D2Lntuh_leading(data_np,
                                                 tokenize_alphabets,
                                                 clamp_size=config['max_position_embeddings'],
                                                 train=False)
    
    D_loader = DataLoader(_data,
                            batch_size=1,
                            shuffle=False,
                            num_workers=2,
                            collate_fn=alphabert_dataset.collate_fn)
    
    bert_ds = alphabert_dataset.D_bert(data_np,
                                         tokenize_alphabets,
                                         bert_tokenizer,
                                         clamp_size=config['max_position_embeddings'],
                                         train=False)
    
    bert_dl = DataLoader(bert_ds,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=alphabert_dataset.collate_fn)
    
    leading_token_idx = tokenize_alphabets.alphabet2idx['|']
    padding_token_idx = tokenize_alphabets.alphabet2idx[' ']
    out_pred_res = []
    
    out_a = []
    out_b = []
    out_c = []
    out_d = []
    
    modelpath= os.path.join(filepath,bert_type)
    DS_model = torch.load(modelpath)
    DS_model.to(device)
    DS_model.eval()
    
    thrsh = 0.37
    
    
    dloader = bert_dl
        
    out_pred_res = []
    out_trg_res = []
    
    all_pred_trg_biobert = {'pred':[], 'trg':[]}
    
    rouge_set = []
    with torch.no_grad():
        for batch_idx, sample in enumerate(dloader): 
            print(batch_idx)
            src = sample['src_token']
            trg = sample['trg']
            att_mask = sample['mask_padding']
            origin_len = sample['origin_seq_length']
            
            bs = src.shape
        
            src = src.float().to(device)
            trg = trg.float().to(device)
            att_mask = att_mask.float().to(device)
            origin_len = origin_len.to(device)
            
            pred_prop_ = DS_model(input_ids=src.long(),
                                 attention_mask=att_mask)
        
            cls_idx = src==101
            max_pred_prop, pred_prop2 = pred_prop_.view(*bs,-1).max(dim=2)
        
            pred_prop_bin_softmax = nn.Softmax(dim=-1)(pred_prop_.view(*bs,-1))
            pred_prop = pred_prop_bin_softmax[:,:,1]
                                             
            for i, src_ in enumerate(src):
                all_pred_trg_biobert['pred'].append(pred_prop[i][cls_idx[i]].cpu())
                all_pred_trg_biobert['trg'].append(trg[i][cls_idx[i]].cpu())                               
        
                referecne = []
                hypothesis = []
                isselect_pred = False
                isselect_trg = False
                for j, wp in enumerate(src_):
                    if wp == 101:
                        if pred_prop[i][j]  > thrsh:
                            isselect_pred = True
                        else:
                            isselect_pred = False
                        if trg[i][j]  > 0:
                            isselect_trg = True
                        else:
                            isselect_trg = False
        
                    else:
                        if isselect_pred:
                            if wp > 0:
                                hypothesis.append(wp.item())
                        if isselect_trg:
                            if wp > 0:
                                referecne.append(wp.item())
                hypothesis = bert_tokenizer.convert_ids_to_tokens(hypothesis)
                referecne = bert_tokenizer.convert_ids_to_tokens(referecne)
                
                hypothesis = bert_tokenizer.convert_tokens_to_string(hypothesis)
                referecne = bert_tokenizer.convert_tokens_to_string(referecne)
                
                hypothesis_list = hypothesis.split()
                referecne_list = referecne.split()
                
                rouge_set.append((hypothesis_list,referecne_list))
        
                out_trg_res.append((referecne))        
                out_pred_res.append((hypothesis))      
    return all_pred_trg_biobert

class test_loaders_LSTM():
    def __init__(self, datapath, config, tokenize_alphabets, num_workers=4, batch_size=2):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.config = config
        self.tokenize_alphabets = tokenize_alphabets
        self.filepath = datapath
        self.clamp_size = config['max_position_embeddings']
       
    def get_data_np(self):
        filename1a = os.path.join(self.filepath,'DD_ROC.csv')
        data_a = pd.read_csv(filename1a, encoding='utf-8')
        D_L_test = np.array(data_a)[:,[1,3]]
        return D_L_test,data_a

def ROC3():
    filepath = '../checkpoint_exe'
    tloader= test_loaders_LSTM(datapath=filepath, 
                              config=config, 
                              tokenize_alphabets=tokenize_alphabets_lstm,
                              num_workers=4,
                              batch_size=batch_size)
    
    data_np, data_pd= tloader.get_data_np()
    
    
    D2S_test = alphabert_dataset.D2Lntuh(data_np,
                                         tokenize_alphabets_lstm,
                                         clamp_size=config['max_position_embeddings'],
                                         train=False)
    
    D2S_testloader = DataLoader(D2S_test,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=alphabert_dataset.collate_fn_lstm)
    
    out_pred_res = []
    # device = 'cpu'
    modelpath= os.path.join(filepath,'W_lstm_pretrain.pth')
    DS_model = torch.load(modelpath)
    DS_model.to(device)
    DS_model.eval()
    threshold = 0.33
    
    
    dloader = D2S_testloader
        
    out_pred_res = []
    out_trg_res = []
    
    all_pred_trg_lstm = {'pred':[], 'trg':[]}
    
    rouge_set = []
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(dloader): 
            print(batch_idx)
            src = sample['src_token']
            trg = sample['trg']
            att_mask = sample['mask_padding']
            origin_len = sample['origin_seq_length']
    
            src = src.float().to(device)
            trg = trg.float().to(device)
            att_mask = att_mask.float().to(device)
            origin_len = origin_len.to(device)
            
            pred_prop = DS_model(x=src,
                                 x_lengths=origin_len)
            if True:
                pred_prop = clean_up(src,pred_prop,mean_max=max,tokenize_alphabets=tokenize_alphabets_lstm)
            
            for i, src_ in enumerate(src):
                # all_pred_trg_lstm['pred'].append(pred_prop[i][:origin_len[i]].cpu())
                # all_pred_trg_lstm['trg'].append(trg[i][:origin_len[i]].cpu())
                
                src_split, src_isword = split2words(src_,
                                                    tokenize_alphabets=tokenize_alphabets_lstm,
                                                    rouge=True)                            
                referecne = []
                hypothesis = []
                trg_ = []
                pred_= []

                for j in range(len(src_split)):
                    if src_isword[j]>0:
                        if trg[i][src_split[j]][0].cpu()>threshold:
                            referecne.append(tokenize_alphabets_lstm.convert_idx2str(src_[src_split[j]]))
                        if pred_prop[i][src_split[j]][0].cpu()>threshold:
                            hypothesis.append(tokenize_alphabets_lstm.convert_idx2str(src_[src_split[j]]))

                        trg_.append(trg[i][src_split[j]][0].cpu())
                        pred_.append(pred_prop[i][src_split[j]][0].cpu())
                       
                rouge_set.append((hypothesis,referecne))

                all_pred_trg_lstm['trg'].append(torch.tensor(trg_))
                all_pred_trg_lstm['pred'].append(torch.tensor(pred_))  
    
    return all_pred_trg_lstm

class test_loaders():
    def __init__(self, datapath, config, tokenize_alphabets, num_workers=4, batch_size=2):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.config = config
        self.tokenize_alphabets = tokenize_alphabets
        self.filepath = datapath
        self.clamp_size = config['max_position_embeddings']
       
    def get_data_np(self):
        filename1a = os.path.join(self.filepath,'DD_ROC.csv')
        data_a = pd.read_csv(filename1a, encoding='utf-8')
        D_L_test = np.array(data_a)[:,[1,3]]
        return D_L_test,data_a
    
def ROC4():    
    filepath = '../checkpoint_exe'
    tloader= test_loaders(datapath=filepath, 
                              config=config, 
                              tokenize_alphabets=tokenize_alphabets,
                              num_workers=4,
                              batch_size=batch_size)
    
    data_np, data_pd= tloader.get_data_np()
    
    
    D2S_test = alphabert_dataset.D2Lntuh(data_np,
                                         tokenize_alphabets,
                                         clamp_size=config['max_position_embeddings'],
                                         train=False)
    
    D2S_testloader = DataLoader(D2S_test,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=alphabert_dataset.collate_fn)
    
    out_pred_res = []
    
    #device = 'cpu'
    modelpath= os.path.join(filepath,'W_d2s_total_0302_947.pth')
    DS_model = torch.load(modelpath)
    DS_model.to(device)
    DS_model.eval()
    threshold = 0.96

    
    
    dloader = D2S_testloader
        
    out_pred_res = []
    out_trg_res = []
    
    all_pred_trg_ours = {'pred':[], 'trg':[]}
    
    rouge_set = []
    
    leading_token_idx = tokenize_alphabets.alphabet2idx['|']
    padding_token_idx = tokenize_alphabets.alphabet2idx[' ']
    
    ensemble = True
    mean_max = 'mean'

    with torch.no_grad():
        for batch_idx, sample in enumerate(dloader): 
            print(batch_idx)
            src = sample['src_token']
            trg = sample['trg']
            att_mask = sample['mask_padding']
            origin_len = sample['origin_seq_length']
    
            src = src.float().to(device)
            trg = trg.float().to(device)
            att_mask = att_mask.float().to(device)
            origin_len = origin_len.to(device)
            
            bs = src.shape
     
    
            pooled_output, = DS_model(input_ids=src,
                                 attention_mask=att_mask,
                                 out='finehead')
            pred_prop_bin = pooled_output.view(*bs,-1)
            
            pred_prop = clean_up_v204_ft(src,pred_prop_bin,
                                         tokenize_alphabets=tokenize_alphabets,
                                         mean_max=mean_max)
    
            pred_selected = pred_prop > threshold
            trg_selected = trg > threshold
            
            for i, src_ in enumerate(src):                                
                src_split, src_isword = split2words(src_,
                                                    tokenize_alphabets=tokenize_alphabets,
                                                    rouge=True)
                referecne = []
                hypothesis = []
                trg_ = []
                pred_= []
                
                for j in range(len(src_split)):
                    if src_isword[j]>0:
                        if trg[i][src_split[j]][0].cpu()>threshold:
                            referecne.append(tokenize_alphabets.convert_idx2str(src_[src_split[j]]))
                        if pred_prop[i][src_split[j]][0].cpu()>threshold:
                            hypothesis.append(tokenize_alphabets.convert_idx2str(src_[src_split[j]]))
    
                        trg_.append(trg[i][src_split[j]][0].cpu())
                        pred_.append(pred_prop[i][src_split[j]][0].cpu())
                        
                rouge_set.append((hypothesis,referecne))
                
                all_pred_trg_ours['trg'].append(torch.tensor(trg_))
                all_pred_trg_ours['pred'].append(torch.tensor(pred_))  
                 
                a_ = tokenize_alphabets.convert_idx2str(src_[:origin_len[i]])
                s_ = ''.join(i+' ' for i in hypothesis)
                t_ = ''.join(i+' ' for i in referecne)
    
            out_pred_res.append(s_)
            out_trg_res.append(t_)   
    
    return all_pred_trg_ours


all_pred_trg_bert = ROC12(bert_type='W_bert.pth')
all_pred_trg_biobert = ROC12(bert_type='W_biobert.pth')
all_pred_trg_lstm = ROC3()
all_pred_trg_ours = ROC4()
    
def check_len():
    for i in range(250):
        l0=len(all_pred_trg_bert['pred'][i])
        l1=len(all_pred_trg_biobert['pred'][i])
        l2=len(all_pred_trg_lstm['pred'][i])
        l3=len(all_pred_trg_ours['pred'][i])
        
        print(l0,l1,l2,l3)
        
all_pred_trg = [all_pred_trg_biobert,all_pred_trg_bert,all_pred_trg_lstm,all_pred_trg_ours]

plot_all_ROC(all_pred_trg)