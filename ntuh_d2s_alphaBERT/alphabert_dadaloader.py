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

import alphabert_dataset_v05 as alphabert_dataset

class alphabet_loaders():
    def __init__(self, datapath, config, tokenize_alphabets, num_workers=4, batch_size=2,
                 bioBERTcorpus=0,):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.config = config
        self.tokenize_alphabets = tokenize_alphabets
        self.filepath = datapath
        self.clamp_size = config['max_position_embeddings']
        self.bioBERTcorpus = bioBERTcorpus
       
    def get_data_np(self):
        filename1a = os.path.join(self.filepath,'finetune_train.csv')
        data_a = pd.read_csv(filename1a, header=None, encoding='utf-8')
        data_train = data_a

        filename1b = os.path.join(self.filepath,'finetune_valtest.csv')
        data_val = pd.read_csv(filename1b, header=None, encoding='utf-8')

        data_test = data_val.sample(frac=0.5,random_state=1)
        data_val = data_val.drop(data_test.index)
        D_L_set = np.array(data_a)[:,[1,3]]
        D_L_train = np.array(data_train)[:,[1,3]]
        D_L_val = np.array(data_val)[:,[1,3]]
        D_L_test = np.array(data_test)[:,[1,3]]
   
        filename2 = os.path.join(self.filepath,'codes.csv')
        data2 = pd.read_csv(filename2, header=None)
        icd_wd = np.array(data2[4])
   
        filename3 = os.path.join(self.filepath,'C2018_OUTPUT_CD.txt')
        data3 = pd.read_csv(filename3,encoding = 'utf8',sep='\t')
        ntuh_wd = np.array(data3['TYPECONTENT'])
        if self.bioBERTcorpus == 1:
            bioBERTcorpus = from_bioBERT_corpus()
            stage1_wd = np.concatenate([icd_wd,ntuh_wd,bioBERTcorpus],axis=0)
        elif self.bioBERTcorpus == 2:
            bioBERTcorpus = from_bioBERT_corpus()
            Others = from_Other_corpus()
            stage1_wd = np.concatenate([icd_wd,ntuh_wd,bioBERTcorpus,Others],axis=0)
        elif self.bioBERTcorpus == 3:
            bioBERTcorpus = from_bioBERT_corpus()
            Others = from_Other_corpus()
            Gutenberg = from_Gutenberg()
            stage1_wd = np.concatenate([icd_wd,ntuh_wd,bioBERTcorpus,Others,Gutenberg],axis=0)    
        elif self.bioBERTcorpus == 4:
            bioBERTcorpus = from_bioBERT_corpus()
            Gutenberg = from_Gutenberg()
            stage1_wd = np.concatenate([icd_wd,ntuh_wd,bioBERTcorpus,Gutenberg],axis=0)    
        else:
            bioBERTcorpus = from_bioBERT_corpus
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

        
        data_np = {'D_L_train':D_L_train,
                   'D_L_val':D_L_val,
                   'D_L_test':D_L_test,
                   'stage1_wd':stage1_wd,
                   'stage1_test':stage1_test,
                   'D_L_test_cyy':D_L_test_cyy,
                   'D_L_test_lin':D_L_test_lin
                   }
        return data_np
    
    def make_loaders(self,finetune=False,head=False,headlong=False,headfirst=False,
                     ensemble=False,ahead=False,pretrain=False,trvte='test'):
        data_np = self.get_data_np()
        train = False
        if finetune:
            if trvte == 'train':
                train = True
                ds = data_np['D_L_train']
            elif trvte == 'val':                
                ds = data_np['D_L_val']                
            elif trvte == 'test':
                ds = data_np['D_L_test']
            elif trvte == 'test_cyy':
                ds = data_np['D_L_test_cyy']                               
        
            if head:
                if ensemble:
                    D2S_data = alphabert_dataset.D2Lntuh_ensemble(ds,
                                                                 self.tokenize_alphabets,
                                                                 self.clamp_size,
                                                                 train=train)
                else:
                    D2S_data = alphabert_dataset.D2Lntuh_leading(ds,
                                                                 self.tokenize_alphabets,
                                                                 self.clamp_size,
                                                                 train=train)
            else:
                D2S_data = alphabert_dataset.D2Lntuh(ds,
                                                     self.tokenize_alphabets,
                                                     clamp_size=self.clamp_size,
                                                     train=train)
        
        
            D2S_loader = DataLoader(D2S_data,
                                    batch_size=self.batch_size,
                                    shuffle=train,
                                    num_workers=self.num_workers,
                                    collate_fn=alphabert_dataset.collate_fn)
            if head:
                if ensemble:
                    if trvte == 'train':
                        loaders = {'D2S_ensemble_trainloader':D2S_loader,
                                   'D2S_ensemble_trainset':D2S_data}
                    elif trvte == 'val':
                        loaders = {'D2S_ensemble_valloader':D2S_loader,
                                   'D2S_ensemble_valset':D2S_data}               
                    elif trvte == 'test':  
                        loaders = {'D2S_ensemble_testloader':D2S_loader,
                                   'D2S_ensemble_testset':D2S_data}
                    elif trvte == 'test_cyy':
                        loaders = {'D2S_ensemble_cyy_testloader':D2S_loader,
                                   'D2S_ensemble_test_cyy':D2S_data}
                else:
                    if trvte == 'train':
                        loaders = {'D2S_head_trainloader':D2S_loader,
                                   'D2S_head_trainset':D2S_data}
                    elif trvte == 'val':
                        loaders = {'D2S_head_valloader':D2S_loader,
                                   'D2S_head_valset':D2S_data}               
                    elif trvte == 'test':  
                        loaders = {'D2S_head_testloader':D2S_loader,
                                   'D2S_head_testset':D2S_data}
                    elif trvte == 'test_cyy':
                        loaders = {'D2S_head_cyy_testloader':D2S_loader,
                                   'D2S_head_test_cyy':D2S_data}
            else:
                if trvte == 'train':
                    loaders = {'D2S_trainloader':D2S_loader,
                               'D2S_trainset':D2S_data}
                elif trvte == 'val':
                    loaders = {'D2S_valloader':D2S_loader,
                               'D2S_valset':D2S_data}               
                elif trvte == 'test':  
                    loaders = {'D2S_testloader':D2S_loader,
                               'D2S_testset':D2S_data}
                elif trvte == 'test_cyy':
                    loaders = {'D2S_cyy_testloader':D2S_loader,
                               'D2S_test_cyy':D2S_data} 

            return loaders            
#            else:
#                if trvte == 'train':
#                    train = True
#                    ds = data_np.D_L_train
#                elif trvte == 'val':
#                    train = False
#                    ds = data_np.D_L_val                
#                elif trvte == 'test':
#                    train = False
#                    ds = data_np.D_L_test                    
#                
#                D2S_data = alphabert_dataset.D2Lntuh(ds,
#                                                     self.tokenize_alphabets,
#                                                     clamp_size=self.clamp_size,
#                                                     train=train)
#            
#                D2S_loader = DataLoader(D2S_data,
#                                        batch_size=self.batch_size,
#                                        shuffle=train,
#                                        num_workers=self.num_workers,
#                                        collate_fn=alphabert_dataset.collate_fn)
#                if trvte == 'train':
#                    loaders = {'D2S_trainloader':D2S_loader,
#                               'D2S_trainset':D2S_data}
#                elif trvte == 'val':
#                    loaders = {'D2S_valloader':D2S_loader,
#                               'D2S_valset':D2S_data}               
#                elif trvte == 'test':  
#                    loaders = {'D2S_testloader':D2S_loader,
#                               'D2S_testset':D2S_data}
#                    
#                return loaders
          
        if pretrain:
            ds_train = data_np['stage1_wd']
            ds_test = data_np['stage1_test']
            if head:
                stage1_head_dataset = alphabert_dataset.D_stage1_head_token(ds_train,
                                                            self.tokenize_alphabets,
                                                            clamp_size=self.clamp_size,
                                                            icd10=71704,
                                                            train=True)
        
                stage1_head_dataset_test = alphabert_dataset.D_stage1_head_token(ds_test,
                                                            self.tokenize_alphabets,
                                                            clamp_size=self.clamp_size,
                                                            train=False)

                stage1_head_dataloader = DataLoader(stage1_head_dataset,
                                               batch_size=self.batch_size,
                                               shuffle=True,
                                               num_workers=self.num_workers,
                                               collate_fn=alphabert_dataset.collate_fn_head)
                
                stage1_head_dataloader_test = DataLoader(stage1_head_dataset_test,
                                               batch_size=self.batch_size,
                                               shuffle=True,
                                               num_workers=self.num_workers,
                                               collate_fn=alphabert_dataset.collate_fn_head)
    
                loaders = {'stage1_head_dataloader':stage1_head_dataloader,
                           'stage1_head_dataloader_test':stage1_head_dataloader_test,
                           'stage1_head_dataset':stage1_head_dataset,
                           'stage1_head_dataset_test':stage1_head_dataset_test}
                
                return loaders 
            
            elif headlong:
                stage1_head_dataset = alphabert_dataset.D_stage1_head_longtoken(ds_train,
                                                            self.tokenize_alphabets,
                                                            clamp_size=self.clamp_size,
                                                            icd10=71704,
                                                            train=True)
        
                stage1_head_dataset_test = alphabert_dataset.D_stage1_head_longtoken(ds_test,
                                                            self.tokenize_alphabets,
                                                            clamp_size=self.clamp_size,
                                                            train=False)

                stage1_head_dataloader = DataLoader(stage1_head_dataset,
                                               batch_size=self.batch_size,
                                               shuffle=True,
                                               num_workers=self.num_workers,
                                               collate_fn=alphabert_dataset.collate_fn_head)
                
                stage1_head_dataloader_test = DataLoader(stage1_head_dataset_test,
                                               batch_size=self.batch_size,
                                               shuffle=True,
                                               num_workers=self.num_workers,
                                               collate_fn=alphabert_dataset.collate_fn_head)
    
                loaders = {'stage1_head_dataloader':stage1_head_dataloader,
                           'stage1_head_dataloader_test':stage1_head_dataloader_test,
                           'stage1_head_dataset':stage1_head_dataset,
                           'stage1_head_dataset_test':stage1_head_dataset_test}
                
                return loaders      
            
            elif headfirst:
                stage1_head_dataset = alphabert_dataset.D_stage1_head_firsttoken(ds_train,
                                                            self.tokenize_alphabets,
                                                            clamp_size=self.clamp_size,
                                                            icd10=71704,
                                                            train=True)
        
                stage1_head_dataset_test = alphabert_dataset.D_stage1_head_firsttoken(ds_test,
                                                            self.tokenize_alphabets,
                                                            clamp_size=self.clamp_size,
                                                            train=False)

                stage1_head_dataloader = DataLoader(stage1_head_dataset,
                                               batch_size=self.batch_size,
                                               shuffle=True,
                                               num_workers=self.num_workers,
                                               collate_fn=alphabert_dataset.collate_fn_head)
                
                stage1_head_dataloader_test = DataLoader(stage1_head_dataset_test,
                                               batch_size=self.batch_size,
                                               shuffle=True,
                                               num_workers=self.num_workers,
                                               collate_fn=alphabert_dataset.collate_fn_head)
    
                loaders = {'stage1_head_dataloader':stage1_head_dataloader,
                           'stage1_head_dataloader_test':stage1_head_dataloader_test,
                           'stage1_head_dataset':stage1_head_dataset,
                           'stage1_head_dataset_test':stage1_head_dataset_test}
                
                return loaders               
            else:
                if ahead:
                    stage1_dataset = alphabert_dataset.D_stage1_ahead(ds_train,
                                                                self.tokenize_alphabets,
                                                                clamp_size=self.clamp_size,
                                                                icd10=71704,
                                                                train=True)
        
                    stage1_dataset_test = alphabert_dataset.D_stage1_ahead(ds_test,
                                                                self.tokenize_alphabets,
                                                                clamp_size=self.clamp_size,
                                                                train=False)
                
                    stage1_dataloader = DataLoader(stage1_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=self.num_workers,
                                                   collate_fn=alphabert_dataset.collate_fn)
                      
                    stage1_dataloader_test = DataLoader(stage1_dataset_test,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=self.num_workers,
                                                   collate_fn=alphabert_dataset.collate_fn)
        
                    loaders = {'stage1_dataloader':stage1_dataloader,
                               'stage1_dataloader_test':stage1_dataloader_test,
                               'stage1_dataset':stage1_dataset,
                               'stage1_dataset_test':stage1_dataset_test}
        
                    return loaders                    
                else:
                    stage1_dataset = alphabert_dataset.D_stage1(ds_train,
                                                                self.tokenize_alphabets,
                                                                clamp_size=self.clamp_size,
                                                                icd10=71704,
                                                                train=True)
        
                    stage1_dataset_test = alphabert_dataset.D_stage1(ds_test,
                                                                self.tokenize_alphabets,
                                                                clamp_size=self.clamp_size,
                                                                train=False)
                
                    stage1_dataloader = DataLoader(stage1_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=self.num_workers,
                                                   collate_fn=alphabert_dataset.collate_fn)
                      
                    stage1_dataloader_test = DataLoader(stage1_dataset_test,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=self.num_workers,
                                                   collate_fn=alphabert_dataset.collate_fn)
        
                    loaders = {'stage1_dataloader':stage1_dataloader,
                               'stage1_dataloader_test':stage1_dataloader_test,
                               'stage1_dataset':stage1_dataset,
                               'stage1_dataset_test':stage1_dataset_test}
        
                    return loaders
                

def from_bioBERT_corpus():
    print(' == load bioBERT_corpus dataset == ')
    import glob
    fp = '../Downloads'
    fp1 = os.path.join(fp,'REdata','*')
    fp1 = glob.glob(fp1)
    
    fp2 = []
    for fp in fp1:
        fpi = os.path.join(fp,'*')
        fpi = glob.glob(fpi)
        fp2 += fpi
        
    fp3a = []
    for fp in fp2:
        fpi = os.path.join(fp,'test.*')
        fpi = glob.glob(fpi)
        fp3a += fpi

    fp3b = []
    for fp in fp2:
        fpi = os.path.join(fp,'train.*')
        fpi = glob.glob(fpi)
        fp3b += fpi
    
    _wd= []
    for fp in fp3a:
        _dat = pd.read_csv(fp, header=None, encoding='utf-8',sep='\t',)
        _wd.append(np.array(_dat[1]))
    for fp in fp3b:
        _dat = pd.read_csv(fp, header=None, encoding='utf-8',sep='\t',)
        _wd.append(np.array(_dat[0]))  
        
    wd = np.concatenate(_wd,axis=0)
#=========================================
    
    fp = '../Downloads'
    fp1 = os.path.join(fp,'QA','BioASQ','Bio*')
    fp1 = glob.glob(fp1)
    
    _wdd= []
    for fp in fp1:
        _dat = pd.read_json(fp)
        _dat = _dat.data[0]
        _dat = _dat['paragraphs']
        for _dati in _dat:
            _wdd.append(_dati['context'])
    
    wd2 = np.array(_wdd)
#=========================================
    wda = np.concatenate([wd,wd2],axis=0)
    
    return wda

def from_Other_corpus():
    print(' == load Other_corpus dataset == ')
    import glob
    fp = '../Downloads'
    fp1 = os.path.join(fp,'ICD','*')
    fp1 = glob.glob(fp1)
    
    _wd= []
    
    _dat = pd.read_csv(fp1[0], encoding='utf-8',sep='\t',)
    _wd.append(np.array(_dat['TYPECONTENT']))

    _dat = pd.read_csv(fp1[2], encoding='utf-8',sep='\t',)
    _wd.append(np.array(_dat['TYPECONTENT']))
        
    wd = np.concatenate(_wd,axis=0)
#=========================================   
    return wd

def from_Gutenberg():
    print(' == load Gutenberg dataset == ')
    import glob
    fp = '../Downloads'
    fp1 = os.path.join(fp,'Gutenberg','txt','*.txt')
    fp1 = glob.glob(fp1)
    
    _wd= []
    for fp_ptah in fp1:
        try:
            fp = open(fp_ptah, "r")
            _str = fp.read().replace('\n', ' ')
            fp.close()
                
            _str_list = _str.split('.')
            _stxt_list = []
            n = 100
            for i in range(int(len(_str_list)/n)):
                _si = _str_list[n*i:n*(i+1)]
                _stxt = ''.join(_i for _i in _si)
                _stxt_list.append(_stxt)
            
            _wd += _stxt_list
        except:
            pass
    
    _wd2 = [i for i in _wd if len(i)>0]
    
#    ridx = random.randint(0,len(_wd2)-10000)        
#    wd = np.array(_wd2[ridx:ridx+10000])
    wd = np.array(_wd2)
#=========================================   
    return wd


if __name__ == '__main__':
    a = from_bioBERT_corpus()
    
 
    
