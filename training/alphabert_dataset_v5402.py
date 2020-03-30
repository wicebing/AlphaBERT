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
from alphabert_utils import split2words


class D2Lntuh(Dataset):
    def __init__(self,ds,tokenizer,clamp_size,train=True):
        self.len = len(ds)
        self.ds = ds
        self.tokenizer = tokenizer
        self.clamp_size = clamp_size
        self.train = train
    
    def __getitem__(self, index):
        diagnosis = self.ds[index][0]
        label = self.ds[index][1]

        if self.train:
            if random.random()>0.8:
                diagnoses_num = random.randint(0,25)
                random_sequence = torch.randperm(self.len)
                random_diagnoses_idx = random_sequence[:diagnoses_num]
                random_diagnoses = self.ds[random_diagnoses_idx]
                if diagnoses_num>1:
                    for d in random_diagnoses:
                        if len(diagnosis) < 2000:
                            diagnosis += ' ; '+d[0]
                            label += '000'+d[1]
                elif diagnoses_num ==1 :
                    if len(diagnosis) < 2000:
                        diagnosis += ' ; '+random_diagnoses[0]
                        label += '000'+random_diagnoses[1]
            else:
                first_idx = random.choices(range(self.len),k=3)
                random_diagnoses_idx = random.choices(first_idx,k=25)
                random_diagnoses = self.ds[random_diagnoses_idx]
                selected_ids = [index,]
                for i,d in enumerate(random_diagnoses):
                    if len(diagnosis) < 2000:
                        ridx = random_diagnoses_idx[i]                        
                        if ridx in selected_ids:
                            diagnosis += ' ; '+d[0]
                            label += '000'+'0'*len(d[0])
                        else:
                            selected_ids.append(ridx)
                            diagnosis += ' ; '+d[0]
                            label += '000'+d[1]                 
            
        diagnosis_list = list(diagnosis)
        
        label_list = list(label)        
        label_np = np.array(label_list, dtype='int')
        
        diagnosis_token = self.tokenizer.tokenize(diagnosis_list)
        
#        indexed_tokens_inp = self.tokenizer.encode(st)
#        indexed_tokens_trg = self.tokenizer.encode(st)
        
        tokens_tensors = torch.tensor(diagnosis_token)
        labels_tensors = torch.tensor(label_np)
        
        if len(tokens_tensors) > self.clamp_size:
            clamp = random.randint(0,len(tokens_tensors)-self.clamp_size)
            tokens_tensors = tokens_tensors[clamp:clamp+self.clamp_size]
            labels_tensors = labels_tensors[clamp:clamp+self.clamp_size]
            
        s,t = make_error_typo(tokens_tensors,self.tokenizer,percent=0.10)
        
        sample = {'src_token':s,
                  'trg':labels_tensors,
#                  'src':diagnosis_list,
                  }        
        return sample
    
    def __len__(self):
        return self.len

def collate_fn(datas):
    from torch.nn.utils.rnn import pad_sequence
    batch = {}
    tokens_tensors = [DD['src_token'] for DD in datas]
    labels_tensors = [DD['trg'] for DD in datas]
    origin_seq_length = [len(d) for d in tokens_tensors]

    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True,
                                  padding_value=0)
    labels_tensors = pad_sequence(labels_tensors,
                                  batch_first=True,
                                  padding_value=-1)

    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    for i, e in enumerate(origin_seq_length):
        masks_tensors[i,:e+1] = 1 
#        labels_tensors[i,e:] = -1

    batch['src_token'] = tokens_tensors
    batch['trg'] = labels_tensors
    batch['mask_padding'] = masks_tensors
    batch['origin_seq_length'] = torch.tensor(origin_seq_length)

    return batch

def collate_fn_lstm(datas):
    from torch.nn.utils.rnn import pad_sequence
    batch = {}
    tokens_tensors = [DD['src_token'] for DD in datas]
    labels_tensors = [DD['trg'] for DD in datas]
    origin_seq_length = [len(d) for d in tokens_tensors]
    
    temp = torch.tensor(origin_seq_length)
    
    sorted_seq_length,sorted_seq = temp.sort(descending=True)
    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True)
    labels_tensors = pad_sequence(labels_tensors,
                                  batch_first=True)

    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    for i, e in enumerate(origin_seq_length):
        masks_tensors[i,:e+1] = 1 
        labels_tensors[i,e:] = -1

    batch['src_token'] = tokens_tensors[sorted_seq]
    batch['trg'] = labels_tensors[sorted_seq]
    batch['mask_padding'] = masks_tensors[sorted_seq]
    batch['origin_seq_length'] = sorted_seq_length

    return batch

def collate_fn_head(datas):
    from torch.nn.utils.rnn import pad_sequence
    batch = {}
    tokens_tensors = [DD['src_token'] for DD in datas]
    labels_tensors = [DD['trg'] for DD in datas]
    origin_seq_length = [len(d) for d in tokens_tensors]
    tokens_ori_tensors = [DD['src_origin'] for DD in datas]
    resdis_tensors = [DD['dis'] for DD in datas]

    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True,
                                  padding_value=0)
    labels_tensors = pad_sequence(labels_tensors,
                                  batch_first=True,
                                  padding_value=-1)
    tokens_ori_tensors = pad_sequence(tokens_ori_tensors, 
                                  batch_first=True,
                                  padding_value=0)
    resdis_tensors = pad_sequence(resdis_tensors,
                                  batch_first=True,
                                  padding_value=-1)

    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    for i, e in enumerate(origin_seq_length):
        masks_tensors[i,:e+1] = 1 
#        labels_tensors[i,e:] = -1

    batch['src_token'] = tokens_tensors
    batch['trg'] = labels_tensors
    batch['mask_padding'] = masks_tensors
    batch['origin_seq_length'] = torch.tensor(origin_seq_length)
    batch['src_origin'] = tokens_ori_tensors
    batch['dis'] = resdis_tensors

    return batch


def make_cloze(src,tokenize_alphabets,percent=0.15, fix=False,word=1):
    if random.random()>word:
        word_idx = split2words(src,tokenize_alphabets,rouge=False,iswordfilt=False)
        max_len = len(word_idx)
#        err_sequence = torch.randperm(max_len)
#        err_replace = int(percent*max_len)
#        err_sequence = err_sequence[:err_replace]
        
        words = [e for e in word_idx if len(word_idx[e])>1]
        random.shuffle(words)
        err_replace2 = int(percent*len(words))
        err_sequence2 = words[:err_replace2]
        
        s = []
        t = []
        
        for e in range(max_len):
            if e not in err_sequence2:
                s.append(src[word_idx[e]])
                t.append(torch.tensor([-1]*len(word_idx[e])))
            else:
                s.append(torch.tensor([tokenize_alphabets.alphabet2idx['^']]*len(word_idx[e])))
                t.append(src[word_idx[e]])
        _s = torch.cat(s,dim=0)
        _t = torch.cat(t,dim=0)
        return _s, _t
    else:
        max_len = len(src)
        err_sequence = torch.randperm(max_len)
        err_replace = int(percent*max_len)
        err_sequence = err_sequence[:err_replace]
    
        numbers = torch.arange(tokenize_alphabets.alphabet2idx['0'],
                               tokenize_alphabets.alphabet2idx['9']+1)
        alphabets = torch.arange(tokenize_alphabets.alphabet2idx['A'],
                                 tokenize_alphabets.alphabet2idx['z']+1)
        alphabets = alphabets[alphabets!=tokenize_alphabets.alphabet2idx['^']]
        try:
            alphabets = alphabets[alphabets!=tokenize_alphabets.alphabet2idx['`']]
        except:
            pass
        try:
            alphabets = alphabets[alphabets!=tokenize_alphabets.alphabet2idx['[']]
            alphabets = alphabets[alphabets!=tokenize_alphabets.alphabet2idx[']']]
        except:
            pass
        
        numbers_li = list(numbers.numpy())
        alphabets_li = list(alphabets.numpy())
        others_li = [o for o in range(tokenize_alphabets.n_alphabets) if o not in numbers_li+alphabets_li]
        others = torch.tensor(others_li)
        
        try:
            others = others[others!=tokenize_alphabets.alphabet2idx['$']]
            others = others[others!=tokenize_alphabets.alphabet2idx['|']]
            others = others[others!=tokenize_alphabets.alphabet2idx['!']]
            others = others[others!=tokenize_alphabets.alphabet2idx['#0#']]
            others = others[others!=tokenize_alphabets.alphabet2idx['#1#']]
            others = others[others!=tokenize_alphabets.alphabet2idx['#2#']]
            others = others[others!=tokenize_alphabets.alphabet2idx['#3#']]                                                        
            others = others[others!=tokenize_alphabets.alphabet2idx['#4#']]
            others = others[others!=tokenize_alphabets.alphabet2idx['#5#']]                                                        
            others = others[others!=tokenize_alphabets.alphabet2idx['#6#']]
        except:
            pass
                                                                
        s = src.clone()
        for e in err_sequence:
            if s[e] in alphabets:
                if fix:
                    s[e] = tokenize_alphabets.alphabet2idx['^']
                else:
                    r = random.random()
                    if r>0.2:
                        s[e] = tokenize_alphabets.alphabet2idx['^']
                    elif r <= 0.2 and r > 0.1:
    #                    random_idx = random.randint(0, len(alphabets)-1)
    #                    s[e] = alphabets[random_idx]
                        s[e] = random.choice(alphabets)
            elif s[e] in numbers:
                if fix:
                    pass
    #                s[e] = tokenize_alphabets.alphabet2idx['^']
                else:
                    r = random.random()
                    if r>0.8:
                        s[e] = tokenize_alphabets.alphabet2idx['^']
                    elif r <= 0.8 and r > 0.4:
    #                    random_idx = random.randint(0, len(numbers)-1)
    #                    s[e] = numbers[random_idx] 
                        s[e] = random.choice(numbers)
            elif s[e] ==0:
                if fix:
                    pass
    #                s[e] = tokenize_alphabets.alphabet2idx['^']
                else:
                    r = random.random()
                    if r>0.9:
                        s[e] = tokenize_alphabets.alphabet2idx['^']
                    elif r <= 0.9 and r > 0.8:
    #                    random_idx = random.randint(0, len(others)-1)
    #                    s[e] = others[random_idx]
                        s[e] = random.choice(others)
            else:
                if fix:
                    pass
    #                s[e] = tokenize_alphabets.alphabet2idx['^']
                else:
                    r = random.random()
                    if r>0.8:
                        s[e] = tokenize_alphabets.alphabet2idx['^']
                    elif r <= 0.8 and r > 0.4:
    #                    random_idx = random.randint(0, len(others)-1)
    #                    s[e] = others[random_idx]   
                        s[e] = random.choice(others)
                        
        rev_err_sequence = [idx for idx in range(max_len) if idx not in err_sequence]
        t = src.clone()
        t[rev_err_sequence] = -1
        
        return s, t


def make_error_typo(src,tokenize_alphabets,percent=0.15):
    max_len = len(src)
    err_sequence = torch.randperm(max_len)
    err_replace = int(percent*max_len)
    err_sequence = err_sequence[:err_replace]

    numbers = torch.arange(tokenize_alphabets.alphabet2idx['0'],
                           tokenize_alphabets.alphabet2idx['9']+1)
    alphabets = torch.arange(tokenize_alphabets.alphabet2idx['A'],
                             tokenize_alphabets.alphabet2idx['z']+1)
    alphabets = alphabets[alphabets!=tokenize_alphabets.alphabet2idx['^']]
    try:
        alphabets = alphabets[alphabets!=tokenize_alphabets.alphabet2idx['`']]
    except:
        pass
    try:
        alphabets = alphabets[alphabets!=tokenize_alphabets.alphabet2idx['[']]
        alphabets = alphabets[alphabets!=tokenize_alphabets.alphabet2idx[']']]
    except:
        pass
    
    numbers_li = list(numbers.numpy())
    alphabets_li = list(alphabets.numpy())
    others_li = [o for o in range(tokenize_alphabets.n_alphabets) if o not in numbers_li+alphabets_li]
    others = torch.tensor(others_li)
    
    try:
        others = others[others!=tokenize_alphabets.alphabet2idx['$']]
        others = others[others!=tokenize_alphabets.alphabet2idx['|']]
        others = others[others!=tokenize_alphabets.alphabet2idx['!']]
        others = others[others!=tokenize_alphabets.alphabet2idx['#0#']]
        others = others[others!=tokenize_alphabets.alphabet2idx['#1#']]
        others = others[others!=tokenize_alphabets.alphabet2idx['#2#']]
        others = others[others!=tokenize_alphabets.alphabet2idx['#3#']]                                                        
        others = others[others!=tokenize_alphabets.alphabet2idx['#4#']]
        others = others[others!=tokenize_alphabets.alphabet2idx['#5#']]                                                        
        others = others[others!=tokenize_alphabets.alphabet2idx['#6#']]
    except:
        pass
                                                            
    s = src.clone()
    for e in err_sequence:
        r = random.random()
        if s[e] in alphabets:
            if r>0.9:
                s[e] = random.choice(alphabets)
        elif s[e] in numbers:
            if r>0.95:
                s[e] = random.choice(numbers)
        elif s[e] ==0:
            pass
        else:
            if r>0.95:
                s[e] = random.choice(others)
                    
    rev_err_sequence = [idx for idx in range(max_len) if idx not in err_sequence]
    t = src.clone()
    t[rev_err_sequence] = -1
    
    return s, t

class D_stage1(Dataset):
    def __init__(self,ds,tokenizer,clamp_size,icd10=71704, train=True, fix=False):
        self.len = len(ds)
        self.ds = ds
        self.tokenizer = tokenizer
        self.clamp_size = clamp_size
        self.icd = icd10
        self.train = train
        self.fix = fix
    
    def __getitem__(self, index):
        diagnosis = self.ds[index]
#        diagnosis = '|'+ diagnosis
        if self.train:
            if index < self.icd:
                diagnoses_num = random.randint(0,29)
                random_sequence = torch.randperm(self.icd)
                random_diagnoses_idx = random_sequence[:diagnoses_num]
                random_diagnoses = self.ds[random_diagnoses_idx]
                if diagnoses_num>1:
                    for d in random_diagnoses:
                        diagnosis += '; '+d
                elif diagnoses_num ==1 :
                    diagnosis += '; '+random_diagnoses
            else:
                diagnoses_num = random.randint(0,8)
                random_sequence = torch.randperm(self.len-self.icd)
                random_sequence += self.icd
                random_diagnoses_idx = random_sequence[:diagnoses_num]
                random_diagnoses = self.ds[random_diagnoses_idx]
                if diagnoses_num>1:
                    for d in random_diagnoses:
                        if len(diagnosis) < 2000:
                            diagnosis += '; '+d
                elif diagnoses_num ==1 :
                    if len(diagnosis) < 2000:
                        diagnosis += '; '+random_diagnoses    
                
        diagnosis_list = list(diagnosis)
        
        diagnosis_token = self.tokenizer.tokenize(diagnosis_list)
        
        tokens_tensors = torch.tensor(diagnosis_token)

        if len(tokens_tensors) > self.clamp_size:
            clamp = random.randint(0,len(tokens_tensors)-self.clamp_size)
            tokens_tensors = tokens_tensors[clamp:clamp+self.clamp_size]

        s,t = make_cloze(tokens_tensors,self.tokenizer,
                         percent=0.15, 
                         fix=self.fix,
                         word=0.2)
        
        sample = {'src_token':s,
                  'trg':t,
#                  'src':diagnosis_list,
                  }        
        return sample
   
    def __len__(self):
        return self.len
    
    
class D_bert(Dataset):
    def __init__(self,ds,tokenizer, bert_tokenizer,clamp_size,train=True):
        self.len = len(ds)
        self.ds = ds
        self.tokenizer = tokenizer
        self.bert_tokenizer = bert_tokenizer
        self.clamp_size = clamp_size
        self.train = train
    
    def __getitem__(self, index):
        diagnosis = self.ds[index][0]
        label = self.ds[index][1]

        if self.train:
            diagnoses_num = random.randint(0,25)
            random_sequence = torch.randperm(self.len)
            random_diagnoses_idx = random_sequence[:diagnoses_num]
            random_diagnoses = self.ds[random_diagnoses_idx]
            if diagnoses_num>1:
                for d in random_diagnoses:
                    if len(diagnosis) < 2000:
                        diagnosis += ' ; '+d[0]
                        label += '000'+d[1]
            elif diagnoses_num ==1 :
                if len(diagnosis) < 2000:
                    diagnosis += ' ; '+random_diagnoses[0]
                    label += '000'+random_diagnoses[1]
        
        diagnosis_list = list(diagnosis)
        
        label_list = list(label)        
        label_np = np.array(label_list, dtype='int')
        
        diagnosis_token = self.tokenizer.tokenize(diagnosis_list)
        
#        indexed_tokens_inp = self.tokenizer.encode(st)
#        indexed_tokens_trg = self.tokenizer.encode(st)
        
        tokens_tensors = torch.tensor(diagnosis_token)
        labels_tensors = torch.tensor(label_np)
        
        if self.train:
            if len(tokens_tensors) > self.clamp_size:
                clamp = random.randint(0,len(tokens_tensors)-self.clamp_size)
                tokens_tensors = tokens_tensors[clamp:clamp+self.clamp_size]
                labels_tensors = labels_tensors[clamp:clamp+self.clamp_size]
        else:
            if len(tokens_tensors) > self.clamp_size:
                tokens_tensors = tokens_tensors[:self.clamp_size]
                labels_tensors = labels_tensors[:self.clamp_size]  

        src_split, src_isword = split2words(tokens_tensors,rouge=True,tokenize_alphabets=self.tokenizer)
        bert_cls_label = []
        bert_cls_seqence = []
        bert_seqence = []
        
        for j in range(len(src_split)):
            if src_isword[j]>0:
#                if labels_tensors[src_split[j]][0].cpu()>0:
                # use cls represent the whole word
                bert_cls_label.append(labels_tensors[src_split[j]][0].item())
                bert_cls_seqence.append(self.bert_tokenizer.cls_token)
                bert_cls_seqence.append(self.tokenizer.convert_idx2str(tokens_tensors[src_split[j]]))
                bert_seqence.append(self.tokenizer.convert_idx2str(tokens_tensors[src_split[j]]))
        
        CLS_str = ''
        for w in bert_cls_seqence:
            CLS_str += w + ' '
        
        bert_cls_seqence_tokens = self.bert_tokenizer.tokenize(CLS_str)
        bert_label = -1*np.ones(len(bert_cls_seqence_tokens))
        
        bert_cls_seqence_tokens = np.array(bert_cls_seqence_tokens)        
        cls_mask = bert_cls_seqence_tokens == self.bert_tokenizer.cls_token
        bert_label[cls_mask] = np.array(bert_cls_label)
        
        if len(bert_cls_seqence_tokens) > 512:
            bert_cls_seqence_tokens = bert_cls_seqence_tokens[:512]
            bert_label = bert_label[:512]
            
        bert_cls_seqence_ids = self.bert_tokenizer.convert_tokens_to_ids(bert_cls_seqence_tokens)        
        bert_cls_seqence_ids = torch.tensor(bert_cls_seqence_ids)

        bert_label = torch.tensor(bert_label)

                    
        sample = {'src_token':bert_cls_seqence_ids,
                  'trg':bert_label,
#                  'CLS_src':torch.tensor(bert_cls_seqence),
#                  'src':torch.tensor(bert_seqence),
                  }        
        return sample
    
    def __len__(self):
        return self.len  
    
class D2Lntuh_filter(Dataset):
    def __init__(self,ds,tokenizer,clamp_size,train=True):
        self.len = len(ds)
        self.ds = ds
        self.tokenizer = tokenizer
        self.clamp_size = clamp_size
        self.train = train
    
    def __getitem__(self, index):
        diagnosis = self.ds[index][0]
        label = self.ds[index][1]

        if self.train:
            diagnoses_num = random.randint(0,25)
            random_sequence = torch.randperm(self.len)
            random_diagnoses_idx = random_sequence[:diagnoses_num]
            random_diagnoses = self.ds[random_diagnoses_idx]
            if diagnoses_num>1:
                for d in random_diagnoses:
                    if len(diagnosis) < 2000:
                        diagnosis += ' ; '+d[0]
                        label += '000'+d[1]
            elif diagnoses_num ==1 :
                if len(diagnosis) < 2000:
                    diagnosis += ' ; '+random_diagnoses[0]
                    label += '000'+random_diagnoses[1]
        
        diagnosis_list = list(diagnosis)
        
        label_list = list(label)        
        label_np = np.array(label_list, dtype='int')
        
        diagnosis_token = self.tokenizer.tokenize(diagnosis_list)
        
        tokens_tensors = torch.tensor(diagnosis_token)
        labels_tensors = torch.tensor(label_np)
        
        if self.train:
            if len(tokens_tensors) > self.clamp_size:
                clamp = random.randint(0,len(tokens_tensors)-self.clamp_size)
                tokens_tensors = tokens_tensors[clamp:clamp+self.clamp_size]
                labels_tensors = labels_tensors[clamp:clamp+self.clamp_size]
        else:
            if len(tokens_tensors) > self.clamp_size:
                tokens_tensors = tokens_tensors[:self.clamp_size]
                labels_tensors = labels_tensors[:self.clamp_size]  

        src_split, src_isword = split2words(tokens_tensors,rouge=True,tokenize_alphabets=self.tokenizer)
        bert_fliter_label_ = []
        bert_fliter_seqence_ = []
        
        for j in range(len(src_split)):
            if src_isword[j]>0:
#                if labels_tensors[src_split[j]][0].cpu()>0:
                # use cls represent the whole word
                bert_fliter_label_.append(torch.tensor([-1]))
                bert_fliter_label_.append(labels_tensors[src_split[j]])
                bert_fliter_seqence_.append(torch.tensor([0]))
                bert_fliter_seqence_.append(tokens_tensors[src_split[j]])
        
        labels_tensors = torch.cat(bert_fliter_label_)
        tokens_tensors = torch.cat(bert_fliter_seqence_)

        if len(tokens_tensors) > self.clamp_size:
            tokens_tensors = tokens_tensors[:self.clamp_size]
            labels_tensors = labels_tensors[:self.clamp_size]
        
        sample = {'src_token':tokens_tensors,
                  'trg':labels_tensors,
#                  'src':diagnosis_list,
                  }        
        return sample

    def __len__(self):
        return self.len

class D2Lntuh_leading(Dataset):
    def __init__(self,ds,tokenizer,clamp_size,train=True):
        self.len = len(ds)
        self.ds = ds
        self.tokenizer = tokenizer
        self.clamp_size = clamp_size
        self.train = train
        self.leading_token_idx = self.tokenizer.alphabet2idx['|']
    
    def __getitem__(self, index):
        diagnosis = self.ds[index][0]
        label = self.ds[index][1]

        if self.train:
            if random.random()>0.8:
                diagnoses_num = random.randint(0,25)
                random_sequence = torch.randperm(self.len)
                random_diagnoses_idx = random_sequence[:diagnoses_num]
                random_diagnoses = self.ds[random_diagnoses_idx]
                if diagnoses_num>1:
                    for d in random_diagnoses:
                        if len(diagnosis) < 2000:
                            diagnosis += ' ; '+d[0]
                            label += '000'+d[1]
                elif diagnoses_num ==1 :
                    if len(diagnosis) < 2000:
                        diagnosis += ' ; '+random_diagnoses[0]
                        label += '000'+random_diagnoses[1]
            else:
                first_idx = random.choices(range(self.len),k=3)
                random_diagnoses_idx = random.choices(first_idx,k=25)
                random_diagnoses = self.ds[random_diagnoses_idx]
                selected_ids = [index,]
                for i,d in enumerate(random_diagnoses):
                    if len(diagnosis) < 2000:
                        ridx = random_diagnoses_idx[i]                        
                        if ridx in selected_ids:
                            diagnosis += ' ; '+d[0]
                            label += '000'+'0'*len(d[0])
                        else:
                            selected_ids.append(ridx)
                            diagnosis += ' ; '+d[0]
                            label += '000'+d[1]
        
        diagnosis_list = list(diagnosis)
        
        label_list = list(label)        
        label_np = np.array(label_list, dtype='int')
        
        diagnosis_token = self.tokenizer.tokenize(diagnosis_list)
        
        tokens_tensors = torch.tensor(diagnosis_token)
        labels_tensors = torch.tensor(label_np)
        
        if self.train:
            if len(tokens_tensors) > self.clamp_size:
                if random.random()>0.3:
                    tokens_tensors = tokens_tensors[:self.clamp_size]
                    labels_tensors = labels_tensors[:self.clamp_size]
                else:
                    clamp = random.randint(0,len(tokens_tensors)-self.clamp_size)
                    tokens_tensors = tokens_tensors[clamp:clamp+self.clamp_size]
                    labels_tensors = labels_tensors[clamp:clamp+self.clamp_size]
        else:
            if len(tokens_tensors) > self.clamp_size:
                tokens_tensors = tokens_tensors[:self.clamp_size]
                labels_tensors = labels_tensors[:self.clamp_size]  

        src_split, src_isword = split2words(tokens_tensors,rouge=True,tokenize_alphabets=self.tokenizer)
        bert_fliter_label_ = []
        bert_fliter_seqence_ = []
        bert_cls_label = []
               
        for j in range(len(src_split)):
            if src_isword[j]>0:
#                if labels_tensors[src_split[j]][0].cpu()>0:
                # use cls represent the whole word
                bert_cls_label.append(labels_tensors[src_split[j]][0].item())
#                bert_fliter_label_.append(torch.tensor([-1]))
#                bert_fliter_label_.append(labels_tensors[src_split[j]])
                bert_fliter_seqence_.append(torch.tensor([self.leading_token_idx]))
                bert_fliter_seqence_.append(tokens_tensors[src_split[j]])
        
#        labels_tensors = torch.cat(bert_fliter_label_)
        tokens_tensors = torch.cat(bert_fliter_seqence_)
        
        labels_tensors = -1*torch.ones(tokens_tensors.shape)
              
        cls_mask = tokens_tensors == self.leading_token_idx
        labels_tensors[cls_mask] = torch.tensor(bert_cls_label,dtype=torch.float)

        if len(tokens_tensors) > self.clamp_size:
            tokens_tensors = tokens_tensors[:self.clamp_size]
            labels_tensors = labels_tensors[:self.clamp_size]
        
        sample = {'src_token':tokens_tensors,
                  'trg':labels_tensors,
#                  'src':diagnosis_list,
                  }        
        return sample
    
    def __len__(self):
        return self.len

class D2Lntuh_ensemble(Dataset):
    def __init__(self,ds,tokenizer,clamp_size,train=True):
        self.len = len(ds)
        self.ds = ds
        self.tokenizer = tokenizer
        self.clamp_size = clamp_size
        self.train = train
        self.leading_token_idx = self.tokenizer.alphabet2idx['|']
    
    def __getitem__(self, index):
        diagnosis = self.ds[index][0]
        label = self.ds[index][1]

        if self.train:
            if random.random()>0.8:
                diagnoses_num = random.randint(0,25)
                random_sequence = torch.randperm(self.len)
                random_diagnoses_idx = random_sequence[:diagnoses_num]
                random_diagnoses = self.ds[random_diagnoses_idx]
                if diagnoses_num>1:
                    for d in random_diagnoses:
                        if len(diagnosis) < 2000:
                            diagnosis += ' ; '+d[0]
                            label += '000'+d[1]
                elif diagnoses_num ==1 :
                    if len(diagnosis) < 2000:
                        diagnosis += ' ; '+random_diagnoses[0]
                        label += '000'+random_diagnoses[1]
            else:
                first_idx = random.choices(range(self.len),k=3)
                random_diagnoses_idx = random.choices(first_idx,k=25)
                random_diagnoses = self.ds[random_diagnoses_idx]
                selected_ids = [index,]
                for i,d in enumerate(random_diagnoses):
                    if len(diagnosis) < 2000:
                        ridx = random_diagnoses_idx[i]                        
                        if ridx in selected_ids:
                            diagnosis += ' ; '+d[0]
                            label += '000'+'0'*len(d[0])
                        else:
                            selected_ids.append(ridx)
                            diagnosis += ' ; '+d[0]
                            label += '000'+d[1]
        
        diagnosis_list = list(diagnosis)
        
        label_list = list(label)        
        label_np = np.array(label_list, dtype='int')
        
        diagnosis_token = self.tokenizer.tokenize(diagnosis_list)
        
        tokens_tensors = torch.tensor(diagnosis_token)
        labels_tensors = torch.tensor(label_np)
        
        if self.train:
            if len(tokens_tensors) > self.clamp_size:
                if random.random()>0.0:
                    tokens_tensors = tokens_tensors[:self.clamp_size]
                    labels_tensors = labels_tensors[:self.clamp_size]
                else:
                    clamp = random.randint(0,len(tokens_tensors)-self.clamp_size)
                    tokens_tensors = tokens_tensors[clamp:clamp+self.clamp_size]
                    labels_tensors = labels_tensors[clamp:clamp+self.clamp_size]
        else:
            if len(tokens_tensors) > self.clamp_size:
                tokens_tensors = tokens_tensors[:self.clamp_size]
                labels_tensors = labels_tensors[:self.clamp_size]             

        src_split, src_isword = split2words(tokens_tensors,rouge=True,tokenize_alphabets=self.tokenizer)
        bert_fliter_seqence_ = []
        bert_cls_label = []
               
        for j in range(len(src_split)):
            if src_isword[j]>0:
#                if labels_tensors[src_split[j]][0].cpu()>0:
                # use cls represent the whole word
                bert_cls_label.append(labels_tensors[src_split[j]][0].item())
                bert_cls_label+=list(labels_tensors[src_split[j]])

                bert_fliter_seqence_.append(torch.tensor([self.leading_token_idx]))
                bert_fliter_seqence_.append(tokens_tensors[src_split[j]])
        
#        labels_tensors = torch.cat(bert_fliter_label_)
        tokens_tensors = torch.cat(bert_fliter_seqence_)
        
#        labels_tensors = -1*torch.ones(tokens_tensors.shape)
              
#        cls_mask = tokens_tensors == self.leading_token_idx
        labels_tensors = torch.tensor(bert_cls_label,dtype=torch.float)

        if len(tokens_tensors) > self.clamp_size:
            tokens_tensors = tokens_tensors[:self.clamp_size]
            labels_tensors = labels_tensors[:self.clamp_size]
        
        sample = {'src_token':tokens_tensors,
                  'trg':labels_tensors,
                  }        
        return sample
    
    def __len__(self):
        return self.len

class D_stage1_ahead(Dataset):
    def __init__(self,ds,tokenizer,clamp_size,icd10=71704, train=True, fix=False):
        self.len = len(ds)
        self.ds = ds
        self.tokenizer = tokenizer
        self.clamp_size = clamp_size
        self.icd = icd10
        self.train = train
        self.fix = fix
        self.leading_token_idx = self.tokenizer.alphabet2idx['|']
        self.padding_token_idx = self.tokenizer.alphabet2idx[' ']        
    
    def __getitem__(self, index):
        diagnosis = self.ds[index]

        if self.train:
            diagnoses_num = random.randint(3,29)
            random_sequence = torch.randperm(self.len)
            random_diagnoses_idx = random_sequence[:diagnoses_num]
            random_diagnoses = self.ds[random_diagnoses_idx]
            if diagnoses_num>1:
                for d in random_diagnoses:
                    if len(diagnosis) < 2000:
                        diagnosis += '; |'+d
            elif diagnoses_num ==1 :
                if len(diagnosis) < 2000:
                    diagnosis += '; |'+random_diagnoses    
                
        diagnosis_list = list(diagnosis)
        
        diagnosis_token = self.tokenizer.tokenize(diagnosis_list)
        
        tokens_tensors = torch.tensor(diagnosis_token)

        if len(tokens_tensors) > self.clamp_size:
            clamp = random.randint(0,len(tokens_tensors)-self.clamp_size)
            tokens_tensors = tokens_tensors[clamp:clamp+self.clamp_size]

#        src_split, src_isword = split2words(tokens_tensors,rouge=True,tokenize_alphabets=self.tokenizer)
        src_split = split2words(tokens_tensors,
                                iswordfilt=True,
                                tokenize_alphabets=self.tokenizer)
        
        bert_replace_seqence_ = []
        
        for j in range(len(src_split)):
            # use cls represent the whole word
            bert_replace_seqence_.append(torch.tensor([self.leading_token_idx]))
            bert_replace_seqence_.append(tokens_tensors[src_split[j]])                   
#                bert_replace_seqence_.append(torch.tensor([self.padding_token_idx]))
                                  
        tokens_tensors = torch.cat(bert_replace_seqence_)

        if len(tokens_tensors) > self.clamp_size:
            tokens_tensors = tokens_tensors[:self.clamp_size]

        s,t = make_cloze(tokens_tensors,self.tokenizer,percent=0.15, fix=self.fix)
        
        sample = {'src_token':s,
                  'trg':t,
#                  'src':diagnosis_list,
                  }        
        return sample
   
    def __len__(self):
        return self.len

    
class D_stage1_head_token(Dataset):
    def __init__(self,ds,tokenizer,clamp_size,icd10=71704, train=True, fix=False):
        self.len = len(ds)
        self.ds = ds
        self.tokenizer = tokenizer
        self.clamp_size = clamp_size
        self.icd = icd10
        self.train = train
        self.fix = fix
        self.leading_token_idx = self.tokenizer.alphabet2idx['|']
        numbers = torch.arange(self.tokenizer.alphabet2idx['0'],
                               self.tokenizer.alphabet2idx['9']+1)    
        self.numbers_li = list(numbers.numpy())
        
    def is_in_numbers(self,j_1,j0,jr0):
        if j0[0] in self.numbers_li:
            if jr0[0] in self.numbers_li:
                return True
        else:
            if j_1[0] in self.numbers_li:
                return True
        if j0[0] == jr0[0]:
            return True
        if len(j0) == len(jr0):
            return True
        return False    
    
    def __getitem__(self, index):
        diagnosis = self.ds[index]
#        diagnosis = '|'+ diagnosis
        if self.train:
            diagnoses_num = random.randint(8,29)
            random_sequence = torch.randperm(self.len)
            random_diagnoses_idx = random_sequence[:diagnoses_num]
            random_diagnoses = self.ds[random_diagnoses_idx]
            if diagnoses_num>1:
                for d in random_diagnoses:
                    if len(diagnosis) < 3000:
                        diagnosis += '; |'+d
            elif diagnoses_num ==1 :
                if len(diagnosis) < 3000:
                    diagnosis += '; |'+random_diagnoses    
                
        diagnosis_list = list(diagnosis)
        
        diagnosis_token = self.tokenizer.tokenize(diagnosis_list)
        
        tokens_tensors = torch.tensor(diagnosis_token)

        if len(tokens_tensors) > self.clamp_size:
            clamp = random.randint(0,len(tokens_tensors)-self.clamp_size)
            tokens_tensors = tokens_tensors[clamp:clamp+self.clamp_size]

#        s,t = make_cloze(tokens_tensors,self.tokenizer,percent=0.15, fix=self.fix)

        src_split = split2words(tokens_tensors,iswordfilt=True,
                                tokenize_alphabets=self.tokenizer)
        
        bert_origin_seqence_ = []
        bert_replace_seqence_ = []
        bert_cls_label = []
        checkreplace = 0
        lastcheckreplace = checkreplace
        
        for j in range(len(src_split)):
            isreplace = random.random()
            checkreplace = max(0,checkreplace-1)
            if isreplace >= .2 or lastcheckreplace >0:
                # use cls represent the whole word
                bert_cls_label.append(torch.tensor([-1]))
                bert_replace_seqence_.append(torch.tensor([self.leading_token_idx]))
                bert_replace_seqence_.append(tokens_tensors[src_split[j]])                   
            else:
                j_replace = random.choice(range(len(src_split)))
                if j_replace==j:
                    bert_cls_label.append(torch.tensor([-1]))
                    bert_replace_seqence_.append(torch.tensor([self.leading_token_idx]))
                    bert_replace_seqence_.append(tokens_tensors[src_split[j]])
                else:
                    if self.is_in_numbers(tokens_tensors[src_split[max(j-1,0)]],
                                                                   tokens_tensors[src_split[j]],
                                                                   tokens_tensors[src_split[j_replace]]):
                        bert_cls_label.append(torch.tensor([-1]))
                        bert_replace_seqence_.append(torch.tensor([self.leading_token_idx]))
                        bert_replace_seqence_.append(tokens_tensors[src_split[j]])
                    else:
                        checkreplace +=3 
                        if random.random() >0.5:
    #                        if len(bert_cls_label)>0:
    #                            bert_cls_label[-1]=torch.tensor([1])
                            bert_cls_label.append(torch.tensor([1]))
                            bert_replace_seqence_.append(torch.tensor([self.leading_token_idx]))                        
                            bert_replace_seqence_.append(tokens_tensors[src_split[j_replace]])
                            checkreplace += 1
                        else:
    #                        if len(bert_cls_label)>0:
    #                            bert_cls_label[-1]=torch.tensor([0])
                            bert_cls_label.append(torch.tensor([0]))
                            bert_replace_seqence_.append(torch.tensor([self.leading_token_idx]))
                            bert_replace_seqence_.append(tokens_tensors[src_split[j]])                             
            bert_origin_seqence_.append(torch.tensor([self.leading_token_idx]))
            bert_origin_seqence_.append(tokens_tensors[src_split[j]])
            lastcheckreplace = checkreplace
        
        tokens_tensors = torch.cat(bert_replace_seqence_)
        labels_tensors = -1*torch.ones(tokens_tensors.shape)
        
        tokens_ori_tensors = torch.cat(bert_origin_seqence_)
        
        cls_mask = tokens_tensors == self.leading_token_idx
        labels_tensors[cls_mask] = torch.tensor(bert_cls_label,dtype=torch.float)
        
        if len(tokens_tensors) > self.clamp_size:
            tokens_tensors = tokens_tensors[:self.clamp_size]
            labels_tensors = labels_tensors[:self.clamp_size]
            tokens_ori_tensors = tokens_ori_tensors[:self.clamp_size]
       
        sample = {'src_token':tokens_tensors,
                  'trg':labels_tensors,
                  'src_origin':tokens_ori_tensors,
                  }        
        return sample

    
    def __len__(self):
        return self.len
    
class D_stage1_head_longtoken(Dataset):
    def __init__(self,ds,tokenizer,clamp_size,icd10=71704, train=True, fix=False):
        self.len = len(ds)
        self.ds = ds
        self.tokenizer = tokenizer
        self.clamp_size = clamp_size
        self.icd = icd10
        self.train = train
        self.fix = fix
        self.leading_token_idx = self.tokenizer.alphabet2idx['|']
        self.padding_token_idx = self.tokenizer.alphabet2idx[' ']
        
        numbers = torch.arange(self.tokenizer.alphabet2idx['0'],
                               self.tokenizer.alphabet2idx['9']+1)    
        self.numbers_li = list(numbers.numpy())

    def is_in_numbers(self,j_1,j0,jr0):
        if j0[0] in self.numbers_li:
            if jr0[0] in self.numbers_li:
                return True
        else:
            if j_1[0] in self.numbers_li:
                return True
        if j0[0] == jr0[0]:
            return True
        if len(j0) == len(jr0):
            return True
        return False
    
    def __getitem__(self, index):
        diagnosis = self.ds[index]
#        diagnosis = '|'+ diagnosis
        if self.train:
            diagnoses_num = random.randint(10,29)
            random_sequence = torch.randperm(self.len)
            random_diagnoses_idx = random_sequence[:diagnoses_num]
            random_diagnoses = self.ds[random_diagnoses_idx]
            if diagnoses_num>1:
                for d in random_diagnoses:
                    if len(diagnosis) < 5000:
                        diagnosis += '; |'+d
            elif diagnoses_num ==1 :
                if len(diagnosis) < 5000:
                    diagnosis += '; |'+random_diagnoses    
                
        diagnosis_list = list(diagnosis)
        
        diagnosis_token = self.tokenizer.tokenize(diagnosis_list)
        
        tokens_tensors = torch.tensor(diagnosis_token)

        if len(tokens_tensors) > self.clamp_size:
            clamp = random.randint(0,len(tokens_tensors)-self.clamp_size)
            tokens_tensors = tokens_tensors[clamp:clamp+self.clamp_size]

#        s,t = make_cloze(tokens_tensors,self.tokenizer,percent=0.15, fix=self.fix)

        src_split = split2words(tokens_tensors,
                                iswordfilt=True,
                                tokenize_alphabets=self.tokenizer)
        
        bert_origin_seqence_ = []
        bert_replace_seqence_ = []
        bert_cls_label = []
        bert_replaced_dis = []
        
        min_wds = 10
        max_wds = 20
        focus_w_len = random.randint(min_wds,max_wds)
        checkreplace = 0
        lastcheckreplace = checkreplace
        lastfocus_w_len = focus_w_len
        j_total= 0
        
        for j in range(len(src_split)-max_wds):
            jw = (j-j_total) % focus_w_len             
            if jw<1:
                checkreplace = 0
                isreplace = random.random()
                if isreplace >= .99 or lastcheckreplace > 0:
                    # use cls represent the whole word
                    bert_cls_label.append(torch.tensor([-1]))
                    bert_replace_seqence_.append(torch.tensor([self.leading_token_idx]))
                    bert_replace_seqence_.append(tokens_tensors[src_split[j]])
                    bert_replaced_dis.append(torch.tensor([-1]))
                else:
                    j_replace = random.choice(range(len(src_split)-max_wds))
                    if  abs(j_replace-j)<focus_w_len:
                        bert_cls_label.append(torch.tensor([-1]))
                        bert_replace_seqence_.append(torch.tensor([self.leading_token_idx]))
                        bert_replace_seqence_.append(tokens_tensors[src_split[j]])
                        bert_replaced_dis.append(torch.tensor([-1]))
                    else:
                        if self.is_in_numbers(tokens_tensors[src_split[max(j-1,0)]],
                                                                       tokens_tensors[src_split[j]],
                                                                       tokens_tensors[src_split[j_replace]]):
                            bert_cls_label.append(torch.tensor([-1]))
                            bert_replace_seqence_.append(torch.tensor([self.leading_token_idx]))
                            bert_replace_seqence_.append(tokens_tensors[src_split[j]])
                            bert_replaced_dis.append(torch.tensor([-1]))
                        else:                       
                            checkreplace = 1
                            train_len = int(lastfocus_w_len*0.52)
                            if random.random() >0.5:
                                if len(bert_cls_label)>0:
                                    bert_cls_label[-1*train_len:] = [torch.tensor([1])]*train_len
                                    checkreplace = 2
                                    bert_replaced_dis[-1*train_len:] = torch.arange(train_len,0,step=-1)
                                else:
                                    checkreplace = 3
                                bert_cls_label.append(torch.tensor([1]))
                                bert_replace_seqence_.append(torch.tensor([self.leading_token_idx]))                        
                                bert_replace_seqence_.append(tokens_tensors[src_split[j_replace+jw]])
                                bert_replaced_dis.append(torch.tensor([1]))
                            else:
                                if len(bert_cls_label)>0:
                                    bert_cls_label[-1*train_len:] = [torch.tensor([0])]*train_len
                                    bert_replaced_dis[-1*train_len:] = torch.arange(train_len,0,step=-1)
                                bert_cls_label.append(torch.tensor([0]))
                                bert_replace_seqence_.append(torch.tensor([self.leading_token_idx]))
                                bert_replace_seqence_.append(tokens_tensors[src_split[j]])
                                bert_replaced_dis.append(torch.tensor([1]))
                                
                bert_origin_seqence_.append(torch.tensor([self.leading_token_idx]))
                bert_origin_seqence_.append(tokens_tensors[src_split[j]])
                lastcheckreplace = max(checkreplace,lastcheckreplace-1)
                lastfocus_w_len = focus_w_len
            else:
                train_len = int(lastfocus_w_len*0.52)
                if checkreplace==3:
                    bert_cls_label.append(torch.tensor([1]))
                    bert_replace_seqence_.append(torch.tensor([self.leading_token_idx]))
                    bert_replace_seqence_.append(tokens_tensors[src_split[j_replace+jw]])
                    bert_replaced_dis[-1] = torch.tensor([lastfocus_w_len-jw+1])
                    bert_replaced_dis.append(torch.tensor([1]))
                elif checkreplace==2:
                    if jw < train_len:
                        bert_cls_label.append(torch.tensor([1]))
                        bert_replace_seqence_.append(torch.tensor([self.leading_token_idx]))
                        bert_replace_seqence_.append(tokens_tensors[src_split[j_replace+jw]])
                        bert_replaced_dis.append(torch.tensor([1+jw]))
                    else:
                        bert_cls_label.append(torch.tensor([-1]))
                        bert_replace_seqence_.append(torch.tensor([self.leading_token_idx]))
                        bert_replace_seqence_.append(tokens_tensors[src_split[j_replace+jw]])
                        bert_replaced_dis.append(torch.tensor([-1]))
                elif checkreplace==1:
                    if jw < train_len:
                        bert_cls_label.append(torch.tensor([0]))
                        bert_replace_seqence_.append(torch.tensor([self.leading_token_idx]))
                        bert_replace_seqence_.append(tokens_tensors[src_split[j]])
                        bert_replaced_dis.append(torch.tensor([1+jw]))
                    else:
                        bert_cls_label.append(torch.tensor([-1]))
                        bert_replace_seqence_.append(torch.tensor([self.leading_token_idx]))
                        bert_replace_seqence_.append(tokens_tensors[src_split[j]])
                        bert_replaced_dis.append(torch.tensor([-1]))
                else:
                    bert_cls_label.append(torch.tensor([-1]))
                    bert_replace_seqence_.append(torch.tensor([self.leading_token_idx]))
                    bert_replace_seqence_.append(tokens_tensors[src_split[j]])
                    bert_replaced_dis.append(torch.tensor([-1]))
                    
                bert_origin_seqence_.append(torch.tensor([self.padding_token_idx]))
                bert_origin_seqence_.append(tokens_tensors[src_split[j]])
                
                if jw==focus_w_len-1:
                    j_total += focus_w_len
                    focus_w_len = random.randint(min_wds,max_wds)
                
                    
        tokens_tensors = torch.cat(bert_replace_seqence_)
        labels_tensors = -1*torch.ones(tokens_tensors.shape)
        resdis_tensors = -1*torch.ones(tokens_tensors.shape)
        
        tokens_ori_tensors = torch.cat(bert_origin_seqence_)
        
        cls_mask = tokens_tensors == self.leading_token_idx
        labels_tensors[cls_mask] = torch.tensor(bert_cls_label,dtype=torch.float)
        resdis_tensors[cls_mask] = torch.tensor(bert_replaced_dis,dtype=torch.float)
        
        if len(tokens_tensors) > self.clamp_size:
            tokens_tensors = tokens_tensors[:self.clamp_size]
            labels_tensors = labels_tensors[:self.clamp_size]
            tokens_ori_tensors = tokens_ori_tensors[:self.clamp_size]
            resdis_tensors = resdis_tensors[:self.clamp_size]
       
        sample = {'src_token':tokens_tensors,
                  'trg':labels_tensors,
                  'src_origin':tokens_ori_tensors,
                  'dis':resdis_tensors
                  }        
        return sample

    
    def __len__(self):
        return self.len
    
class D_stage1_head_firsttoken(Dataset):
    def __init__(self,ds,tokenizer,clamp_size,icd10=71704, train=True, fix=False):
        self.len = len(ds)
        self.ds = ds
        self.tokenizer = tokenizer
        self.clamp_size = clamp_size
        self.icd = icd10
        self.train = train
        self.fix = fix
        self.leading_token_idx = self.tokenizer.alphabet2idx['|']
        self.padding_token_idx = self.tokenizer.alphabet2idx[' ']
            
    def __getitem__(self, index):
        diagnosis = self.ds[index]

        second_idx = random.choice(range(self.len))
        diagnosis_2 = self.ds[second_idx]
        
        diagnosis_list = list(diagnosis)
        diagnosis_list_2 = list(diagnosis_2)
        
        diagnosis_token = self.tokenizer.tokenize(diagnosis_list)
        diagnosis_token_2 = self.tokenizer.tokenize(diagnosis_list_2)
        
        tokens_tensors = torch.tensor(diagnosis_token)
        tokens_tensors_2 = torch.tensor(diagnosis_token_2)

        if len(tokens_tensors) > self.clamp_size:
            clamp = random.randint(0,len(tokens_tensors)-self.clamp_size)
            tokens_tensors = tokens_tensors[clamp:clamp+self.clamp_size]

        if len(tokens_tensors_2) > self.clamp_size:
            clamp = random.randint(0,len(tokens_tensors_2)-self.clamp_size)
            tokens_tensors_2 = tokens_tensors_2[clamp:clamp+self.clamp_size]

        src_split = split2words(tokens_tensors,
                                iswordfilt=True,
                                tokenize_alphabets=self.tokenizer)

        src_split_2 = split2words(tokens_tensors_2,
                                iswordfilt=True,
                                tokenize_alphabets=self.tokenizer)

        
        bert_origin_seqence_ = []
        bert_replace_seqence_ = []
        bert_cls_label = []
        
#        min_wds = min(int(len(src_split)/2),int(len(src_split_2)/2))
#        max_wds = min(int(len(src_split)/2),int(len(src_split_2)/2))
#        focus_w_len = random.randint(min_wds,max_wds)
        focus_w_len = int(len(src_split)/2)
        
        isreplace = random.random()
        
#        print(int(len(src_split)/2),int(len(src_split_2)/2))
#        print(len(src_split_2)+len(src_split)-2*focus_w_len)
#        print(len(src_split_2),len(src_split),2*focus_w_len,min_wds,max_wds)
        if isreplace >= 0.5:
            for j in range(focus_w_len+int(len(src_split_2)/2)):
                if j < focus_w_len:
                    if j <1:
                        bert_cls_label.append(torch.tensor([1]))
                        bert_replace_seqence_.append(torch.tensor([self.leading_token_idx]))
                        bert_replace_seqence_.append(tokens_tensors[src_split[j]])
                    else:
                        bert_cls_label.append(torch.tensor([-1]))
                        bert_replace_seqence_.append(torch.tensor([self.leading_token_idx]))
                        bert_replace_seqence_.append(tokens_tensors[src_split[j]])
                else:
                    bert_cls_label.append(torch.tensor([-1]))
                    bert_replace_seqence_.append(torch.tensor([self.leading_token_idx]))
                    bert_replace_seqence_.append(tokens_tensors_2[src_split_2[int(len(src_split_2)/2)+j-focus_w_len]])                                 
        else:
            for j in range(len(src_split)):
                if j <1:
                    bert_cls_label.append(torch.tensor([0]))
                    bert_replace_seqence_.append(torch.tensor([self.leading_token_idx]))
                    bert_replace_seqence_.append(tokens_tensors[src_split[j]])
                else:
                    bert_cls_label.append(torch.tensor([-1]))
                    bert_replace_seqence_.append(torch.tensor([self.leading_token_idx]))
                    bert_replace_seqence_.append(tokens_tensors[src_split[j]])
                
 
        for j in range(len(src_split)):
            bert_origin_seqence_.append(torch.tensor([self.padding_token_idx]))
            bert_origin_seqence_.append(tokens_tensors[src_split[j]])             
            
        try:
            tokens_tensors = torch.cat(bert_replace_seqence_)
            tokens_ori_tensors = torch.cat(bert_origin_seqence_)
            labels_tensors = -1*torch.ones(tokens_tensors.shape)
            resdis_tensors = -1*torch.ones(tokens_tensors.shape)
            
            cls_mask = tokens_tensors == self.leading_token_idx
            labels_tensors[cls_mask] = torch.tensor(bert_cls_label,dtype=torch.float)
        except:
            tokens_tensors = torch.zeros([10])
            tokens_ori_tensors = torch.zeros([10])
            labels_tensors = -1*torch.ones(tokens_tensors.shape)
            resdis_tensors = -1*torch.ones(tokens_tensors.shape)
   
        
        if len(tokens_tensors) > self.clamp_size:
            tokens_tensors = tokens_tensors[:self.clamp_size]
            labels_tensors = labels_tensors[:self.clamp_size]
            tokens_ori_tensors = tokens_ori_tensors[:self.clamp_size]
            resdis_tensors = resdis_tensors[:self.clamp_size]
       
        sample = {'src_token':tokens_tensors,
                  'trg':labels_tensors,
                  'src_origin':tokens_ori_tensors,
                  'dis':resdis_tensors
                  }        
        return sample

    
    def __len__(self):
        return self.len
    