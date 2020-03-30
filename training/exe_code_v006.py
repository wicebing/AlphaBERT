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
from alphabert_utils import clean_up_v204_ft, split2words, rouge12l, make_statistics,save_checkpoint,load_checkpoint,clean_up
from alphabert_utils import count_parameters,clean_up_ensemble
from alphabert_dadaloader import alphabet_loaders
from transformers import tokenization_bert, modeling_bert, configuration_bert


batch_size = 1
device = 'cuda'
parallel = True

checkpoint_file = './checkpoint_d2s/new_v402_20191218'

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

def test_alphaBert(DS_model,dloader,threshold=0.5, is_clean_up=True, ep=0, train=False,mean_max='mean',rouge=False):
    if not train:
        DS_model.to(device)
        DS_model = torch.nn.DataParallel(DS_model)
    DS_model.eval()
    
    out_pred_res = []
    
    all_pred_trg = {'pred':[], 'trg':[]}
    rouge_set = []
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(dloader): 
            src = sample['src_token']
            trg = sample['trg']
            att_mask = sample['mask_padding']
            origin_len = sample['origin_seq_length']
    
            src = src.float().to(device)
            trg = trg.float().to(device)
            att_mask = att_mask.float().to(device)
            origin_len = origin_len.to(device)
            
            bs = src.shape
            
            prediction_scores, (pooled_output,head_outputs) = DS_model(input_ids=src,
                                 attention_mask=att_mask)
            pred_prop_bin = pooled_output[0].view(*bs,-1)
            
            if is_clean_up:
                pred_prop = clean_up_v204_ft(src,pred_prop_bin,
                                             tokenize_alphabets=tokenize_alphabets,
                                             mean_max=mean_max)
            else:                
                pred_prop_value, pred_prop = pred_prop_bin.max(dim=2)
                pred_prop = pred_prop.float()

            pred_selected = pred_prop > threshold
            trg_selected = trg > threshold
            
            for i, src_ in enumerate(src):                                
                if rouge:
                    src_split, src_isword = split2words(src_,
                                                        tokenize_alphabets=tokenize_alphabets,
                                                        rouge=rouge)
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
                    
                    all_pred_trg['trg'].append(torch.tensor(trg_))
                    all_pred_trg['pred'].append(torch.tensor(pred_))  
                     
                    a_ = tokenize_alphabets.convert_idx2str(src_[:origin_len[i]])
                    s_ = ''.join(i+' ' for i in hypothesis)
                    t_ = ''.join(i+' ' for i in referecne)
                else:                                
                    all_pred_trg['pred'].append(pred_prop[i][:origin_len[i]].cpu())
                    all_pred_trg['trg'].append(trg[i][:origin_len[i]].cpu())

                    a_ = tokenize_alphabets.convert_idx2str(src_[:origin_len[i]])
                    s_ = tokenize_alphabets.convert_idx2str(src_[pred_selected[i]])
                    t_ = tokenize_alphabets.convert_idx2str(src_[trg_selected[i]])
                
                out_pred_res.append((a_,s_,t_,pred_prop[i]))
            print(batch_idx, len(dloader))
    out_pd_res = pd.DataFrame(out_pred_res)
    out_pd_res.to_csv('./iou_pic/test_pred.csv', sep=',')
    
    make_statistics(all_pred_trg, ep=ep)

    DS_model.train()
    
    if rouge:
        rouge_res = rouge12l(rouge_set)
        rouge_res_pd = pd.DataFrame(rouge_res)
        rouge_res_pd.to_csv('./iou_pic/rouge_res.csv',index=False)
        rouge_res_np = np.array(rouge_res_pd)
        
        pd.DataFrame(rouge_res_np.mean(axis =0)).to_csv('./iou_pic/rouge_res_mean.csv',index=False)

def train_alphaBert(DS_model,dloader,lr=1e-4,epoch=10,log_interval=20):
    global checkpoint_file
    DS_model.to(device)
    model_optimizer = optim.Adam(DS_model.parameters(), lr=lr)
    DS_model = torch.nn.DataParallel(DS_model)
    DS_model.train()
    
#    criterion = nn.MSELoss().to(device)
#    criterion = alphabert_loss_v02.Alphabert_loss(device=device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    iteration = 0
    total_loss = []
    for ep in range(epoch):
        DS_model.train()
        
        t0 = time.time()
#        step_loss = 0
        epoch_loss = 0
        epoch_cases =0
        for batch_idx, sample in enumerate(dloader):        
            model_optimizer.zero_grad()
            loss = 0
            
            src = sample['src_token']
            trg = sample['trg']
            att_mask = sample['mask_padding']
            origin_len = sample['origin_seq_length']
            
            bs = len(src)
            
            src = src.float().to(device)
            trg = trg.long().to(device)
            att_mask = att_mask.float().to(device)
            origin_len = origin_len.to(device)

            prediction_scores, (pooled_output,head_outputs) = DS_model(input_ids=src,
                               attention_mask=att_mask)
        
            pred_prop = pooled_output[0]

            loss = criterion(pred_prop,trg.view(-1).contiguous())
            
            loss.backward()
            model_optimizer.step()
                        
            with torch.no_grad():
                epoch_loss += loss.item()*bs
                epoch_cases += bs
                
            if iteration % log_interval == 0:
#                step_loss.backward()
#                model_optimizer.step()
#                print('+++ update +++')
                print('Ep:{} [{} ({:.0f}%)/ ep_time:{:.0f}min] L:{:.4f}'.format(
                        ep, batch_idx * batch_size,
                        100. * batch_idx / len(dloader),
                        (time.time()-t0)*len(dloader)/(60*(batch_idx+1)),
                        loss.item()))
#                print(0,st_target)
#                step_loss = 0
                
            if iteration % 400 == 0:
                save_checkpoint(checkpoint_file,'d2s.pth',DS_model,model_optimizer,parallel=parallel)
                   
            iteration +=1
        if ep % 1 ==0:
            save_checkpoint(checkpoint_file,'d2s.pth',DS_model,model_optimizer,parallel=parallel)
#            test_alphaBert(DS_model,D2S_valloader, 
#                           is_clean_up=True, ep=ep,train=True)

            print('======= epoch:%i ========'%ep)
            
#        print('total loss: {:.4f}'.format(total_loss/len(dloader)))         
        print('++ Ep Time: {:.1f} Secs ++'.format(time.time()-t0)) 
#        total_loss.append(epoch_loss)
        total_loss.append(float(epoch_loss/epoch_cases))
        pd_total_loss = pd.DataFrame(total_loss)
        pd_total_loss.to_csv('./iou_pic/total_loss_finetune.csv', sep = ',')
    print(total_loss)



def test_alphaBert_stage1(TS_model,dloader,cloze_fix=False,train=False,):
    TS_model.eval()
    out_pred_res = []
    with torch.no_grad():
        for batch_idx, sample in enumerate(dloader):        
            src = sample['src_token']
            trg = sample['trg']
            att_mask = sample['mask_padding']
            origin_len = sample['origin_seq_length']
            
            bs, max_len = src.shape
            
#            src, err_cloze = make_cloze(src,
#                                        max_len,
#                                        device=device,
#                                        percent=0.15,
#                                        fix=cloze_fix)
        
            src = src.float().to(device)
            trg = trg.long().to(device)
            att_mask = att_mask.float().to(device)
            origin_len = origin_len.to(device)
        
            prediction_scores, = TS_model(input_ids=src,
                                                    attention_mask=att_mask)
                
            a_ = tokenize_alphabets.convert_idx2str(src[0][:origin_len[0]])
            print(a_)
            print(' ******** ******** ******** ')
            _, show_pred = torch.max(prediction_scores[0],dim = 1)
            err_cloze_ = trg[0] > -1
            src[0][err_cloze_] = show_pred[err_cloze_].float()
            b_ = tokenize_alphabets.convert_idx2str(src[0][:origin_len[0]])
            print(b_)
            print(' ******** ******** ******** ')
            src[0][err_cloze_] = trg[0][err_cloze_].float()
            c_ = tokenize_alphabets.convert_idx2str(src[0][:origin_len[0]])
            print(c_)
            
            out_pred_res.append((a_,b_,c_,err_cloze_))
            
#            out_pd_res = pd.DataFrame(out_pred_res)
#            out_pd_res.to_csv('out_pred_test.csv', sep=',')
                
            TS_model.train()
            
            return (a_,b_,c_,err_cloze_)
    
def train_alphaBert_stage1(TS_model,dloader,testloader,lr=1e-4,epoch=10,log_interval=20,
                           cloze_fix=True, use_amp=False, parallel=True):
    global checkpoint_file
    TS_model.to(device)
    model_optimizer = optim.Adam(TS_model.parameters(), lr=lr)
    if use_amp:
        TS_model, model_optimizer = amp.initialize(TS_model, model_optimizer, opt_level="O1")
    if parallel:
        TS_model = torch.nn.DataParallel(TS_model)

#    torch.distributed.init_process_group(backend='nccl',
#                                         init_method='env://host',
#                                         world_size=0,
#                                         rank=0,
#                                         store=None,
#                                         group_name='')
#    TS_model = DDP(TS_model)
#    TS_model = apex.parallel.DistributedDataParallel(TS_model)
    TS_model.train()
        
#    criterion = alphabert_loss.Alphabert_satge1_loss(device=device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    iteration = 0
    total_loss = []    
    out_pred_res = []
    out_pred_test = []
    for ep in range(epoch):
        t0 = time.time()
#        step_loss = 0
        epoch_loss = 0
        epoch_cases = 0
        for batch_idx, sample in enumerate(dloader):
#            TS_model.train()
            model_optimizer.zero_grad()
            loss = 0
            
            src = sample['src_token']
            trg = sample['trg']
            att_mask = sample['mask_padding']
            origin_len = sample['origin_seq_length']
            
            bs, max_len = src.shape
            
#            src, err_cloze = make_cloze(src,
#                                        max_len,
#                                        device=device,
#                                        percent=0.15,
#                                        fix=cloze_fix)
        
            src = src.float().to(device)
            trg = trg.long().to(device)
            att_mask = att_mask.float().to(device)
            origin_len = origin_len.to(device)
        
            prediction_scores, = TS_model(input_ids=src,
                                                    attention_mask=att_mask)
            
#            print(1111,prediction_scores.view(-1,84).shape)
#            print(1111,trg.view(-1).shape)
            
            loss = criterion(prediction_scores.view(-1,100).contiguous(),trg.view(-1).contiguous())
                     
            if use_amp:
                with amp.scale_loss(loss, model_optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
                
            model_optimizer.step()
                        
            with torch.no_grad():
                epoch_loss += loss.item()*bs
                epoch_cases += bs
            
                if iteration % log_interval == 0:
                    print('Ep:{} [{} ({:.0f}%)/ ep_time:{:.0f}min] L:{:.4f}'.format(
                            ep, batch_idx * batch_size,
                            100. * batch_idx / len(dloader),
                            (time.time()-t0)*len(dloader)/(60*(batch_idx+1)),
                            loss.item()))
                    
                if iteration % 400 == 0:
                    save_checkpoint(checkpoint_file,'DS_pretrain.pth',TS_model,model_optimizer,parallel=parallel)
                    a_ = tokenize_alphabets.convert_idx2str(src[0][:origin_len[0]])
                    print(a_)
                    print(' ******** ******** ******** ')
                    _, show_pred = torch.max(prediction_scores[0],dim = 1)
                    err_cloze_ = trg[0] > -1
                    src[0][err_cloze_] = show_pred[err_cloze_].float()
                    b_ = tokenize_alphabets.convert_idx2str(src[0][:origin_len[0]])
                    print(b_)
                    print(' ******** ******** ******** ')
                    src[0][err_cloze_] = trg[0][err_cloze_].float()
                    c_ = tokenize_alphabets.convert_idx2str(src[0][:origin_len[0]])
                    print(c_)
                    
                    out_pred_res.append((ep,a_,b_,c_,err_cloze_))
                    out_pd_res = pd.DataFrame(out_pred_res)
                    out_pd_res.to_csv('./result/out_pred_train.csv', sep=',')
                
                if iteration % 999 == 0:
                    print(' ===== Show the Test of Pretrain ===== ')
                    test_res = test_alphaBert_stage1(TS_model,testloader)
                    print(' ===== Show the Test of Pretrain ===== ')
                    
                    out_pred_test.append((ep,*test_res))                    
                    out_pd_test = pd.DataFrame(out_pred_test)
                    out_pd_test.to_csv('./result/out_pred_test.csv', sep=',')
                                
            iteration +=1
        if ep % 1 ==0:
            save_checkpoint(checkpoint_file,'DS_pretrain.pth',TS_model,model_optimizer,parallel=parallel)

            print('======= epoch:%i ========'%ep)
            
        print('++ Ep Time: {:.1f} Secs ++'.format(time.time()-t0)) 
        total_loss.append(float(epoch_loss/epoch_cases))
        pd_total_loss = pd.DataFrame(total_loss)
        pd_total_loss.to_csv('./result/total_loss_pretrain.csv', sep = ',')
    print(total_loss)

def test_alphaBert_stage1_head(TS_model,dloader,cloze_fix=False,train=False,):
    TS_model.eval()
    with torch.no_grad():
        for batch_idx, sample in enumerate(dloader):        
            src = sample['src_token']
            trg = sample['trg']
            att_mask = sample['mask_padding']
            origin_len = sample['origin_seq_length']
            src_ori = sample['src_origin']
            
            bs, max_len = src.shape
                
            src = src.float().to(device)
            trg = trg.long().to(device)
            att_mask = att_mask.float().to(device)
            origin_len = origin_len.to(device)
        
            head_outputs, = TS_model(input_ids=src,
                               attention_mask=att_mask,
                               out='head')

            head_outputs = head_outputs.view(bs,max_len,-1)
            show_pred = nn.Softmax(dim=-1)(head_outputs[0])
            head_mask = src[0] == tokenize_alphabets.alphabet2idx['|']
            
            a_ = tokenize_alphabets.convert_idx2str_head(src[0][:origin_len[0]],
                                                         head_mask,
                                                         show_pred[:,1],
                                                         trg[0])
            print(a_)
            print(' ******** ******** ******** ')
            b_ = tokenize_alphabets.convert_idx2str(src_ori[0][:origin_len[0]],head=True)
            print(b_)
                
            TS_model.train()
            
            return (a_,b_)

def train_alphaBert_stage1_head(TS_model,dloader,lr=1e-4,epoch=10,log_interval=20,
                                cloze_fix=True, use_amp=False, parallel=True):
    global checkpoint_file
    TS_model.to(device)
    model_optimizer = optim.Adam(TS_model.parameters(), lr=lr)
    if use_amp:
        TS_model, model_optimizer = amp.initialize(TS_model, model_optimizer, opt_level="O1")
    if parallel:
        TS_model = torch.nn.DataParallel(TS_model)

    TS_model.train()
        
    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    iteration = 0
    total_loss = []    
    out_pred_res = []
    out_pred_test = []
    for ep in range(epoch):
        t0 = time.time()
#        step_loss = 0
        epoch_loss = 0
        epoch_cases = 0
        for batch_idx, sample in enumerate(dloader):
#            TS_model.train()
            model_optimizer.zero_grad()
            loss = 0
            
            src = sample['src_token']
            trg = sample['trg']
            att_mask = sample['mask_padding']
            origin_len = sample['origin_seq_length']
            src_ori = sample['src_origin']
            
            bs, max_len = src.shape
        
            src = src.float().to(device)
            trg = trg.long().to(device)
            att_mask = att_mask.float().to(device)
            origin_len = origin_len.to(device)
        
            head_outputs, = TS_model(input_ids=src,
                                      attention_mask=att_mask,
                                      out='head')
#            head_outputs = outputs[0]
            loss = criterion(head_outputs,trg.view(-1).contiguous())
                                 
            if use_amp:
                with amp.scale_loss(loss, model_optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
                
            model_optimizer.step()
                        
            with torch.no_grad():
                epoch_loss += loss.item()*bs
                epoch_cases += bs
            
                if iteration % log_interval == 0:
                    print('Ep:{} [{} ({:.0f}%)/ ep_time:{:.0f}min] L:{:.4f}'.format(
                            ep, batch_idx * batch_size,
                            100. * batch_idx / len(dloader),
                            (time.time()-t0)*len(dloader)/(60*(batch_idx+1)),
                            loss.item()))
                    
                if iteration % 400 == 0:
                    save_checkpoint(checkpoint_file,'DS_pretrain_head.pth',TS_model,model_optimizer,parallel=parallel)
                    
#                    head_outputs = head_outputs.view(bs,max_len,-1)
                    show_pred = nn.Softmax(dim=-1)(head_outputs.view(bs,max_len,-1)[0])
                    head_mask = src[0] == tokenize_alphabets.alphabet2idx['|']
                    
                    a_ = tokenize_alphabets.convert_idx2str_head(src[0][:origin_len[0]],
                                                                 head_mask,
                                                                 show_pred[:,1],
                                                                 trg[0])
                    print(' ******** ******** ******** ')
                    print(a_)
                    print(' ******** ******** ******** ')
                    b_ = tokenize_alphabets.convert_idx2str(src_ori[0][:origin_len[0]],head=True)
                    print(b_)
                    
                    out_pred_res.append((ep,a_,b_))
                    out_pd_res = pd.DataFrame(out_pred_res)
                    out_pd_res.to_csv('./result/out_pred_head_train.csv', sep=',')
                
                if iteration % 999 == 0:
                    print(' ===== Show the Test of Pretrain ===== ')
#                    test_res = test_alphaBert_stage1_head(TS_model,stage1_head_dataloader_test)
#                    print(' ===== Show the Test of Pretrain ===== ')
#                    
#                    out_pred_test.append((ep,*test_res))                    
#                    out_pd_test = pd.DataFrame(out_pred_test)
#                    out_pd_test.to_csv('./result/out_pred_head_test.csv', sep=',')
                                
            iteration +=1
        if ep % 1 ==0:
            save_checkpoint(checkpoint_file,'DS_pretrain_head.pth',TS_model,model_optimizer,parallel=parallel)

            print('======= epoch:%i ========'%ep)
            
        print('++ Ep Time: {:.1f} Secs ++'.format(time.time()-t0)) 
        total_loss.append(float(epoch_loss/epoch_cases))
        pd_total_loss = pd.DataFrame(total_loss)
        pd_total_loss.to_csv('./result/total_loss_pretrain_head.csv', sep = ',')
    print(total_loss)


def test_alphaBert_head_2(TS_model,dloader,cloze_fix=False,train=False,):
    TS_model.eval()
    with torch.no_grad():
        for batch_idx, sample in enumerate(dloader):        
            src = sample['src_token']
            trg = sample['trg']
            att_mask = sample['mask_padding']
            origin_len = sample['origin_seq_length']
            
            bs, max_len = src.shape
                
            src = src.float().to(device)
            trg = trg.long().to(device)
            att_mask = att_mask.float().to(device)
            origin_len = origin_len.to(device)
        
            head_outputs, = TS_model(input_ids=src,
                                     attention_mask=att_mask,
                                     out='finehead')

            head_outputs = head_outputs.view(bs,max_len,-1)
            show_pred = nn.Softmax(dim=-1)(head_outputs[0])
            head_mask = src[0] == tokenize_alphabets.alphabet2idx['|']
            
            a_ = tokenize_alphabets.convert_idx2str_head(src[0][:origin_len[0]],
                                                         head_mask,
                                                         show_pred[:,1],
                                                         trg[0])
            print(a_)
            print(' ******** ******** ******** ')
                
            TS_model.train()
            
            return (a_)

def test_alphaBert_head(DS_model,dloader,threshold=0.5, is_clean_up=True, ep=0, train=False,mean_max='mean',rouge=False):
    if not train:
        DS_model.to(device)
        DS_model = torch.nn.DataParallel(DS_model)
    DS_model.eval()
    
    out_pred_res = []        
    all_pred_trg = {'pred':[], 'trg':[]}
    rouge_set = []
    leading_token_idx = tokenize_alphabets.alphabet2idx['|']
    padding_token_idx = tokenize_alphabets.alphabet2idx[' ']
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(dloader): 
            src = sample['src_token']
            trg = sample['trg']
            att_mask = sample['mask_padding']
            origin_len = sample['origin_seq_length']
    
            src = src.float().to(device)
            trg = trg.float().to(device)
            att_mask = att_mask.float().to(device)
            origin_len = origin_len.to(device)
            
            bs = src.shape
            
            pred_prop_bin, = DS_model(input_ids=src,
                                 attention_mask=att_mask,
                                 out = 'finehead')

#            max_pred_prop, pred_prop = pred_prop_bin.view(*bs,-1).max(dim=2)   
            cls_idx = src==leading_token_idx
            pred_prop_bin_softmax = nn.Softmax(dim=-1)(pred_prop_bin.view(*bs,-1))
            pred_prop = pred_prop_bin_softmax[:,:,1]
                
            pred_selected = pred_prop > threshold
            trg_selected = trg > threshold      
            
            for i, src_ in enumerate(src):
                all_pred_trg['pred'].append(pred_prop[i][cls_idx[i]].cpu())
                all_pred_trg['trg'].append(trg[i][cls_idx[i]].cpu())  
                                
                if rouge:
#                    src_split, src_isword = split2words(src_,rouge=rouge)
                    referecne = []
                    hypothesis = []
                    isselect_pred = False
                    isselect_trg = False

                    for j, wp in enumerate(src_):
                        if wp == leading_token_idx:
                            if pred_prop[i][j]  > threshold:
                                if isselect_pred:
                                    hypothesis.append(padding_token_idx)                                  
                                isselect_pred = True
                            else:                                 
                                if isselect_pred:
                                    hypothesis.append(padding_token_idx)
                                isselect_pred = False
                            if trg[i][j]  > 0:
                                if isselect_trg:
                                    referecne.append(padding_token_idx)
                                isselect_trg = True
                            else:
                                if isselect_trg:
                                    referecne.append(padding_token_idx)
                                isselect_trg = False                                
                        else:
                            if isselect_pred:
                                if wp != leading_token_idx:
                                    hypothesis.append(wp.item())
                            if isselect_trg:
                                if wp != leading_token_idx:
                                    referecne.append(wp.item())                    
                    hypothesis = tokenize_alphabets.convert_idx2str(hypothesis)
                    referecne = tokenize_alphabets.convert_idx2str(referecne)
                    
                    hypothesis_list = hypothesis.split()
                    referecne_list = referecne.split()

                    rouge_set.append((hypothesis_list,referecne_list))
                    
                    a_ = tokenize_alphabets.convert_idx2str(src_[:origin_len[i]])
                    s_ = hypothesis
                    t_ = referecne
                else:                                
                    a_ = tokenize_alphabets.convert_idx2str(src_[:origin_len[i]])
                    s_ = tokenize_alphabets.convert_idx2str(src_[pred_selected[i]])
                    t_ = tokenize_alphabets.convert_idx2str(src_[trg_selected[i]])
           
                out_pred_res.append((a_,s_,t_))
            print(batch_idx, len(dloader))
    out_pd_res = pd.DataFrame(out_pred_res)
    out_pd_res.to_csv('./iou_pic/test_pred.csv', sep=',')

    if not train:    
        make_statistics(all_pred_trg, ep=ep)
        if rouge:
            rouge_res = rouge12l(rouge_set)
            rouge_res_pd = pd.DataFrame(rouge_res)
            rouge_res_pd.to_csv('./iou_pic/rouge_res.csv',index=False)
            rouge_res_np = np.array(rouge_res_pd)
            
            pd.DataFrame(rouge_res_np.mean(axis =0)).to_csv('./iou_pic/rouge_res_mean.csv',index=False)
    DS_model.train()
    

def train_alphaBert_head(TS_model,dloader,lr=1e-4,epoch=10,log_interval=20,
                                cloze_fix=True, use_amp=False, parallel=True):
    global checkpoint_file
    TS_model.to(device)
    model_optimizer = optim.Adam(TS_model.parameters(), lr=lr)
    if use_amp:
        TS_model, model_optimizer = amp.initialize(TS_model, model_optimizer, opt_level="O1")
    if parallel:
        TS_model = torch.nn.DataParallel(TS_model)

    TS_model.train()
        
    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    iteration = 0
    total_loss = []    
    out_pred_res = []
    out_pred_test = []
    for ep in range(epoch):
        t0 = time.time()
#        step_loss = 0
        epoch_loss = 0
        epoch_cases = 0
        for batch_idx, sample in enumerate(dloader):
#            TS_model.train()
            model_optimizer.zero_grad()
            loss = 0
            
            src = sample['src_token']
            trg = sample['trg']
            att_mask = sample['mask_padding']
            origin_len = sample['origin_seq_length']
            
            bs, max_len = src.shape
        
            src = src.float().to(device)
            trg = trg.long().to(device)
            att_mask = att_mask.float().to(device)
            origin_len = origin_len.to(device)
        
            head_outputs, = TS_model(input_ids=src,
                                      attention_mask=att_mask,
                                      out='finehead')
#            head_outputs = outputs[0]
            trg_view = trg.view(-1).contiguous()
            trg_mask0 = trg_view == 0
            trg_mask1 = trg_view == 1
            
            loss0 = criterion(head_outputs[trg_mask0],trg_view[trg_mask0])
            loss1 = criterion(head_outputs[trg_mask1],trg_view[trg_mask1])
            
            loss = loss0+loss1
                                 
            if use_amp:
                with amp.scale_loss(loss, model_optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
                
            model_optimizer.step()
                        
            with torch.no_grad():
                epoch_loss += loss.item()*bs
                epoch_cases += bs
            
                if iteration % log_interval == 0:
                    print('Ep:{} [{} ({:.0f}%)/ ep_time:{:.0f}min] L:{:.4f} L0:{:.4f} L1:{:.4f}'.format(
                            ep, batch_idx * batch_size,
                            100. * batch_idx / len(dloader),
                            (time.time()-t0)*len(dloader)/(60*(batch_idx+1)),
                            loss.item(),loss0.item(),loss1.item()))
                    
                if iteration % 400 == 0:
                    save_checkpoint(checkpoint_file,'d2s_head.pth',TS_model,model_optimizer,parallel=parallel)
                    
#                    head_outputs = head_outputs.view(bs,max_len,-1)
                    show_pred = nn.Softmax(dim=-1)(head_outputs.view(bs,max_len,-1)[0])
                    head_mask = src[0] == tokenize_alphabets.alphabet2idx['|']
                    
                    a_ = tokenize_alphabets.convert_idx2str_head(src[0][:origin_len[0]],
                                                                 head_mask,
                                                                 show_pred[:,1],
                                                                 trg[0])
                    print(' ******** ******** ******** ')
                    print(a_)
                    print(' ******** ******** ******** ')
#                    b_ = tokenize_alphabets.convert_idx2str(src_ori[0][:origin_len[0]],head=True)
#                    print(b_)
                    
                    out_pred_res.append((ep,a_))
                    out_pd_res = pd.DataFrame(out_pred_res)
                    out_pd_res.to_csv('./result/out_fine_head_train.csv', sep=',')
                
#                if iteration % 999 == 0:
#                    print(' ===== Show the Test of Pretrain ===== ')
#                    test_res = test_alphaBert_stage1_head(TS_model,stage1_head_dataloader_test)
#                    print(' ===== Show the Test of Pretrain ===== ')
#                    
#                    out_pred_test.append((ep,*test_res))                    
#                    out_pd_test = pd.DataFrame(out_pred_test)
#                    out_pd_test.to_csv('./result/out_fine_head_test.csv', sep=',')
                                
            iteration +=1
        if ep % 1 ==0:
            save_checkpoint(checkpoint_file,'d2s_head.pth',TS_model,model_optimizer,parallel=parallel)

            print('======= epoch:%i ========'%ep)
            
        print('++ Ep Time: {:.1f} Secs ++'.format(time.time()-t0)) 
        total_loss.append(float(epoch_loss/epoch_cases))
        pd_total_loss = pd.DataFrame(total_loss)
        pd_total_loss.to_csv('./result/total_loss_finetune_head.csv', sep = ',')
    print(total_loss)
    
    

try:
    task = sys.argv[1]
    train_val_test = sys.argv[2] 
    print('***** task = ',task)
    print('***** mode = ',train_val_test)
except:
    task = 'test'
    train_val_test = 'test'

if task == 'pretrain':
    d2s_satge1_model = alphaBERT_model.alphaBertForMaskedLM(config)   
    try:      
        d2s_satge1_model = load_checkpoint(checkpoint_file,'DS_pretrain.pth',d2s_satge1_model)
    except:
        print('*** No Pretrain_Model ***')
        pass
    loader = alphaloader.make_loaders(finetune=False,head=False,ahead=True,pretrain=True,trvte='test')
    stage1_dataloader = loader['stage1_dataloader']
    stage1_test_dataloader = loader['stage1_dataloader_test']
    train_alphaBert_stage1(d2s_satge1_model,
                           stage1_dataloader,
                           stage1_test_dataloader,
                           lr=1e-5,
                           epoch=50,
                           log_interval=5,
                           cloze_fix=False, 
                           use_amp=False,
                           parallel=parallel)

alphaloader = alphabet_loaders(datapath=filepath, 
                          config=config, 
                          tokenize_alphabets=tokenize_alphabets,
                          num_workers=4,
                          batch_size=batch_size)

if task == 'head':
    d2s_head_model = alphaBERT_model.alphaBertForMaskedLM(config)   
    try:      
        d2s_head_model = load_checkpoint(checkpoint_file,'DS_pretrain_head.pth',d2s_head_model)
    except:
        print('*** No Pretrain_Model ***')
        d2s_head_model = load_checkpoint(checkpoint_file,'DS_pretrain.pth',d2s_head_model)
        pass
    loader = alphaloader.make_loaders(finetune=False,head=True,ahead=False,pretrain=True,trvte='test')
    stage1_head_dataloader = loader['stage1_head_dataloader']
    train_alphaBert_stage1_head(d2s_head_model,stage1_head_dataloader,
                           lr=1e-5,
                           epoch=50,
                           log_interval=5,
                           cloze_fix=False, 
                           use_amp=False,
                           parallel=parallel)

if task == 'finetune':
    d2s_Model = alphaBERT_model.alphaBertForMaskedLM(config)   
    try:      
        d2s_Model = load_checkpoint(checkpoint_file,'d2s.pth',d2s_Model)
    except:
        print('*** No Pretrain_Model ***')
        d2s_Model = load_checkpoint(checkpoint_file,'DS_pretrain.pth',d2s_Model)
        pass

    loader = alphaloader.make_loaders(finetune=True,head=False,ahead=False,pretrain=False,trvte=train_val_test)
    if train_val_test == 'train':
        D2S_trainloader = loader['D2S_trainloader']
        train_alphaBert(d2s_Model,D2S_trainloader,lr=1e-5,epoch=100,log_interval=3)
        
    elif train_val_test == 'val':
        D2S_valloader = loader['D2S_valloader']
        test_alphaBert(d2s_Model,D2S_valloader,threshold=0.5,
                       is_clean_up=True,
                       ep='f1_mean2',
                       mean_max='mean',
                       rouge=True)
        
    elif train_val_test == 'test':
        D2S_testloader = loader['D2S_testloader']
        test_alphaBert(d2s_Model,
                       D2S_testloader,
                       threshold=0.54, 
                       is_clean_up=True, 
                       ep='f1_test',
                       mean_max='mean',
                       rouge=True)
        
if task == 'finetune_head':
    d2s_Model = alphaBERT_model.alphaBertForMaskedLM(config)   
    try:      
        d2s_Model = load_checkpoint(checkpoint_file,'d2s_head.pth',d2s_Model)
    except:
        print('*** No Pretrain_Model ***')
        d2s_Model = load_checkpoint(checkpoint_file,'DS_pretrain_head.pth',d2s_Model)
        pass

    loader = alphaloader.make_loaders(finetune=True,head=True,ahead=False,pretrain=False,trvte=train_val_test)
    if train_val_test == 'train':
        D2S_head_trainloader = loader['D2S_head_trainloader']
        train_alphaBert_head(d2s_Model,D2S_head_trainloader,lr=1e-5,epoch=100,log_interval=3)
        
    elif train_val_test == 'val':
        D2S_head_valloader = loader['D2S_head_valloader']
        test_alphaBert_head(d2s_Model,D2S_head_valloader,
                            threshold=0.8,
                            is_clean_up=True,
                            ep='head_mean_val',
                            mean_max='mean',
                            rouge=True)
        
    elif train_val_test == 'test':
        D2S_head_testloader = loader['D2S_head_testloader']
        test_alphaBert_head(d2s_Model,
                            D2S_head_testloader,
                            threshold=0.54, 
                            is_clean_up=True, 
                            ep='head_mean_test',
                            mean_max='mean',
                            rouge=True)


class test_loaders():
    def __init__(self, datapath, config, tokenize_alphabets, num_workers=4, batch_size=2):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.config = config
        self.tokenize_alphabets = tokenize_alphabets
        self.filepath = datapath
        self.clamp_size = config['max_position_embeddings']
       
    def get_data_np(self):
        filename1a = os.path.join(self.filepath,'DD_test_lstm.csv')
        data_a = pd.read_csv(filename1a, encoding='utf-8')
        D_L_test = np.array(data_a)[:,[1,3]]
        return D_L_test,data_a
    
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
                           num_workers=1,
                           collate_fn=alphabert_dataset.collate_fn)

#leading_token_idx = tokenize_alphabets.alphabet2idx['|']
#padding_token_idx = tokenize_alphabets.alphabet2idx[' ']
out_pred_res = []

out_a = []
out_b = []
out_c = []
out_d = []

#device = 'cpu'
modelpath= os.path.join(filepath,'W_d2s_total_0302_long.pth')
DS_model = torch.load(modelpath)
DS_model.to(device)
#DS_model = torch.nn.DataParallel(DS_model)
DS_model.eval()
threshold = 0.52
#import bert_cls_Model
#DS_model = bert_cls_Model.bert_baseModel(bioBERT=False)
#DS_model = load_checkpoint(filepath,'bert_pretrain.pth', DS_model)
#DS_model.to(device)
#DS_model = torch.nn.DataParallel(DS_model)
#DS_model.eval()

#import gruModel
#DS_model = gruModel.LSTM_baseModel(config)
#DS_model = load_checkpoint(filepath,'lstm_pretrain.pth',DS_model)
#DS_model.to(device)
#DS_model.eval()


dloader = D2S_testloader
    
out_pred_res = []
out_trg_res = []

all_pred_trg = {'pred':[], 'trg':[]}

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
            
            all_pred_trg['trg'].append(torch.tensor(trg_))
            all_pred_trg['pred'].append(torch.tensor(pred_))  
             
            a_ = tokenize_alphabets.convert_idx2str(src_[:origin_len[i]])
            s_ = ''.join(i+' ' for i in hypothesis)
            t_ = ''.join(i+' ' for i in referecne)

        out_pred_res.append(s_)
        out_trg_res.append(t_)              
 
out_np_res = np.array(out_pred_res).reshape(-1,1)   
out_pred_BERT = pd.DataFrame(out_pred_res)

out_np_trg = np.array(out_trg_res).reshape(-1,1)   
out_trg_BERT = pd.DataFrame(out_np_trg)

#data_pd['BERT'] = out_np_BERT
data_pd['ours'] = out_pred_BERT
#data_pd['ours'] = out_trg_BERT

data_pd.to_csv('DD_test_ours_0302_long.csv',index=None)

#aaaa.to_csv('DD_test_BERT.csv',index=None)
#
# import sqlite3

# conn = sqlite3.connect('DD_proposals_ours_0113.sqlite')

# data_pd.to_sql('DD_proposals',conn,if_exists='replace', index=False)
