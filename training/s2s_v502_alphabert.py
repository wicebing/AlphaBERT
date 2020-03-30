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
import dg_alphabets
import bert_unet_001
import alphabert_loss_v02

import alphabert_dataset_v05 as alphabert_dataset
import alphaBERT_model
from alphabert_utils import clean_up_v204_ft, split2words, rouge12l, make_statistics,save_checkpoint
from alphabert_utils import count_parameters,clean_up_ensemble
from alphabert_dadaloader import alphabet_loaders


batch_size = 4
device = 'cuda'
parallel = True

checkpoint_file = './checkpoint_d2s/best_v501'
filepath = './data'

all_expect_alphabets = [' ', '#', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.',
 '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?',
 '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '^', 'a', 'b', 'c', 'd', 'e', 'f',
 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
 'x', 'y', 'z', '~']

tokenize_alphabets = dg_alphabets.Diagnosis_alphabets()
for a in all_expect_alphabets:
    tokenize_alphabets.addalphabet(a)

config = {'hidden_size': 64,
          'max_position_embeddings':1350,
          'eps': 1e-7,
          'input_size': tokenize_alphabets.n_alphabets, 
          'vocab_size': tokenize_alphabets.n_alphabets, 
          'hidden_dropout_prob': 0.1,
          'num_attention_heads': 8, 
          'attention_probs_dropout_prob': 0.1,
          'intermediate_size': 256,
          'num_hidden_layers': 16,
          }
    
def load_checkpoint(checkpoint_path, model, parallel=True):
    state = torch.load(os.path.join('./checkpoint_d2s/best_501',checkpoint_path),
#                       map_location={'cuda:0':'cuda:1'}
                       )
    if parallel:
        try:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state['state_dict'].items():
                name = k[7:] # remove module.
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            print(' === load from parallel model === ')
        except:
            model.load_state_dict(state['state_dict'])
            print(' No parallel === load from model === ')
    else:
        try:
            model.load_state_dict(state['state_dict'])
        except:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state['state_dict'].items():
                name = k[7:] # remove module.
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            print(' === load from parallel model === ')            
#    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
    name_='W_'+checkpoint_path
    torch.save(model,os.path.join('./checkpoint_d2s/best_501',name_))
    print('model saved to %s / %s' % ('./checkpoint_d2s/best_501',name_))
    return model
    
d2s_satge1_model = bert_unet_001.alphaBertForMaskedLM(config)   
try:      
    d2s_satge1_model = load_checkpoint('DS_pretrain0.pth',d2s_satge1_model,parallel=parallel)
except:
    print('*** No Pretrain_Model ***')
    pass

d2s_Model = d2s_satge1_model.bert
try:      
    d2s_Model = load_checkpoint('d2s0.pth',d2s_Model)
except:
    print('*** No Pretrain_Model ***')
    pass


def same_set(pre_symbol,symbol):
    numbers = torch.arange(tokenize_alphabets.alphabet2idx['0'],
                           tokenize_alphabets.alphabet2idx['9']+1)
    alphabets = torch.arange(tokenize_alphabets.alphabet2idx['A'],
                             tokenize_alphabets.alphabet2idx['z']+1)
    alphabets = alphabets[alphabets!=tokenize_alphabets.alphabet2idx['^']]
    
    num_abs = torch.cat([numbers,alphabets],dim=0)
    num_abs = num_abs.float()
    
    if pre_symbol is None:
        return False
    if pre_symbol.cpu().float() in num_abs and symbol.cpu().float() in num_abs:
        return True
    else:
        return False    


def split2words(src,rouge=False):
    numbers = torch.arange(tokenize_alphabets.alphabet2idx['0'],
                           tokenize_alphabets.alphabet2idx['9']+1)
    alphabets = torch.arange(tokenize_alphabets.alphabet2idx['A'],
                             tokenize_alphabets.alphabet2idx['z']+1)
    alphabets = alphabets[alphabets!=tokenize_alphabets.alphabet2idx['^']]
    
    num_abs = torch.cat([numbers,alphabets],dim=0)
    num_abs = num_abs.float()

    word_idx = {}
    isword = []
    pre_symbol = None
    word_num = -1
        
    for i, symbol in enumerate(src):
        if same_set(pre_symbol,symbol):
            word_idx[word_num].append(i)
        else:
            word_num += 1
            word_idx[word_num] = [i,]
            pre_symbol = symbol
            
            if symbol.cpu().float() in num_abs:
                isword.append(1)
            else:
                isword.append(0)
    if rouge:
        return word_idx,isword
    else:
        return word_idx

def clean_up(src,pred_prop, clean_type=0, mean_max='mean'):
    for i, src_ in enumerate(src):
        src_split = split2words(src_)
        if clean_type == 0:
            for wordpiece in src_split.values():
                if len(wordpiece) > 1:
                    if mean_max=='mean':
                        pred_prop[i][wordpiece] = torch.mean(pred_prop[i][wordpiece])
                    else:
                        pred_prop[i][wordpiece] = torch.max(pred_prop[i][wordpiece])
    return pred_prop
        
def IOU_ACC(pred_prop,trg,origin_len, threshold):
    pred_selected = pred_prop > threshold
    trg_selected = trg > threshold

    pred_pos = pred_selected.float()
    GT_pos = trg_selected.float()
    
    union = pred_pos+GT_pos
    IOU = []
    for j, u in enumerate(union):        
        I = u[:origin_len[j]] > 1
        U = u[:origin_len[j]] > 0
        I = sum(I.float())
        U = sum(U.float())
        iu = I/(U+1e-12)
        
        P = I/(sum(pred_pos[j][:origin_len[j]])+1e-12)
        R = I/(sum(GT_pos[j][:origin_len[j]])+1e-12)
        
        f1 = 2*P*R/(P+R+1e-12)
        
        FPR = (sum(pred_pos[j][:origin_len[j]])-I)/(origin_len[j]-sum(GT_pos[j][:origin_len[j]]))

#        IOU.append(iu)
        IOU.append(f1)
        
    return IOU
    
def ROC(pred_prop,trg,origin_len, ROC_threshold):
#    ROC_threshold = torch.linspace(0,1,100)
    ROC_IOU = []
    for t in ROC_threshold:
        IOU_set = IOU_ACC(pred_prop,trg,origin_len, t)
        ROC_IOU.append(IOU_set)
        
    return torch.tensor(ROC_IOU)

def f1_score(pred,trg,threshold):
    pred_selected = pred > threshold
    trg_selected = trg > threshold

    pred_pos = pred_selected.float()
    GT_pos = trg_selected.float()
    
    union = pred_pos+GT_pos
    
    I = union > 1
    U = union > 0
    I = sum(I.float())
    U = sum(U.float())
    iu = I/(U+1e-12)    
    
    P = I/(sum(pred_pos)+1e-12)
    R = I/(sum(GT_pos)+1e-12)
    
    f1 = 2*P*R/(P+R+1e-12)
    
    FPR = (sum(pred_pos)-I)/(len(trg)-sum(GT_pos)+1e-12)
    
    return P,R,FPR,f1,iu

def make_statistics(result,ep = 0):
    
    statistics = {'precision':[],
                  'recall_sensitivity':[],
                  'FPR':[],
                  'f1':[],
                  'IOU':[]}
    
    pred_ = result['pred']
    trg_ = result['trg']
    
    pred = torch.cat(pred_, dim=0)
    trg = torch.cat(trg_, dim=0)
    
    ROC_threshold = torch.linspace(0,1,100)
    
    for t in ROC_threshold:
        P,R,FPR,f1,iu = f1_score(pred,trg,t)
        statistics['precision'].append(P)
        statistics['recall_sensitivity'].append(R)
        statistics['FPR'].append(FPR)
        statistics['f1'].append(f1)
        statistics['IOU'].append(iu)
    
    plot_2(statistics, ep=ep)
    
#    return statistics

def plot_2(statistics, ep = 0):
    from sklearn.metrics import roc_curve, auc
    
    ROC_threshold = torch.linspace(0,1,100).numpy()
    P = torch.tensor(statistics['precision'],dtype=torch.float).numpy()
    R = torch.tensor(statistics['recall_sensitivity'],dtype=torch.float).numpy()
    FPR = torch.tensor(statistics['FPR'],dtype=torch.float).numpy()
    F1 = torch.tensor(statistics['f1'],dtype=torch.float).numpy()
    IOU = torch.tensor(statistics['IOU'],dtype=torch.float).numpy()
    
    fig = plt.figure(figsize=(18,8),dpi=100)
#    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(ROC_threshold,P,label='precision')
    ax.plot(ROC_threshold,R,label='recall') 
#    ax.plot(ROC_threshold,FPR) 
    ax.plot(ROC_threshold,F1,label='f1-measure') 
    ax.plot(ROC_threshold,IOU,label='IOU') 
    ax.set_xlabel('threshold')
    ax.set_title('Relation of threshold & f1-measure')
    plt.legend()
#    plt.ylabel = ''
    plt.xlim(0.,1.)
    plt.ylim(0.,1.)
    
    roc_auc = auc(FPR,R)
    label_auc = 'ROC, auc='+str(roc_auc)
    ax2 = fig.add_subplot(122)
    ax2.plot(FPR,R,label=label_auc)
    ax2.plot(ROC_threshold,ROC_threshold,'-.',label='random')
    
    ax2.set_xlabel('1-specificity')
    ax2.set_ylabel('sensitivity')
    ax2.set_title('ROC curve')
    plt.legend()
    plt.xlim(0.,1.)
    plt.ylim(0.,1.)
#    plt.title('mIOU')
    plt.legend()
    plt.savefig('./iou_pic/f1_'+str(ep)+'.jpg')
#    plt.show()
    plt.close()
    
    res_statistics = np.array([ROC_threshold,
                               P,
                               R,
                               FPR,
                               F1,
                               IOU])
    pd_res_statistics = pd.DataFrame(res_statistics)
    pd_res_statistics.to_csv('./iou_pic/statistics.csv')


#def rouge12l(rouge_set):
#    rouge_res = {'m':[],
#                 'n':[],
#                 'rouge-1p':[],
#                 'rouge-1r':[],
#                 'rouge-1f':[],
#                 'rouge-2p':[],
#                 'rouge-2r':[],
#                 'rouge-2f':[],
#                 'rouge-lcsp':[],
#                 'rouge-lcsr':[],
#                 'rouge-lcsf':[],
#                 }
#    for i, (hyp,ref) in enumerate(rouge_set):
#        m,n = len(hyp), len(ref)
#        
#        #rouge-1
#        rouge1_matrix = torch.zeros([m,n])        
#        for mi in range(m):
#            for ni in range(n):
#                if hyp[mi] == ref[ni]:
#                    rouge1_matrix[mi,ni] = 1
#        p1 = torch.sum(rouge1_matrix,dim = 0)
#        r1 = torch.sum(rouge1_matrix,dim = 1)
#        
#        P1 = sum((p1>0).float())/m if m > 0 else 0
#        R1 = sum((r1>0).float())/n if n > 0 else 0
#        F1 = 2*P1*R1/(P1+R1) if (P1+R1)>0 else 0
#
#        #rouge-2
#        if m>1 and n>1:
#            rouge2_matrix = torch.zeros([m,n])         
#            for mi in range(m-1):
#                for ni in range(n-1):
#                    if hyp[mi:mi+2] == ref[ni:ni+2]:
#                        rouge2_matrix[mi,ni] = 1
#            p2 = torch.sum(rouge2_matrix,dim = 0)
#            r2 = torch.sum(rouge2_matrix,dim = 1)
#            
#            P2 = sum((p2>0).float())/(m-1) if m-1 > 0 else 0
#            R2 = sum((r2>0).float())/(n-1) if n-1 > 0 else 0
#            F2 = 2*P2*R2/(P2+R2) if (P2+R2)>0 else 0
#        else:
#            P2,R2,F2 = P1,R1,F1
#            
#        #rouge-l
#        rougelcs_matrix = torch.zeros([m,n])
#        if m>1 and n>1:
#            for mi in range(m):
#                for ni in range(n):
#                    if hyp[mi] == ref[ni]:
#                        try:
#                            rougelcs_matrix[mi,ni] = rougelcs_matrix[mi-1,ni-1]+1 
#                        except:
#                            rougelcs_matrix[mi,ni] = 1
#                    else:
#                        if mi>1:
#                            if ni>1:
#                                rougelcs_matrix[mi,ni] = max(rougelcs_matrix[mi-1,ni],rougelcs_matrix[mi,ni-1])
#                            else:
#                                rougelcs_matrix[mi,ni] = rougelcs_matrix[mi-1,ni]
#            Plcs = rougelcs_matrix[-1,-1].item()/m if m > 0 else 0
#            Rlcs = rougelcs_matrix[-1,-1].item()/n if n > 0 else 0
#            Flcs = 2*Plcs*Rlcs/(Plcs+Rlcs) if (Plcs+Rlcs)>0 else 0
#        else:
#            Plcs,Rlcs,Flcs= P1,R1,F1
#        
#        rouge_res['m'].append(m)
#        rouge_res['n'].append(n)
#        rouge_res['rouge-1p'].append(float(P1))
#        rouge_res['rouge-1r'].append(float(R1))
#        rouge_res['rouge-1f'].append(float(F1))
#        rouge_res['rouge-2p'].append(float(P2))
#        rouge_res['rouge-2r'].append(float(R2))
#        rouge_res['rouge-2f'].append(float(F2))
#        rouge_res['rouge-lcsp'].append(float(Plcs))
#        rouge_res['rouge-lcsr'].append(float(Rlcs))
#        rouge_res['rouge-lcsf'].append(float(Flcs))
#    return rouge_res

        
def test_alphaBert(DS_model,dloader,threshold=0.5, is_clean_up=True, ep=0, train=False,mean_max='mean',rouge=False):
    if not train:
        DS_model.to(device)
        DS_model = torch.nn.DataParallel(DS_model)
    DS_model.eval()
    
    out_pred_res = []
    mIOU = []
    ROC_IOU_ = []
    
    ROC_threshold = torch.linspace(0,1,100).to(device)
    
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
            
            pred_prop, sequence_output = DS_model(input_ids=src,
                                 attention_mask=att_mask)
            if is_clean_up:
                pred_prop = clean_up(src,pred_prop,mean_max=mean_max)
            
            for i, src_ in enumerate(src):
                all_pred_trg['pred'].append(pred_prop[i][:origin_len[i]].cpu())
                all_pred_trg['trg'].append(trg[i][:origin_len[i]].cpu())
                                
                if rouge:
                    src_split, src_isword = split2words(src_,rouge=rouge)
                    referecne = []
                    hypothesis = []
                    for j in range(len(src_split)):
                        if src_isword[j]>0:
                            if trg[i][src_split[j]][0].cpu()>threshold:
                                referecne.append(tokenize_alphabets.convert_idx2str(src_[src_split[j]]))
                            if pred_prop[i][src_split[j]][0].cpu()>threshold:
                                hypothesis.append(tokenize_alphabets.convert_idx2str(src_[src_split[j]]))
                            
                    rouge_set.append((hypothesis,referecne))
            
#            mIOU += IOU_ACC(pred_prop,trg,origin_len, threshold)           
#            ROC_IOU_.append(ROC(pred_prop,trg,origin_len, ROC_threshold))
            
            pred_selected = pred_prop > threshold
            trg_selected = trg > threshold
            
            for i, src_ in enumerate(src):
                a_ = tokenize_alphabets.convert_idx2str(src_[:origin_len[i]])
                s_ = tokenize_alphabets.convert_idx2str(src_[pred_selected[i]])
                t_ = tokenize_alphabets.convert_idx2str(src_[trg_selected[i]])
#                print(a_,pred_prop[0],s_,t_)
                
                out_pred_res.append((a_,s_,t_,pred_prop[0]))
            print(batch_idx, len(dloader))
    out_pd_res = pd.DataFrame(out_pred_res)
    out_pd_res.to_csv('test_pred.csv', sep=',')
    
    make_statistics(all_pred_trg, ep=ep)
#    mIOU = torch.tensor(mIOU)
#    out_pd_mIOU = pd.DataFrame(mIOU)
#    out_pd_mIOU.to_csv('mIOU.csv', sep=',')
    
#    ROC_IOU = torch.cat(ROC_IOU_,dim=1)
#    out_pd_ROC_IOU = pd.DataFrame(ROC_IOU.cpu().numpy())
#    out_pd_ROC_IOU.to_csv('ROC_IOU.csv', sep=',')
#    
#    ROC_IOU_mean = torch.mean(ROC_IOU,dim =1)
#    ROC_IOU_mean = torch.stack([ROC_threshold.cpu(),ROC_IOU_mean])
#    out_pd_ROC_IOU_mean = pd.DataFrame(ROC_IOU_mean.cpu().numpy())
#    out_pd_ROC_IOU_mean.to_csv('ROC_IOU_mean.csv', sep=',')
#    
#    plot_1(ROC_IOU_mean,ep)
    DS_model.train()
    
    if rouge:
        rouge_res = rouge12l(rouge_set)
        rouge_res_pd = pd.DataFrame(rouge_res)
        rouge_res_pd.to_csv('./iou_pic/rouge_res.csv',index=False)
        rouge_res_np = np.array(rouge_res_pd)
        
        pd.DataFrame(rouge_res_np.mean(axis =0)).to_csv('./iou_pic/rouge_res_mean.csv',index=False)
    
def plot_1(ROC_IOU_mean, ep = 0):  
    fig = plt.figure(figsize=(8,6),dpi=100)
#    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ROC_IOU_mean[0].numpy(),ROC_IOU_mean[1].numpy())  

    plt.title('mIOU')
#    plt.legend()
    plt.savefig('./iou_pic/mIOU_'+str(ep)+'.jpg')
#    plt.show()
    plt.close()

def train_alphaBert(DS_model,dloader,lr=1e-4,epoch=10,log_interval=20):
    DS_model.to(device)
    model_optimizer = optim.Adam(DS_model.parameters(), lr=lr)
    DS_model = torch.nn.DataParallel(DS_model)
    DS_model.train()
    
#    criterion = nn.MSELoss().to(device)
    criterion = alphabert_loss_v02.Alphabert_loss(device=device)
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
            trg = trg.float().to(device)
            att_mask = att_mask.float().to(device)
            origin_len = origin_len.to(device)
        
            pred_prop = DS_model(input_ids=src,
                                 attention_mask=att_mask)
            loss = criterion(pred_prop[0],trg,origin_len)
            
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
                save_checkpoint('d2s.pth',DS_model,model_optimizer)
                    
            iteration +=1
        if ep % 1 ==0:
            save_checkpoint('d2s.pth',DS_model,model_optimizer)
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

def make_cloze(src,max_len,device,percent=0.15, fix=True):
    err_sequence = torch.randperm(max_len)
    err_replace = int(percent*max_len)
    err_sequence = err_sequence[:err_replace]

    numbers = torch.arange(tokenize_alphabets.alphabet2idx['0'],
                           tokenize_alphabets.alphabet2idx['9']+1)
    alphabets = torch.arange(tokenize_alphabets.alphabet2idx['A'],
                             tokenize_alphabets.alphabet2idx['z']+1)
    alphabets = alphabets[alphabets!=tokenize_alphabets.alphabet2idx['^']]
    
    numbers_li = list(numbers.numpy())
    alphabets_li = list(alphabets.numpy())
    others_li = [o for o in range(tokenize_alphabets.n_alphabets) if o not in numbers_li+alphabets_li]
    others = torch.tensor(others_li)
    
    for i,s in enumerate(src):
        for e in err_sequence:
            if s[e] in alphabets:
                if fix:
                    s[e] = 56
                else:
                    r = random.random()
                    if r>0.2:
                        s[e] = 56
                    elif r <= 0.2 and r > 0.1:
                        random_idx = random.randint(0, len(alphabets)-1)
                        s[e] = alphabets[random_idx]
            elif s[e] in numbers:
                if fix:
                    s[e] = 56
                else:
                    r = random.random()
                    if r>0.8:
                        s[e] = 56
                    elif r <= 0.8 and r > 0.4:
                        random_idx = random.randint(0, len(numbers)-1)
                        s[e] = numbers[random_idx]               
            elif s[e] ==0:
                if fix:
                    s[e] = 56
                else:
                    r = random.random()
                    if r>0.9:
                        s[e] = 56
                    elif r <= 0.9 and r > 0.8:
                        random_idx = random.randint(0, len(others)-1)
                        s[e] = others[random_idx]
            else:
                if fix:
                    s[e] = 56
                else:
                    r = random.random()
                    if r>0.8:
                        s[e] = 56
                    elif r <= 0.8 and r > 0.4:
                        random_idx = random.randint(0, len(others)-1)
                        s[e] = others[random_idx]                
    
    return src.to(device), err_sequence.to(device)

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
        
            prediction_scores, pred_prop = TS_model(input_ids=src,
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
    
def train_alphaBert_stage1(TS_model,dloader,lr=1e-4,epoch=10,log_interval=20,
                           cloze_fix=True, use_amp=False, parallel=True):
    TS_model.to(device)
    model_optimizer = optim.Adam(TS_model.parameters(), lr=lr)

    if parallel:
        TS_model = torch.nn.DataParallel(TS_model)

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
        
            prediction_scores, pred_prop = TS_model(input_ids=src,
                                                    attention_mask=att_mask)
            
#            print(1111,prediction_scores.view(-1,84).shape)
#            print(1111,trg.view(-1).shape)
            
            loss = criterion(prediction_scores.view(-1,84).contiguous(),trg.view(-1).contiguous())
                     
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
                    save_checkpoint('DS_pretrain.pth',TS_model,model_optimizer)
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
                    test_res = test_alphaBert_stage1(TS_model,stage1_dataloader_test)
                    print(' ===== Show the Test of Pretrain ===== ')
                    
                    out_pred_test.append((ep,*test_res))                    
                    out_pd_test = pd.DataFrame(out_pred_test)
                    out_pd_test.to_csv('./result/out_pred_test.csv', sep=',')
                                
            iteration +=1
        if ep % 1 ==0:
            save_checkpoint('DS_pretrain.pth',TS_model,model_optimizer)

            print('======= epoch:%i ========'%ep)
            
        print('++ Ep Time: {:.1f} Secs ++'.format(time.time()-t0)) 
        total_loss.append(float(epoch_loss/epoch_cases))
        pd_total_loss = pd.DataFrame(total_loss)
        pd_total_loss.to_csv('./result/total_loss_pretrain.csv', sep = ',')
    print(total_loss)





#train_alphaBert(d2s_Model,D2S_trainloader,lr=1e-5,epoch=200,log_interval=3)
#train_alphaBert_stage1(d2s_satge1_model,stage1_dataloader,
#                       lr=1e-5,
#                       epoch=10,
#                       log_interval=5,
#                       cloze_fix=False, 
#                       use_amp=False,
#                       parallel=parallel)

#test_alphaBert(d2s_Model,D2S_valloader,threshold=0.54, is_clean_up=True, ep='f1_Max',mean_max='max',rouge=True)
#test_alphaBert(d2s_Model,D2S_testloader,threshold=0.54, is_clean_up=True, ep='f1',mean_max='max',rouge=True)
#test_alphaBert(d2s_Model,D2S_cyy_testloader,threshold=0.61, is_clean_up=True, ep='f1_Max_cyy',mean_max='max',rouge=True)

try:
    task = sys.argv[1]
    train_val_test = sys.argv[2] 
    print('***** task = ',task)
    print('***** mode = ',train_val_test)
except:
    task = 'pretrain'
    train_val_test = 'val'

alphaloader = alphabet_loaders(datapath=filepath, 
                          config=config, 
                          tokenize_alphabets=tokenize_alphabets,
                          num_workers=4,
                          batch_size=batch_size,
                          bioBERTcorpus=0)

if task == 'pretrain':
    d2s_satge1_model = alphaBERT_model.alphaBertForMaskedLM(config)   
    try:      
        d2s_satge1_model = load_checkpoint(checkpoint_file,'DS_pretrain.pth',d2s_satge1_model)
    except:
        print('*** No Pretrain_Model ***')
        pass
    loader = alphaloader.make_loaders(finetune=False,head=False,ahead=False,pretrain=True,trvte='test')
    stage1_dataloader = loader['stage1_dataloader']
    stage1_test_dataloader = loader['stage1_dataloader_test']
#    train_alphaBert_stage1(d2s_satge1_model,
#                           stage1_dataloader,
#                           stage1_test_dataloader,
#                           lr=1e-5,
#                           epoch=50,
#                           log_interval=5,
#                           cloze_fix=False, 
#                           use_amp=False,
#                           parallel=parallel)

    train_alphaBert_stage1(d2s_satge1_model,stage1_dataloader,
                           lr=1e-5,
                           epoch=10,
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

    loader = alphaloader.make_loaders(finetune=True,head=True,ensemble=True,ahead=False,pretrain=False,trvte=train_val_test)
    if train_val_test == 'train':
        D2S_trainloader = loader['D2S_ensemble_trainloader']
        train_alphaBert(d2s_Model,D2S_trainloader,lr=1e-5,epoch=200,log_interval=3)
        
    elif train_val_test == 'val':
        D2S_valloader = loader['D2S_ensemble_valloader']
        test_alphaBert(d2s_Model,D2S_valloader,threshold=0.5,
                       is_clean_up=True,
                       ep='f1_max3',
                       mean_max='max',
                       rouge=True)
        
    elif train_val_test == 'test':
        D2S_testloader = loader['D2S_ensemble_testloader']
        test_alphaBert(d2s_Model,
                       D2S_testloader,
                       threshold=0.54, 
                       is_clean_up=True, 
                       ep='f1_test',
                       mean_max='mean',
                       rouge=True)
        

