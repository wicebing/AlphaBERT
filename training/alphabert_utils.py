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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  

def same_set(pre_symbol,symbol,tokenize_alphabets):
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

def split2words(src,tokenize_alphabets,rouge=False,iswordfilt=False):
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
    
    if iswordfilt:
        for i, symbol in enumerate(src):
            if symbol.cpu().float() in num_abs:
                if same_set(pre_symbol,symbol,tokenize_alphabets=tokenize_alphabets):
                    word_idx[word_num].append(i)
                else:
                    word_num += 1
                    word_idx[word_num] = [i,]   
            pre_symbol = symbol
        return word_idx
    
    else:
        for i, symbol in enumerate(src):
            if same_set(pre_symbol,symbol,tokenize_alphabets=tokenize_alphabets):
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

def clean_up(src,pred_prop,tokenize_alphabets, clean_type=0, mean_max='mean'):
    for i, src_ in enumerate(src):
        src_split = split2words(src_,tokenize_alphabets=tokenize_alphabets)
        if clean_type == 0:
            for wordpiece in src_split.values():
                if len(wordpiece) > 1:
                    if mean_max=='mean':
                        pred_prop[i][wordpiece] = torch.mean(pred_prop[i][wordpiece])
                    else:
                        pred_prop[i][wordpiece] = torch.max(pred_prop[i][wordpiece])
    return pred_prop

def clean_up_v204_sf(src,pred_prop_bin,tokenize_alphabets, clean_type=0, mean_max='mean'):
    pred_prop_value, pred_prop = pred_prop_bin.max(dim=2)
    pred_prop = pred_prop.float()
    for i, src_ in enumerate(src):
        src_split = split2words(src_,tokenize_alphabets=tokenize_alphabets)
        if clean_type == 0:
            for wordpiece in src_split.values():
                if len(wordpiece) > 1:
                    if mean_max=='mean':
                        pred_prop[i][wordpiece] = torch.mean(pred_prop[i][wordpiece])
                    else:
                        pred_prop[i][wordpiece] = torch.max(pred_prop[i][wordpiece])
    return pred_prop

def clean_up_v204_ft(src,pred_prop_bin,tokenize_alphabets, clean_type=0, mean_max='mean'):
#    pred_prop_bin_sigmoid = nn.Sigmoid()(pred_prop_bin)    
#    pred_prop = pred_prop_bin_sigmoid[:,:,1]/(pred_prop_bin_sigmoid[:,:,1]+pred_prop_bin_sigmoid[:,:,0])
    
    pred_prop_bin_softmax = nn.Softmax(dim=-1)(pred_prop_bin)
    pred_prop = pred_prop_bin_softmax[:,:,1]
    
    for i, src_ in enumerate(src):
        src_split = split2words(src_,tokenize_alphabets=tokenize_alphabets)
        if clean_type == 0:
            for wordpiece in src_split.values():
                if len(wordpiece) > 1:
                    if mean_max=='mean':
                        pred_prop[i][wordpiece] = torch.mean(pred_prop[i][wordpiece])
                    else:
                        pred_prop[i][wordpiece] = torch.max(pred_prop[i][wordpiece])
    return pred_prop

def clean_up_ensemble(src,pred_prop_bin,tokenize_alphabets, clean_type=0, mean_max='mean'):
#    pred_prop_bin_sigmoid = nn.Sigmoid()(pred_prop_bin)    
#    pred_prop = pred_prop_bin_sigmoid[:,:,1]/(pred_prop_bin_sigmoid[:,:,1]+pred_prop_bin_sigmoid[:,:,0])
    
    pred_prop_bin_softmax = nn.Softmax(dim=-1)(pred_prop_bin)
    pred_prop = pred_prop_bin_softmax[:,:,1]
    
    for i, src_ in enumerate(src):
        src_split = split2words(src_,tokenize_alphabets=tokenize_alphabets)
        if clean_type == 0:
            for wordpiece in src_split.values():
                wp = [wordpiece[0]-1,]
                wp2 = [wordpiece[0]-1,wordpiece[0]]
                if len(wordpiece) > 1:
                    if mean_max=='mean':
                        pred_prop[i][wordpiece] = torch.mean(pred_prop[i][wordpiece])
                        pred_prop[i][wp] = torch.mean(pred_prop[i][wp2])
                    else:
                        pred_prop[i][wordpiece] = torch.max(pred_prop[i][wordpiece])
                        pred_prop[i][wp] = torch.mean(pred_prop[i][wp2])
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
#    ROC_threshold, _ = pred.sort()
    
    for t in ROC_threshold:
        P,R,FPR,f1,iu = f1_score(pred,trg,t)
        statistics['precision'].append(P)
        statistics['recall_sensitivity'].append(R)
        statistics['FPR'].append(FPR)
        statistics['f1'].append(f1)
        statistics['IOU'].append(iu)
    
    plot_2(statistics,pred,trg, ep=ep)
    
#    return statistics

def plot_2(statistics,pred,trg, ep = 0):
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
    
#    roc_auc = auc(FPR,R)
#    label_auc = 'ROC, auc='+str(roc_auc)
    ax2 = fig.add_subplot(122)
#    ax2.plot(FPR,R,label=label_auc)

    fpr, tpr, _ = roc_curve(trg.cpu().numpy(), pred.cpu().numpy())
    roc_auc2 = auc(fpr,tpr)
    label_auc2 = 'ROC, auc='+str(roc_auc2)
    ax2.plot(fpr,tpr,label=label_auc2)
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


def rouge12l(rouge_set):
    rouge_res = {'m':[],
                 'n':[],
                 'rouge-1p':[],
                 'rouge-1r':[],
                 'rouge-1f':[],
                 'rouge-2p':[],
                 'rouge-2r':[],
                 'rouge-2f':[],
                 'rouge-lcsp':[],
                 'rouge-lcsr':[],
                 'rouge-lcsf':[],
                 }
    for i, (hyp,ref) in enumerate(rouge_set):
        m,n = len(hyp), len(ref)
        
        #rouge-1
        rouge1_matrix = torch.zeros([m,n])        
        for mi in range(m):
            for ni in range(n):
                if hyp[mi] == ref[ni]:
                    rouge1_matrix[mi,ni] = 1
        n1 = torch.sum(rouge1_matrix,dim = 0)
        m1 = torch.sum(rouge1_matrix,dim = 1)
        
        R1 = sum((n1>0).float())/n if n > 0 else 0
        P1 = sum((m1>0).float())/m if m > 0 else 0
        F1 = 2*P1*R1/(P1+R1) if (P1+R1)>0 else 0
        
        #rouge-2
        if m>1 and n>1:
            rouge2_matrix = torch.zeros([m,n])         
            for mi in range(m-1):
                for ni in range(n-1):
                    if hyp[mi:mi+2] == ref[ni:ni+2]:
                        rouge2_matrix[mi,ni] = 1
            n2 = torch.sum(rouge2_matrix,dim = 0)
            m2 = torch.sum(rouge2_matrix,dim = 1)
            
            R2 = sum((n2>0).float())/(n-1) if n-1 > 0 else 0
            P2 = sum((m2>0).float())/(m-1) if m-1 > 0 else 0
            F2 = 2*P2*R2/(P2+R2) if (P2+R2)>0 else 0
        else:
            P2,R2,F2 = P1,R1,F1
            
        #rouge-l
        rougelcs_matrix = torch.zeros([m+1,n+1])
        if m>1 and n>1:
            for mi in range(m):
                for ni in range(n):
                    if hyp[mi] == ref[ni]:
                        rougelcs_matrix[mi+1,ni+1] = rougelcs_matrix[mi,ni]+1
                    else:
                        rougelcs_matrix[mi+1,ni+1] = max(rougelcs_matrix[mi,ni+1],rougelcs_matrix[mi+1,ni])                                

            Plcs = rougelcs_matrix[-1,-1].item()/m if m > 0 else 0
            Rlcs = rougelcs_matrix[-1,-1].item()/n if n > 0 else 0
            Flcs = 2*Plcs*Rlcs/(Plcs+Rlcs) if (Plcs+Rlcs)>0 else 0
        else:
            Plcs,Rlcs,Flcs= P1,R1,F1
        
        rouge_res['m'].append(m)
        rouge_res['n'].append(n)
        rouge_res['rouge-1p'].append(float(P1))
        rouge_res['rouge-1r'].append(float(R1))
        rouge_res['rouge-1f'].append(float(F1))
        rouge_res['rouge-2p'].append(float(P2))
        rouge_res['rouge-2r'].append(float(R2))
        rouge_res['rouge-2f'].append(float(F2))
        rouge_res['rouge-lcsp'].append(float(Plcs))
        rouge_res['rouge-lcsr'].append(float(Rlcs))
        rouge_res['rouge-lcsf'].append(float(Flcs))
    return rouge_res
    
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

def make_cloze(src,max_len,device,tokenize_alphabets,percent=0.15, fix=True):
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
    
    return src.to(src.device), err_sequence.to(src.device)

def save_checkpoint(checkpoint_file,checkpoint_path, model, optimizer, parallel):
    if parallel:
        from collections import OrderedDict
        state_dict = OrderedDict()
        for k, v in model.state_dict().items():
            name = k[7:] # remove module.
            state_dict[name] = v
    else:
        state_dict = model.state_dict()

    state = {'state_dict': state_dict,
             'optimizer' : optimizer.state_dict()}
    torch.save(state, os.path.join(checkpoint_file,checkpoint_path))

    print('model saved to %s / %s' % (checkpoint_file,checkpoint_path))
    
def load_checkpoint(checkpoint_file,checkpoint_path, model):
    state = torch.load(os.path.join(checkpoint_file,checkpoint_path),
#                       map_location={'cuda:0':'cuda:1'}
                       )
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s / %s' % (checkpoint_file,checkpoint_path))
    name_='W_'+checkpoint_path
    torch.save(model,os.path.join(checkpoint_file,name_))
    print('model saved to %s / %s' % (checkpoint_file,name_))
    return model

def rouge12l_recheck(rouge_set):
    rouge_res = {'m':[],
                 'n':[],
                 'rouge-1p':[],
                 'rouge-1r':[],
                 'rouge-1f':[],
                 'rouge-2p':[],
                 'rouge-2r':[],
                 'rouge-2f':[],
                 'rouge-lcsp':[],
                 'rouge-lcsr':[],
                 'rouge-lcsf':[],
                 }
    for i, (ref,hyp) in enumerate(rouge_set):
        try:
            hyp = hyp.split()
        except:
            hyp = []
        try:
            ref = ref.split()         
        except:
            ref = []
            
        m,n = len(hyp), len(ref)
        #rouge-1
        rouge1_matrix = torch.zeros([m,n])        
        for mi in range(m):
            for ni in range(n):
                if hyp[mi] == ref[ni]:
                    rouge1_matrix[mi,ni] = 1
        n1 = torch.sum(rouge1_matrix,dim = 0)
        m1 = torch.sum(rouge1_matrix,dim = 1)
        
        R1 = sum((n1>0).float())/n if n > 0 else 0
        P1 = sum((m1>0).float())/m if m > 0 else 0
        F1 = 2*P1*R1/(P1+R1) if (P1+R1)>0 else 0
        
        #rouge-2
        if m>1 and n>1:
            rouge2_matrix = torch.zeros([m,n])         
            for mi in range(m-1):
                for ni in range(n-1):
                    if hyp[mi:mi+2] == ref[ni:ni+2]:
                        rouge2_matrix[mi,ni] = 1
            n2 = torch.sum(rouge2_matrix,dim = 0)
            m2 = torch.sum(rouge2_matrix,dim = 1)
            
            R2 = sum((n2>0).float())/(n-1) if n-1 > 0 else 0
            P2 = sum((m2>0).float())/(m-1) if m-1 > 0 else 0
            F2 = 2*P2*R2/(P2+R2) if (P2+R2)>0 else 0
        else:
            P2,R2,F2 = P1,R1,F1
            
        #rouge-l
        rougelcs_matrix = torch.zeros([m+1,n+1])
        if m>1 and n>1:
            for mi in range(m):
                for ni in range(n):
                    if hyp[mi] == ref[ni]:
                        rougelcs_matrix[mi+1,ni+1] = rougelcs_matrix[mi,ni]+1
                    else:
                        rougelcs_matrix[mi+1,ni+1] = max(rougelcs_matrix[mi,ni+1],rougelcs_matrix[mi+1,ni])                                

            Plcs = rougelcs_matrix[-1,-1].item()/m if m > 0 else 0
            Rlcs = rougelcs_matrix[-1,-1].item()/n if n > 0 else 0
            Flcs = 2*Plcs*Rlcs/(Plcs+Rlcs) if (Plcs+Rlcs)>0 else 0
        else:
            Plcs,Rlcs,Flcs= P1,R1,F1
        
        rouge_res['m'].append(m)
        rouge_res['n'].append(n)
        rouge_res['rouge-1p'].append(float(P1))
        rouge_res['rouge-1r'].append(float(R1))
        rouge_res['rouge-1f'].append(float(F1))
        rouge_res['rouge-2p'].append(float(P2))
        rouge_res['rouge-2r'].append(float(R2))
        rouge_res['rouge-2f'].append(float(F2))
        rouge_res['rouge-lcsp'].append(float(Plcs))
        rouge_res['rouge-lcsr'].append(float(Rlcs))
        rouge_res['rouge-lcsf'].append(float(Flcs))
    return rouge_res