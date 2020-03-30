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

import gruModel
import alphabert_dataset_v02
import dg_alphabets
#import bert_unet_001
import alphabert_loss_v02
from alphabert_utils import clean_up, split2words, rouge12l, make_statistics


try:
    task = sys.argv[1]
    print('*****task= ',task)
except:
    task = 'test'

batch_size = 8
device = 'cuda'
parallel = False

#device_ids = list(range(rank * n, (rank + 1) * n))

filepath = './data'
filename1a = os.path.join(filepath,'finetune_train.csv')
data_a = pd.read_csv(filename1a, header=None, encoding='utf-8')
data_train = data_a

filename1b = os.path.join(filepath,'finetune_valtest.csv')
data_val = pd.read_csv(filename1b, header=None, encoding='utf-8')

data_test = data_val.sample(frac=0.5,random_state=1)
data_val = data_val.drop(data_test.index)

D_L_set = np.array(data_a)[:,[1,3]]

D_L_train = np.array(data_train)[:,[1,3]]
D_L_val = np.array(data_val)[:,[1,3]]
D_L_test = np.array(data_test)[:,[1,3]]


filename2 = os.path.join(filepath,'codes.csv')
data2 = pd.read_csv(filename2, header=None)
icd_wd = np.array(data2[4])

filename3 = os.path.join(filepath,'C2018_OUTPUT_CD.txt')
data3 = pd.read_csv(filename3,encoding = 'utf8',sep='\t')
ntuh_wd = np.array(data3['TYPECONTENT'])

stage1_wd = np.concatenate([icd_wd,ntuh_wd],axis=0)

filename4 = os.path.join(filepath,'valtest_18488.csv')
data4= pd.read_csv(filename4,encoding = 'utf8')
stage1_test = np.array(data4['Diagnosis'])

def normalizeString(s):
#    s = unicodeToAscii(s.lower().strip())
#    s = re.sub(r"([.!?])", r" ", s)
    s = re.sub(r"[^a-zA-Z,.0-9/ ^`'<>:;+=~`@#$%&*()?!-]", r" ", s)
    return s

all_expect_alphabets = []
unsame = 0 
same = 0
max_txt = 0

for i in D_L_set:
#    print(len(i[0]),len(i[1]))
    max_txt = max(len(i[0]),max_txt)
    if len(i[0]) != len(i[1]):
        unsame+=1
    else:
        same +=1
        
    ss = normalizeString(i[0])
    sl = list(ss)
   
    for sli in sl:
        if sli not in all_expect_alphabets:
            all_expect_alphabets.append(sli)

all_expect_alphabets.sort()

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

D2S_datatrain = alphabert_dataset_v02.D2Lntuh(D_L_train,
                                        tokenize_alphabets,
                                        clamp_size=config['max_position_embeddings'],
                                        train=True)

D2S_trainloader = DataLoader(D2S_datatrain,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4,
                            collate_fn=alphabert_dataset_v02.collate_fn_lstm)

D2S_val = alphabert_dataset_v02.D2Lntuh(D_L_val,
                                    tokenize_alphabets,
                                    clamp_size=config['max_position_embeddings'],
                                    train=False)

D2S_valloader = DataLoader(D2S_val,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=4,
                           collate_fn=alphabert_dataset_v02.collate_fn_lstm)

D2S_test = alphabert_dataset_v02.D2Lntuh(D_L_test,
                                     tokenize_alphabets,
                                     clamp_size=config['max_position_embeddings'],
                                     train=False)

D2S_testloader = DataLoader(D2S_test,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=4,
                           collate_fn=alphabert_dataset_v02.collate_fn_lstm)

filename1cyy = os.path.join(filepath,'finetune_test_cyy.csv')
data_test_cyy = pd.read_csv(filename1cyy, header=None, encoding='utf-8')
D_L_test_cyy = np.array(data_test_cyy)[:,[1,3]]

D2S_test_cyy = alphabert_dataset_v02.D2Lntuh(D_L_test_cyy,
                                     tokenize_alphabets,
                                     clamp_size=config['max_position_embeddings'],
                                     train=False)

D2S_cyy_testloader = DataLoader(D2S_test_cyy,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=4,
                           collate_fn=alphabert_dataset_v02.collate_fn_lstm)

filename1lin= os.path.join(filepath,'finetune_test_lin.csv')
data_test_lin = pd.read_csv(filename1lin, header=None, encoding='utf-8')
D_L_test_lin = np.array(data_test_lin)[:,[1,3]]

D2S_test_lin = alphabert_dataset_v02.D2Lntuh(D_L_test_lin,
                                     tokenize_alphabets,
                                     clamp_size=config['max_position_embeddings'],
                                     train=False)

D2S_lin_testloader = DataLoader(D2S_test_lin,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=4,
                           collate_fn=alphabert_dataset_v02.collate_fn_lstm)

D_L_test_all = np.concatenate([D_L_test,D_L_test_cyy,D_L_test_lin],axis=0)


D2S_test_all = alphabert_dataset_v02.D2Lntuh(D_L_test_all,
                                     tokenize_alphabets,
                                     clamp_size=config['max_position_embeddings'],
                                     train=False)

D2S_all_testloader = DataLoader(D2S_test_all,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=4,
                           collate_fn=alphabert_dataset_v02.collate_fn_lstm)

stage1_dataset = alphabert_dataset_v02.D_stage1(stage1_wd,
                                            tokenize_alphabets,
                                            clamp_size=config['max_position_embeddings'],
                                            icd10=71704,
                                            train=True)

stage1_dataloader = DataLoader(stage1_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=4,
                               collate_fn=alphabert_dataset_v02.collate_fn)

stage1_dataset_test = alphabert_dataset_v02.D_stage1(stage1_test,
                                            tokenize_alphabets,
                                            clamp_size=config['max_position_embeddings'],
                                            train=False)

stage1_dataloader_test = DataLoader(stage1_dataset_test,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=4,
                               collate_fn=alphabert_dataset_v02.collate_fn)

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, os.path.join('./checkpoint_lstm',checkpoint_path))
    name_='W_'+checkpoint_path
    torch.save(model,os.path.join('./checkpoint_lstm',name_))
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, parallel=parallel):
    state = torch.load(os.path.join('./checkpoint_lstm',checkpoint_path),
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
    name_='WW_'+checkpoint_path
    torch.save(model,os.path.join('./checkpoint_lstm',name_))
    print('model saved to %s / %s' % ('./checkpoint_lstm',name_))
    return model

'''
pretrained embedding by W2V

from gensim.models import Word2Vec
embedding_dim = 64
window_ = 10
min_count_ = 0
sample_ = 1e-3
iter_ = 20

pretrained_emb_tokens = all_expect_alphabets + list(stage1_wd)

w2v_model = Word2Vec(pretrained_emb_tokens, size=embedding_dim, window=window_, min_count=min_count_,sample=sample_,iter=iter_,  workers=32)

w2v_weights = torch.FloatTensor(w2v_model.wv.vectors)
vocab = dict([(k, v.index) for k,v in w2v_model.wv.vocab.items()])
vocab_list = [k for k,v in w2v_model.wv.vocab.items()]

init_weight = w2v_weights[:84]

print(w2v_model.wv.most_similar('e', topn=10))
lstm_model = gruModel.LSTM_baseModel(config,init_weight)  
'''

#alphabert 904405 parameters 
#lstm      899841 parameters    
lstm_model = gruModel.LSTM_baseModel(config)
try:      
    lstm_model = load_checkpoint('lstm_pretrain.pth',lstm_model,parallel=parallel)
except:
    print('*** No Pretrain_Model ***')
    pass

def test_alphaBert(DS_model,dloader,threshold=0.5, is_clean_up=True, ep=0, train=False,mean_max='mean',rouge=False, parallel=parallel):
    if not train:
        DS_model.to(device)
        if parallel:
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
            
            pred_prop = DS_model(x=src,
                                 x_lengths=origin_len)
            if is_clean_up:
                pred_prop = clean_up(src,pred_prop,mean_max=mean_max,tokenize_alphabets=tokenize_alphabets)
            
            for i, src_ in enumerate(src):
                all_pred_trg['pred'].append(pred_prop[i][:origin_len[i]].cpu())
                all_pred_trg['trg'].append(trg[i][:origin_len[i]].cpu())
                                
                if rouge:
                    src_split, src_isword = split2words(src_,rouge=rouge,tokenize_alphabets=tokenize_alphabets)
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

    DS_model.train()
    
    if rouge:
        rouge_res = rouge12l(rouge_set)
        rouge_res_pd = pd.DataFrame(rouge_res)
        rouge_res_pd.to_csv('./iou_pic/lstm/rouge_res.csv',index=False)
        rouge_res_np = np.array(rouge_res_pd)
        
        pd.DataFrame(rouge_res_np.mean(axis =0)).to_csv('./iou_pic/lstm/rouge_res_mean.csv',index=False)

def train_lstm(DS_model,dloader,lr=1e-4,epoch=10,log_interval=20,parallel=parallel):
    DS_model.to(device)
    model_optimizer = optim.Adam(DS_model.parameters(), lr=lr)
    if parallel:
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
        
            pred_prop = DS_model(x=src,
                                 x_lengths=origin_len)

            loss = criterion(pred_prop,trg,origin_len)
            
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
                save_checkpoint('lstm_pretrain.pth',DS_model,model_optimizer)
                    
            iteration +=1
        if ep % 1 ==0:
            save_checkpoint('lstm_pretrain.pth',DS_model,model_optimizer)
#            test_alphaBert(DS_model,D2S_valloader, 
#                           is_clean_up=True, ep=ep,train=True)

            print('======= epoch:%i ========'%ep)
            
#        print('total loss: {:.4f}'.format(total_loss/len(dloader)))         
        print('++ Ep Time: {:.1f} Secs ++'.format(time.time()-t0)) 
#        total_loss.append(epoch_loss)
        total_loss.append(float(epoch_loss/epoch_cases))
        pd_total_loss = pd.DataFrame(total_loss)
        pd_total_loss.to_csv('./iou_pic/lstm/total_loss_finetune.csv', sep = ',')
    print(total_loss)





#train_lstm(lstm_model,D2S_trainloader,lr=1e-5,epoch=200,log_interval=3,parallel=parallel)


# test_alphaBert(lstm_model,D2S_valloader,threshold=0.5, is_clean_up=True, ep='f1_Max_lstm_val',mean_max='max',rouge=True)
# test_alphaBert(lstm_model,D2S_testloader,threshold=0.38, is_clean_up=True, ep='f1_Max_lstm_test',mean_max='max',rouge=True)
# test_alphaBert(lstm_model,D2S_cyy_testloader,threshold=0.38, is_clean_up=True, ep='f1_Max_lstm_test',mean_max='max',rouge=True)
# test_alphaBert(lstm_model,D2S_lin_testloader,threshold=0.38, is_clean_up=True, ep='f1_Max_lstm_test',mean_max='max',rouge=True)
test_alphaBert(lstm_model,D2S_all_testloader,threshold=0.38, is_clean_up=True, ep='f1_Max_lstm_test',mean_max='max',rouge=True)