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
from alphabert_utils import make_statistics, rouge12l,save_checkpoint,load_checkpoint

from transformers import tokenization_bert, modeling_bert, configuration_bert
import bert_cls_Model

try:
    task = sys.argv[1]
    print('*****task= ',task)
except:
    task = 'test'

batch_size = 8
device = 'cuda'
parallel = True
bioBERT = False

if task == 'bio':
    bioBERT=True

if bioBERT:
    checkpoint_DIR = '../checkpoint_biobert'
    checkpoint_path = 'biobert_pretrain.pth'
    print( '=====  bioBERT  =====' )
else:
    checkpoint_DIR = '../checkpoint_bert'
    checkpoint_path = 'bert_pretrain.pth'
    print( '=====  bert  =====' )

#def save_checkpoint(checkpoint_path, model, optimizer):
#    global checkpoint_DIR
#    state = {'state_dict': model.state_dict(),
#             'optimizer' : optimizer.state_dict()}
#    torch.save(state, os.path.join(checkpoint_DIR,checkpoint_path))
#
#    print('model saved to %s' % checkpoint_path)
#    
#def load_checkpoint(checkpoint_path, model, parallel=parallel):
#    global checkpoint_DIR
#    state = torch.load(os.path.join(checkpoint_DIR,checkpoint_path),
##                       map_location={'cuda:0':'cuda:1'}
#                       )
#    if parallel:
#        try:
#            from collections import OrderedDict
#            new_state_dict = OrderedDict()
#            for k, v in state['state_dict'].items():
#                name = k[7:] # remove module.
#                new_state_dict[name] = v
#            model.load_state_dict(new_state_dict)
#            print(' === load from parallel model === ')
#        except:
#            model.load_state_dict(state['state_dict'])
#            print(' No parallel === load from model === ')
#    else:
#        try:
#            model.load_state_dict(state['state_dict'])
#        except:
#            from collections import OrderedDict
#            new_state_dict = OrderedDict()
#            for k, v in state['state_dict'].items():
#                name = k[7:] # remove module.
#                new_state_dict[name] = v
#            model.load_state_dict(new_state_dict)
#            print(' === load from parallel model === ')            
##    optimizer.load_state_dict(state['optimizer'])
#    print('model loaded from %s' % checkpoint_path)
#    name_='WW_'+checkpoint_path
#    torch.save(model,os.path.join(checkpoint_DIR,name_))
#    print('model saved to %s / %s' % (checkpoint_DIR,name_))
#    return model

#alphabert  904405 parameters 
#lstm       899841 parameters    
#bert    108523714 parameters
#bioBERT 108523714 parameters
pretrained_weights = 'bert-base-cased'
bert_model = bert_cls_Model.bert_baseModel(bioBERT=bioBERT)
bert_tokenizer = tokenization_bert.BertTokenizer.from_pretrained(pretrained_weights)
try:      
#    bert_model = load_checkpoint('bert_pretrain.pth',bert_model,parallel=parallel)
    bert_model = load_checkpoint(checkpoint_DIR,checkpoint_path, bert_model)
except:
    print('*** No Pretrain_Model ***')
    pass


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

bert_dataset = alphabert_dataset_v02.D_bert(D_L_train,
                                            tokenize_alphabets,
                                            bert_tokenizer,
                                            clamp_size=config['max_position_embeddings'],
                                            train=True)
bert_dataloader = DataLoader(bert_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4,
                             collate_fn=alphabert_dataset_v02.collate_fn)

bert_dataset_val = alphabert_dataset_v02.D_bert(D_L_val,
                                                 tokenize_alphabets,
                                                 bert_tokenizer,
                                                 clamp_size=config['max_position_embeddings'],
                                                 train=False)

bert_dataset_valloader = DataLoader(bert_dataset_val,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=4,
                           collate_fn=alphabert_dataset_v02.collate_fn)

bert_dataset_test = alphabert_dataset_v02.D_bert(D_L_test,
                                     tokenize_alphabets,
                                     bert_tokenizer,
                                     clamp_size=config['max_position_embeddings'],
                                     train=False)

bert_dataset_testloader = DataLoader(bert_dataset_test,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=4,
                           collate_fn=alphabert_dataset_v02.collate_fn)

filename1cyy = os.path.join(filepath,'finetune_test_cyy.csv')
data_test_cyy = pd.read_csv(filename1cyy, header=None, encoding='utf-8')
D_L_test_cyy = np.array(data_test_cyy)[:,[1,3]]

D2S_test_cyy = alphabert_dataset_v02.D_bert(D_L_test_cyy,
                                     tokenize_alphabets,
                                     bert_tokenizer,
                                     clamp_size=config['max_position_embeddings'],
                                     train=False)

D2S_cyy_testloader = DataLoader(D2S_test_cyy,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=8,
                           collate_fn=alphabert_dataset_v02.collate_fn)

filename1lin= os.path.join(filepath,'finetune_test_lin.csv')
data_test_lin = pd.read_csv(filename1lin, header=None, encoding='utf-8')
D_L_test_lin = np.array(data_test_lin)[:,[1,3]]

D2S_test_lin = alphabert_dataset_v02.D_bert(D_L_test_lin,
                                     tokenize_alphabets,
                                     bert_tokenizer,
                                     clamp_size=config['max_position_embeddings'],
                                     train=False)

D2S_lin_testloader = DataLoader(D2S_test_lin,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=8,
                           collate_fn=alphabert_dataset_v02.collate_fn)

D_L_test_all = np.concatenate([D_L_test,D_L_test_cyy,D_L_test_lin],axis=0)

D2S_test_all = alphabert_dataset_v02.D_bert(D_L_test_all,
                                     tokenize_alphabets,
                                     bert_tokenizer,
                                     clamp_size=config['max_position_embeddings'],
                                     train=False)

D2S_all_testloader = DataLoader(D2S_test_all,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=8,
                           collate_fn=alphabert_dataset_v02.collate_fn)


def test_BERT(DS_model,dloader,threshold=0.5, is_clean_up=True, ep=0, train=False,mean_max='mean',rouge=False, parallel=parallel):
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
                all_pred_trg['pred'].append(pred_prop[i][cls_idx[i]].cpu())
                all_pred_trg['trg'].append(trg[i][cls_idx[i]].cpu())                               
                if rouge:
#                    src_split, src_isword = split2words(src_,rouge=rouge,tokenize_alphabets=tokenize_alphabets)
                    referecne = []
                    hypothesis = []
                    isselect_pred = False
                    isselect_trg = False
                    for j, wp in enumerate(src_):
                        if wp == 101:
                            if pred_prop[i][j]  > 0.5:
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
                    
                    a_ = bert_tokenizer.convert_ids_to_tokens(src_[:origin_len[i]].detach().cpu().numpy())
                    a_ = bert_tokenizer.convert_tokens_to_string(a_)
                    out_pred_res.append((a_,hypothesis,referecne))
                    
            print(batch_idx, len(dloader))
    out_pd_res = pd.DataFrame(out_pred_res)
    out_pd_res.to_csv('./iou_pic/bert/test_pred.csv', sep=',')
    
    make_statistics(all_pred_trg, ep=ep)

    DS_model.train()
    
    if rouge:
        rouge_res = rouge12l(rouge_set)
        rouge_res_pd = pd.DataFrame(rouge_res)
        rouge_res_pd.to_csv('./iou_pic/bert/rouge_res.csv',index=False)
        rouge_res_np = np.array(rouge_res_pd)
        
        pd.DataFrame(rouge_res_np.mean(axis =0)).to_csv('./iou_pic/bert/rouge_res_mean.csv',index=False)

def finetune_BERT(DS_model,dloader,lr=1e-4,epoch=10,log_interval=20,parallel=parallel):
    DS_model.to(device)
    model_optimizer = optim.Adam(DS_model.parameters(), lr=lr)
    if parallel:
        DS_model = torch.nn.DataParallel(DS_model)
    DS_model.train()
    
#    criterion = nn.MSELoss().to(device)
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
            trg = trg.float().to(device)
            att_mask = att_mask.float().to(device)
            origin_len = origin_len.to(device)
        
            pred_prop = DS_model(input_ids=src.long(),
                                 attention_mask=att_mask)
#            print(pred_prop.shape, trg.shape)
            loss = criterion(pred_prop,trg.view(-1).contiguous().long())

#            print(pred_prop.shape, trg.view(-1).shape)
#            return pred_prop, trg
            
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
#                save_checkpoint('bert_pretrain.pth',DS_model,model_optimizer)
                save_checkpoint(checkpoint_DIR,checkpoint_path, DS_model, model_optimizer, parallel)
                    
            iteration +=1
        if ep % 1 ==0:
            save_checkpoint(checkpoint_DIR,checkpoint_path, DS_model, model_optimizer, parallel)
#            test_alphaBert(DS_model,D2S_valloader, 
#                           is_clean_up=True, ep=ep,train=True)

            print('======= epoch:%i ========'%ep)
            
#        print('total loss: {:.4f}'.format(total_loss/len(dloader)))         
        print('++ Ep Time: {:.1f} Secs ++'.format(time.time()-t0)) 
#        total_loss.append(epoch_loss)
        total_loss.append(float(epoch_loss/epoch_cases))
        pd_total_loss = pd.DataFrame(total_loss)
        pd_total_loss.to_csv('./iou_pic/bert/total_loss_finetune.csv', sep = ',')
    print(total_loss)

#finetune_BERT(bert_model,bert_dataloader,lr=1e-5,epoch=200,log_interval=3,parallel=parallel)


# test_BERT(bert_model,bert_dataset_valloader, is_clean_up=True, ep='f1_bert_val',mean_max='max',rouge=True)
# test_BERT(bert_model,bert_dataset_testloader,threshold=0.45, is_clean_up=True, ep='f1_bert_test',mean_max='max',rouge=True)
# test_BERT(bert_model,D2S_cyy_testloader,threshold=0.45, is_clean_up=True, ep='f1_bert_cyy',mean_max='max',rouge=True)
# test_BERT(bert_model,D2S_lin_testloader,threshold=0.45, is_clean_up=True, ep='f1_bert_cyy',mean_max='max',rouge=True)
# test_BERT(bert_model,D2S_all_testloader,threshold=0.45, is_clean_up=True, ep='f1_bert_cyy',mean_max='max',rouge=True)