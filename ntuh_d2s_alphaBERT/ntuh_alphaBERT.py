import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
#import apex
#from apex import amp
import torch.distributed

#from gensim.models import Word2Vec
import alphabert_dataset_ntuh as alphabert_dataset
import dg_alphabets
from alphabert_utils import clean_up_v204_ft, split2words, rouge12l, make_statistics,save_checkpoint,load_checkpoint,clean_up
from alphabert_utils import count_parameters,clean_up_ensemble
from alphabert_dadaloader import alphabet_loaders
#from transformers import tokenization_bert, modeling_bert, configuration_bert

import json
import flask
from flask import jsonify

batch_size = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
parallel = False
#device_ids = list(range(rank * n, (rank + 1) * n))

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

class test_loaders():
    def __init__(self, datapath, config, tokenize_alphabets, num_workers=4, batch_size=2):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.config = config
        self.tokenize_alphabets = tokenize_alphabets
        self.filepath = datapath
        self.clamp_size = config['max_position_embeddings']
       
    def get_data_np(self):
        filename1a = os.path.join(self.filepath,'DD_test_ours.csv')
        data_a = pd.read_csv(filename1a, encoding='utf-8')
        D_L_test = np.array(data_a)[:,[1,0,3,4]]
        return D_L_test,data_a
    
filepath = './checkpoint_exe'

def json2np(Djs):
    if type(Djs)== str:
        Ddit = json.loads(Djs)
    else:
        Ddit = Djs
    Dpd = pd.DataFrame(Ddit,index=[0])
    Dnp = np.array(Dpd).T
    return Dnp

def run_d2s(DS_model,
            dloader,
            threshold=0.51,
            mean_max = 'mean'):
    
    leading_token_idx = tokenize_alphabets.alphabet2idx['|']
    padding_token_idx = tokenize_alphabets.alphabet2idx[' ']
    out_pred_res = []
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(dloader): 
            print(batch_idx)
            src = sample['src_token']
            att_mask = sample['mask_padding']
                
            src = src.float().to(device)
            att_mask = att_mask.float().to(device)
            
            bs = src.shape          
#            pred_prop_bin, = DS_model(input_ids=src,
#                                 attention_mask=att_mask,
#                                 out = 'finehead')
            pooled_output, = DS_model(input_ids=src,
                                 attention_mask=att_mask,
                                 out='finehead')
            pred_prop_bin = pooled_output.view(*bs,-1)
            
            pred_prop = clean_up_v204_ft(src,pred_prop_bin,
                                         tokenize_alphabets=tokenize_alphabets,
                                         mean_max=mean_max)   
            
            pred_prop2json = pred_prop > threshold
            pred_prop2json = pred_prop2json.int()
            
            srcstr = tokenize_alphabets.convert_idx2str(src[0])
            print('srcstr', srcstr)
#            print('00000', len(src[0]),len(pred_prop[0]),len(pred_prop2json[0]))
            for i, src_ in enumerate(src):
                hypothesis = []
                isselect_pred = False

                src_split, src_isword = split2words(src_,
                                                    tokenize_alphabets=tokenize_alphabets,
                                                    rouge=True)

                for j in range(len(src_split)):
                    if src_isword[j]>0:
                        if pred_prop[i][src_split[j]][0].cpu()>threshold:
                            hypothesis.append(tokenize_alphabets.convert_idx2str(src_[src_split[j]]))
                                      
                s_ = ''.join(i+' ' for i in hypothesis)
                
                srcstr = tokenize_alphabets.convert_idx2str(src_)
    
                out_pred_res.append([srcstr,s_,pred_prop[i].cpu().numpy()])
    
        
     
    out_np_res = np.array(out_pred_res) 
#    out_pred_BERT = pd.DataFrame(out_pred_res)
#    
#    out_json = out_pred_BERT.to_json()
#    print(out_json)
    return out_np_res



def for_YuShan():
    modelpath= os.path.join(filepath,'W_d2s_total_0302_947.pth')
    DS_model = torch.load(modelpath)
    DS_model.to(device)
    DS_model.eval()

    path = './test_data'
    loaders = test_loaders(path,
                           config, 
                           tokenize_alphabets,
                           num_workers=4, 
                           batch_size=1)
    Dnp, _ = loaders.get_data_np()

    Ds = alphabert_dataset.D2Lntuh_json(Dnp,
                                        tokenize_alphabets,
                                        clamp_size=config['max_position_embeddings'],
                                        train=False)
    
    Dl= DataLoader(Ds,
                   batch_size=batch_size,
                   shuffle=False,
                   num_workers=0,
                   collate_fn=alphabert_dataset.collate_fn_json)
    
    predict_np_res = run_d2s(DS_model= DS_model,
                             dloader=Dl,
                             threshold=0.51)
    
    predict_pd_res = pd.DataFrame(predict_np_res)
    predict_pd_res.to_csv('./predict_results/predict_keywrods.csv',index=None, header=None)

    return predict_np_res,Dnp
    
def draw_predict(dat,case_num,rougelcsf):
    txt, _, prob, label, _ = dat    
    l = len(txt)

    pic = np.ndarray((1150,1440,3),dtype='uint8')
    pic[:,:] = [255,255,255]
    cha_x = 20
    cha_y = 50
    line_char_max = 70
   
    def color_rank(prob):
        255 - prob*2
        return [255,255-prob*2,255-prob*2]
   
    def prob2uint(iprob):
        return int(iprob*100)
    
    for i in range(l):
        jj = int(i/line_char_max)
        ii = i % line_char_max
        try:
            if label[i]=='1':
                pic[10+cha_y*jj:30+cha_y*jj,
                    20+cha_x*ii:20+cha_x*(ii+1)] = [0,255,255]
        except:
            pass
           
        pic[30+cha_y*jj:50+cha_y*jj,
            20+cha_x*ii:20+cha_x*(ii+1)] = color_rank(prob2uint(prob[i]))
    
    
    legend_y = 20+50*(jj+1)
    legend_x = 740
#    legend_y = 980
    
    pic[legend_y+30:legend_y+40,legend_x+270:legend_x+620] = [0,255,255]

    for i in range(100):
        pic[legend_y+80:legend_y+90,legend_x+20+6*i:legend_x+20+6*(i+1)] = color_rank(i)
           
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,10), dpi=(120))
    ax = fig.add_subplot(111)
    ax.imshow(pic)
   
    rect = plt.Rectangle((legend_x,legend_y+50),650,60,edgecolor='black',facecolor='none')
    ax.add_patch(rect)
    ax.text(1025,legend_y+70,'Probability',fontsize=7,style='italic')
    rect = plt.Rectangle((legend_x+240,legend_y),410,50,edgecolor='black',facecolor='none')
    ax.add_patch(rect)
    ax.text(1140,legend_y+25,'Reference',fontsize=7,style='italic')
    rect = plt.Rectangle((legend_x,legend_y),240,50,edgecolor='black',facecolor='none')
    ax.add_patch(rect)
    ax.text(760,legend_y+32,'Rouge-L: {:.3f}'.format(rougelcsf),fontsize=9,style='italic')

#    rect = plt.Rectangle((740,1030),650,60,edgecolor='black',facecolor='none')
#    ax.add_patch(rect)
#    ax.text(1025,1050,'Probability',fontsize=7,style='italic')
#    rect = plt.Rectangle((980,980),410,50,edgecolor='black',facecolor='none')
#    ax.add_patch(rect)
#    ax.text(1140,1005,'Reference',fontsize=7,style='italic')
#    rect = plt.Rectangle((740,980),240,50,edgecolor='black',facecolor='none')
#    ax.add_patch(rect)
#    ax.text(760,1012,'Rouge-L: {:.3f}'.format(rougelcsf),fontsize=9,style='italic')
   
    for i in range(11):
        if i < 1:
            ax.text(760+60*i,legend_y+105,'0.0',fontsize=6)
        else:
            num = str(float(i/10))
            ax.text(760+60*i,legend_y+105,num,fontsize=6)
   
    for i in range(l):
        jj = int(i/line_char_max)
        ii = i % line_char_max
        ax.text(30+cha_x*(ii),
                20+cha_y*jj,
                txt[i],
                fontsize=12,
                color='black',
                horizontalalignment='center',
                verticalalignment='center')
        
    plt.axis('off')
   
#    plt.plot()
    plt.savefig('./ana_pic/pred_'+str(case_num)+'_rf_'+str(int(1000*rougelcsf))+'.png')

def test_ana():
    import random
    a = np.linspace(0,1,200)
    random.shuffle(a)
    dat = ['abcdefghij klmn op qr  tuvxwyz ABC;DOEDPSSKSSSZXzXZz   123456789012345678901234567890123456789012345678901234567890',
           a,
           a,
           '0001110111011110111111111111111111111000000000000000000101011110000001111000000001111111110000000011111111000000000',
           a,
           ]
    draw_predict(dat,0,0)
#test_ana()
    
if __name__ == "__main__":
    predict_np_res, Dnp = for_YuShan()
    
    predict_analysis = np.concatenate([predict_np_res,Dnp[:,[2,3]]],axis=1)
    
    from alphabert_utils import rouge12l_recheck
    rouge_res = rouge12l_recheck(predict_analysis[:,[4,1]])
    rouge_score = rouge_res['rouge-lcsf']
    
    for i in range(len(predict_analysis)):
        draw_predict(predict_analysis[i],i,rouge_score[i])
    
    pass


        


