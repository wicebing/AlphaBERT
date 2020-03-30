# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 22:40:09 2019

@author: icebi
"""

import pandas as pd
import numpy as np

batch_size = 1
device = 'cpu'
parallel = False
#device_ids = list(range(rank * n, (rank + 1) * n))

all_expect_alphabets = [' ','0', '1', '2', '3', '4', '5', '6', '7', '8',  
 '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
 'y', 'z',]


res = 'C:\\Users\\icebi\\Downloads\\DD_test_ours_0302_long0.csv'
dat = pd.read_csv(res)

a = np.array(dat.iloc[:,[1,4]])

ds = []
for i, (d,s) in enumerate(a):
    d_list = list(d)
    df_list = []
    for c in d_list:
        if c in all_expect_alphabets:
            df_list.append(c)
        else:
            df_list.append(' ')
#    df_list = [c for c in d_list if c in all_expect_alphabets]
    df = ''.join(c for c in df_list)
    try:
        ds.append([len(df.split()),len(s.split())])
    except:
        ds.append([len(df.split()),0])
        
dsn = np.array(ds)

print(np.mean(dsn,axis =0))
print(np.std(dsn,axis =0))


from alphabert_utils import rouge12l_recheck
ab = np.array(dat.iloc[:,[4,5,6,7,8]])

se = np.array(dat.iloc[:,[9]])

ser = se.reshape(-1)

err = ser > 0
ner = ser ==0

for m in range(1,5):
    if m == 1:
        name = 'bert_'
    elif m == 2:
        name = 'biobert_'
    elif m == 3:
        name = 'lstm_'
    else:
        name = 'ours_'
    
    for d in range(4):
        if d ==0:
            dname = 'bing'
            dset = [j for j in range(250)]
            rouge_res = rouge12l_recheck(ab[:250,[0,m]])
        elif d ==1:
            dname = 'ying'
            dset = [j for j in range(251,498)]
            rouge_res = rouge12l_recheck(ab[250:499,[0,m]])
        elif d ==2:
            dname = 'lin'
            dset = [j for j in range(499,590)]
            rouge_res = rouge12l_recheck(ab[499:,[0,m]])
        else:
            dname = 'all'
            rouge_res = rouge12l_recheck(ab[:,[0,m]])
            
        
        rouge_res_pd = pd.DataFrame(rouge_res)
        #rouge_res_pd.to_csv('C:\\Users\\icebi\\Downloads\\rouge\\bert.csv',index=False)
        rouge_res_np = np.array(rouge_res_pd)
        
        filename = 'C:\\Users\\icebi\\Downloads\\rougelong\\'+name+dname+'.csv'
        pd.DataFrame(rouge_res_np.mean(axis =0)).to_csv(filename,index=False)


    for d in range(2):
        if d ==0:
            dname = 'noerr'
            rouge_res = rouge12l_recheck(ab[ner][:,[0,m]])
        else:
            dname = 'error'
            rouge_res = rouge12l_recheck(ab[err][:,[0,m]])            

        rouge_res_pd = pd.DataFrame(rouge_res)
        #rouge_res_pd.to_csv('C:\\Users\\icebi\\Downloads\\rouge\\bert.csv',index=False)
        rouge_res_np = np.array(rouge_res_pd)
        
        filename = 'C:\\Users\\icebi\\Downloads\\rougelong\\serror\\'+name+dname+'.csv'
        pd.DataFrame(rouge_res_np.mean(axis =0)).to_csv(filename,index=False)

#filename3 = 'C:\\Users\\icebi\\Downloads\\C2018_OUTPUT_CD.txt'
#data3 = pd.read_csv(filename3,encoding = 'utf8',sep='\t')
#ntuh_wd = np.array(data3['TYPECONTENT'])
#
#dsd = []
#for i, d in enumerate(ntuh_wd):
#    d_list = list(d)
#    df_list = []
#    for c in d_list:
#        if c in all_expect_alphabets:
#            df_list.append(c)
#        else:
#            df_list.append(' ')
##    df_list = [c for c in d_list if c in all_expect_alphabets]
#    df = ''.join(c for c in df_list)
#    if len(df.split()) > 0:
#        dsd.append([len(df.split()),len(df_list)])
#
#dsdn = np.array(dsd)
#
#print(np.mean(dsdn,axis =0))
#print(np.std(dsdn,axis =0))