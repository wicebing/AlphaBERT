import numpy as np

class Diagnosis_alphabets:
    def __init__(self):
        self.alphabet2idx ={}
        self.alphabet2count = {}
        self.idx2alphabet = {}
        self.n_alphabets = 0
    
    def addSentence(self,sentence):
        s_list = list(sentence)
        for ab in s_list:
            self.addalphabet(ab)
 
    def addalphabet(self,alphabet):
        if alphabet not in self.alphabet2idx:        
            self.alphabet2idx[alphabet] = self.n_alphabets
            self.alphabet2count[alphabet] = 1
            self.idx2alphabet[self.n_alphabets] = alphabet
            self.n_alphabets += 1
        else:
            self.alphabet2count[alphabet] += 1
            
    def tokenize(self,alphabet_list):
        token_list = []
        
        for i, a in enumerate(alphabet_list):
            if a not in self.alphabet2idx.keys():
                token_list.append(0)
            else:
                token_list.append(self.alphabet2idx[a])
        return np.array(token_list,dtype='int')
    
    def convert_idx2str(self,idx_np,head=False):
        res = ''
        for idx in idx_np:
            if head:
                if idx == self.alphabet2idx['|']:
                    res += ' '
                else:
                    res += self.idx2alphabet[int(idx)]
            else:
                res += self.idx2alphabet[int(idx)]
        return res
    
    def convert_idx2str_head(self,idx_np,head_mask,head_prob,trg_prob):
        res = ''
        for i,idx in enumerate(idx_np):
            if head_mask[i]:
                if trg_prob[i].item() > -1:
                    prob = '{:.2f}'.format(head_prob[i].item())
                    prob_t = '{:d}'.format(trg_prob[i].item())
                    res += ' '+'['+prob+'|'+prob_t+']'
                else:
                    res += ' '
            else:
                res += self.idx2alphabet[int(idx)]
        return res        