import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
reference code
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """

'''

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BertLayerNorm(nn.Module):
    def __init__(self, config):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(config['hidden_size']))
        self.bias = nn.Parameter(torch.zeros(config['hidden_size']))
        self.variance_epsilon = config['eps']

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class alphabetEmbedding(nn.Module):
    def __init__(self, config, init_weight= None):
        super(alphabetEmbedding, self).__init__()
        if init_weight is None:
            self.alphabet_embeddings = nn.Embedding(config['input_size'], config['hidden_size'])
        else:
            self.alphabet_embeddings = nn.Embedding.from_pretrained(init_weight,freeze=False)
        
        self.position_embeddings = nn.Embedding(config['max_position_embeddings'], config['hidden_size'])
        self.LayerNorm = BertLayerNorm(config)
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, x, position_ids=None):
        seq_length = x.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand_as(x)
        
        alphabet_embeddings = self.alphabet_embeddings(x.long())
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = alphabet_embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class alphabetSelfAttention(nn.Module):
    def __init__(self, config):
        super(alphabetSelfAttention, self).__init__()

        self.num_attention_heads = config['num_attention_heads']
        self.attention_head_size = int(config['hidden_size'] / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config['hidden_size'], self.all_head_size)
        self.key = nn.Linear(config['hidden_size'], self.all_head_size)
        self.value = nn.Linear(config['hidden_size'], self.all_head_size)

        self.dropout = nn.Dropout(config['attention_probs_dropout_prob'])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = context_layer
        return outputs

class alphabetSelfOutput(nn.Module):
    def __init__(self, config):
        super(alphabetSelfOutput, self).__init__()
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.LayerNorm = BertLayerNorm(config)
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

def prune_linear_layer(layer, index, dim=0):
    """ Prune a linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer

class alphabetAttention(nn.Module):
    def __init__(self, config):
        super(alphabetAttention, self).__init__()
        self.self = alphabetSelfAttention(config)
        self.output = alphabetSelfOutput(config)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        for head in heads:
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        # Update hyper params
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads

    def forward(self, input_tensor,attention_mask):
        self_outputs = self.self(input_tensor,attention_mask)
        outputs = self.output(self_outputs, input_tensor)
        return outputs

class alphabetIntermediate(nn.Module):
    def __init__(self, config):
        super(alphabetIntermediate, self).__init__()
        self.dense = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class alphabetOutput(nn.Module):
    def __init__(self, config):
        super(alphabetOutput, self).__init__()
        self.dense = nn.Linear(config['intermediate_size'], config['hidden_size'])
        self.LayerNorm = BertLayerNorm(config)
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class alphabetLayer(nn.Module):
    def __init__(self, config):
        super(alphabetLayer, self).__init__()
        self.attention = alphabetAttention(config)
        self.intermediate = alphabetIntermediate(config)
        self.output = alphabetOutput(config)

    def forward(self, hidden_states,attention_mask):
        attention_output = self.attention(hidden_states,attention_mask)
        intermediate_output = self.intermediate(attention_output)
        outputs = self.output(intermediate_output, attention_output)
        return outputs    

class alphabetEncoder(nn.Module):
    def __init__(self, config):
        super(alphabetEncoder, self).__init__()
        self.layer = nn.ModuleList([alphabetLayer(config) for _ in range(config['num_hidden_layers'])])

    def forward(self, hidden_states,attention_mask):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states,attention_mask)

        outputs = hidden_states

        return outputs
    
class alphabetPooler(nn.Module):
    def __init__(self, config):
        super(alphabetPooler, self).__init__()
        self.avgpool = nn.Sequential(nn.Linear(config['hidden_size'],4*config['hidden_size']),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(4*config['hidden_size'],2*config['hidden_size']),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(2*config['hidden_size'],config['hidden_size']),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(config['hidden_size'],2)
                                     )
#        self.activation = nn.Sigmoid()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.avgpool(hidden_states)
        pooled_output = pooled_output.view(-1,2).contiguous()
#        pooled_output = self.activation(pooled_output)
        return pooled_output

class alphaBertModel(nn.Module):
    r"""
 
    """
    def __init__(self, config):
        super(alphaBertModel, self).__init__()
        self.embeddings = alphabetEmbedding(config)
        self.encoder = alphabetEncoder(config)
        self.pooler = alphabetPooler(config)

    def forward(self, input_ids,attention_mask, position_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            
        embedding_output = self.embeddings(input_ids)
        encoder_outputs = self.encoder(embedding_output,extended_attention_mask)
        pooled_output = self.pooler(encoder_outputs)
        
        return pooled_output,encoder_outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.transform_act_fn = gelu
        self.LayerNorm = BertLayerNorm(config)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config['hidden_size'],
                                 config['vocab_size'],
                                 bias=False)

        self.bias = nn.Parameter(torch.zeros(config['vocab_size']))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class alphaBertForMaskedLM(nn.Module):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        >>> config = BertConfig.from_pretrained('bert-base-uncased')
        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        >>> 
        >>> model = BertForMaskedLM(config)
        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        >>> outputs = model(input_ids, masked_lm_labels=input_ids)
        >>> loss, prediction_scores = outputs[:2]

    """
    def __init__(self, config):
        super(alphaBertForMaskedLM, self).__init__()
        self.config = config
        self.bert = alphaBertModel(config)
        self.cls = BertOnlyMLMHead(config)

#        self.apply(self.init_weights)
        self.tie_weights()


    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        first_module.weight = second_module.weight

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.alphabet_embeddings)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)

        sequence_output = outputs[1]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores, outputs)  # Add hidden states and attention is they are here
        if masked_lm_labels is not None:
            loss_fct =  nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config['vocab_size']), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


def test():
    config = {'hidden_size': 128,
              'max_position_embeddings':3000,
              'eps': 1e-12,
              'input_size': 82, 
              'hidden_dropout_prob': 0.1,
              'num_attention_heads': 16, 
              'attention_probs_dropout_prob': 0.2,
              'intermediate_size': 64,
              'num_hidden_layers': 12,
              }
    torch.manual_seed(seed = 0)
    x = torch.randint(0,69,[3,20])
    
    print('==test==',config)

if __name__ == '__main__':
#    test()
    config = {'hidden_size': 128,
              'max_position_embeddings':3000,
              'eps': 1e-12,
              'input_size': 82,
              'vocab_size':82,
              'hidden_dropout_prob': 0.1,
              'num_attention_heads': 16, 
              'attention_probs_dropout_prob': 0.2,
              'intermediate_size': 64,
              'num_hidden_layers': 12,
              }
    torch.manual_seed(seed = 0)
    x = torch.randint(0,69,[3,20])
    x = torch.cat([x.float(),torch.zeros([3,5]).float()],dim=1)
    m = torch.ones([3,20])
    m = torch.cat([m.float(),torch.zeros([3,5]).float()],dim=1)
    
    
    emb = alphabetEmbedding(config)
    enc = alphabetEncoder(config)
    
    absmodel = alphaBertModel(config)
    
    MLModel = alphaBertForMaskedLM(config)
    
    xx = emb(x)
#    y = enc(xx)
    z = absmodel(x,m)
    z2 = MLModel(input_ids=x,attention_mask=m,masked_lm_labels=x.long())