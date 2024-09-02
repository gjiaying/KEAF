import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torch import optim
from transformers import BertTokenizer
from transformers import AutoModel, AutoTokenizer, BertModel

class BERTJapaneseEncoder(nn.Module):
    '''
    BERT Encoder in Japanese
    '''

    def __init__(self, max_length, att, category): 
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.att = att
        self.category = category

    def forward(self, inputs, mark='false'):
        if mark == 'SimpleFSRE':
            x = self.bert(inputs['text'], attention_mask=inputs['mask'])
            return x['pooler_output'], x['last_hidden_state']
        
        x = self.bert(inputs['text'], attention_mask=inputs['mask'])
        if mark == 'True':
            taxonomy = self.bert(inputs['taxonomy'], attention_mask=inputs['mask_t']) 
            return x[1], taxonomy[1]
        else:
            return x[1]
    
    def tokenize(self, raw_tokens, label, raw_category, raw_taxonomy):
        # token -> index
        if self.category == 'category':
            raw_category = '[CLS]' + raw_category + '[SEP]'
            token_dict = self.tokenizer(raw_category)
            token_ids = token_dict.input_ids
            raw_tokens = raw_tokens.split(' ')
            indexed_token = self.tokenizer.convert_tokens_to_ids(raw_tokens)
            indexed_tokens = token_ids + indexed_token
        elif self.category == 'taxonomy':
            raw_taxonomy = '[CLS]' + raw_taxonomy + '[SEP]'
            token_dict = self.tokenizer(raw_taxonomy)
            token_ids = token_dict.input_ids
            raw_tokens = raw_tokens.split(' ')
            indexed_token = self.tokenizer.convert_tokens_to_ids(raw_tokens)
            indexed_tokens = token_ids + indexed_token
        else:
            raw_tokens = '[CLS] ' + raw_tokens
            raw_tokens = raw_tokens.split(' ')
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(raw_tokens)
        
        # padding, pad zeros when length < maximum length
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(indexed_tokens)] = 1

        #labels
        labels =np.zeros((self.max_length), dtype=np.int32)
        labels[:len(label)] = label

        # taxonomy -> index
        taxonomy = '[CLS]' + raw_category + '[SEP]' + raw_taxonomy
        token_dict = self.tokenizer(taxonomy)
        taxonomy_tokens = token_dict.input_ids
        
        while len(taxonomy_tokens) < self.max_length:
            taxonomy_tokens.append(0)
        taxonomy_tokens = taxonomy_tokens[:self.max_length]
        mask_t = np.zeros((self.max_length), dtype=np.int32)
        mask_t[:len(taxonomy_tokens)] = 1

        return indexed_tokens, mask, labels, taxonomy_tokens, mask_t

    def anchor_tokenize(self, raw_tokens, anchor_length):
        # token -> index
        token_dict = self.tokenizer(raw_tokens)
        token_ids = token_dict.input_ids
        
        # padding, pad zeros when length < maximum length
        while len(token_ids) < anchor_length:
            token_ids.append(0)
        token_ids = token_ids[:anchor_length]

        # mask
        mask = np.zeros((anchor_length), dtype=np.int32)
        mask[:len(token_ids)] = 1
        return token_ids, mask
