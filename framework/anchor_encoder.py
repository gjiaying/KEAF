import torch
import torch.nn as nn
from transformers import BertTokenizer
from transformers import AutoModel, AutoTokenizer, BertModel

class BERTJapaneseEncoder2(nn.Module):
    '''
    BERT Encoder in Japanese
    '''

    def __init__(self, max_length): 
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, inputs):
        x = self.bert(inputs['text'], attention_mask=inputs['mask'])
        return x[1]
    
    