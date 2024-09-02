import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torch import optim
from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import pipeline, set_seed

class JapaneseGenerator(nn.Module):
    def __init__(self): 
        nn.Module.__init__(self)
        self.generator = pipeline('text-generation', model='gpt2', pad_token_id = 50256)
        

    def forward(self, inputs):       
        generated_text = self.generator(inputs, max_length=32, num_return_sequences=1)[0]['generated_text']
        return generated_text


    
    
