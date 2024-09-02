import sys
sys.path.append('..')
import framework
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import math

class Siamese(framework.frameworks.MLFSModel):
    
    def __init__(self, encoder, max_length, weight_factor, device_ids, att):
        framework.frameworks.MLFSModel.__init__(self, encoder)
        self.drop = nn.Dropout()
        self.max_length = max_length
        self.weight_factor = weight_factor
        self.encoder = encoder
        self.encoders = nn.DataParallel(self.encoder, device_ids=device_ids)
        self.att = att
        self.linear = nn.Linear(768, 768, bias=True)
        self.normalize = nn.LayerNorm(normalized_shape=768)
        
    def threshold_calculation(self, support_label, logits, N, Q):
        '''
        Input: support_labels, logits
        Output: prediction
        '''
        nums = torch.sum(support_label.float(), dim=3)
        nums = torch.mean(nums, dim=2)
        nums = torch.round(nums) #(B, N)
        x = torch.zeros(support_label.size(dim=0), N, Q, N)
        for i in range(support_label.size(dim=0)):
            for j in range(N):
                for k in range(Q):
                    count = int(nums[i][j].item())
                    _, indice = torch.topk(logits[i][j], k=count)
                    indice = F.one_hot(indice, num_classes=N)
                    indice = torch.sum(indice, dim=0)
                    x[i][j][k] = indice

        x = x.view(-1, N*Q, N)
        
        return x

    def forward(self, support, query, N, K, Q, anchor, support_label):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''

        support = self.encoders(support)
        query = self.encoders(query)
        hidden_size = support.size(-1)

        # Layer Norm
        support = self.normalize(support)
        query = self.normalize(query)
        
        # Dropout ?
        support = self.drop(support)
        query = self.drop(query)
        

        support = support.view(-1, N * K, hidden_size) # (B, N * K, D)
        query = query.view(-1, N * Q, hidden_size) # (B, total_Q, D)
        B = support.size(0) # Batch size
        support = support.unsqueeze(1) # (B, 1, N * K, D)
        query = query.unsqueeze(2) # (B, total_Q, 1, D)

        #  Dot production
        z = (support * query).sum(-1) # (B, total_Q, N * K)
        z = z.view(-1, N * Q, N, K) # (B, total_Q, N, K)

        # Max combination
        logits = z.max(-1)[0] # (B, total_Q, N)

        # NA
        #minn, _ = logits.min(-1)
        #logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2) # (B, total_Q, N + 1)

        #_, pred = torch.max(logits.view(-1, N+1), 1)
        support_label = support_label.view(-1, N, K, self.max_length)
        pred = self.threshold_calculation(support_label, logits, N, Q)
        return logits, pred 
