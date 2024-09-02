import sys
sys.path.append('..')
import framework
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class Mtb(framework.frameworks.MLFSModel):
    """
    Use the same few-shot model as the paper "Matching the Blanks: Distributional Similarity for Relation Learning".
    """
    
    def __init__(self, encoder, max_length, weight_factor, device_ids, att):
        framework.frameworks.MLFSModel.__init__(self, encoder)
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        self.use_dropout = True
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=768)
        self.combiner = 'max'
        self.max_length = max_length
        self.weight_factor = weight_factor
        self.encoder = encoder
        self.encoders = nn.DataParallel(self.encoder, device_ids=device_ids)
        self.att = att

    def __dist__(self, x, y, dim):
        return (x * y).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)
    
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
        if self.use_dropout:
            support = self.drop(support)
            query = self.drop(query)
        support = self.layer_norm(support)
        query = self.layer_norm(query)
        support = support.view(-1, N, K, hidden_size).unsqueeze(1) # (B, 1, N, K, D)
        query = query.view(-1, N*Q, hidden_size).unsqueeze(2).unsqueeze(2) # (B, total_Q, 1, 1, D)

        logits = (support * query).sum(-1) # (B, total_Q, N, K)

        # aggregate result
        if self.combiner == "max":
            combined_logits, _ = logits.max(-1) # (B, total, N)
        elif self.combiner == "avg":
            combined_logits = logits.mean(-1) # (B, total, N)
        else:
            raise NotImplementedError
        #_, pred = torch.max(combined_logits.view(-1, N), -1)
        support_label = support_label.view(-1, N, K, self.max_length)
        pred = self.threshold_calculation(support_label, combined_logits, N, Q)

        return combined_logits, pred
    
    
