import sys
sys.path.append('..')
import framework
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import math


class MLFSModel(framework.frameworks.MLFSModel):
    
    def __init__(self, encoder, max_length, weight_factor, device_ids, att, label_att, shots):
        framework.frameworks.MLFSModel.__init__(self, encoder)
        self.drop = nn.Dropout()
        self.max_length = max_length
        self.weight_factor = weight_factor
        self.encoder = encoder
        self.encoders = nn.DataParallel(self.encoder, device_ids=device_ids)
        self.att = label_att
        self.tax = att
        self.linear = nn.Linear(768, 768, bias=True)
        
        self.conv1 = nn.Conv2d(1, 32, (shots, 1), padding=(shots // 2, 0))
        self.conv2 = nn.Conv2d(32, 64, (shots, 1), padding=(shots // 2, 0))
        self.conv_final = nn.Conv2d(64, 1, (shots, 1), stride=(shots, 1))
        

    def __dist__(self, x, y, dim, score=None):
        if score is None:
            return (torch.pow(x - y, 2)).sum(dim)
        else:
            return (torch.pow(x - y, 2) * score).sum(dim)
        

    def __batch_dist__(self, S, Q, score=None):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3, score)



    
    
    
    def prototype_calculation(self, support, anchor, support_label):
        '''
        Input: support (B, N * K, D)
        Output: prototype (B, N, D)
        '''
        #for i in range(support.size(dim=0)): #batch size
        #Prototype calculation by averaging support
        support_label = torch.narrow(support_label, 2, 0, anchor.size(dim=1))
        support_label = support_label.permute(0, 2, 1)
        weights = torch.sum(support_label, 2)
        
        prototypes = torch.matmul(support_label.float(), support)
        prototypes = prototypes.permute(0, 2, 1)
        weights = weights.unsqueeze(1)
        prototypes = torch.div(prototypes, weights)  #final prototypes without anchor information
        prototypes_scaled = torch.mul(prototypes, (1-self.weight_factor))

        anchor = self.linear(anchor)
        anchor = torch.mul(anchor, self.weight_factor)
        anchor = anchor.permute(0, 2, 1)
        prototype = torch.add(prototypes_scaled, anchor)
        prototype = prototype.permute(0, 2, 1)

        return prototype

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

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output    

    def forward(self, support, query, N, K, Q, anchor, support_label):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        anchor: embedding of anchors
        '''
        support, taxonomy_s = self.encoders(support, 'True')
        query, taxonomy_q = self.encoders(query, 'True')
        anchor = self.encoders(anchor)
        hidden_size = support.size(-1)
        
        if self.att == 'cross':
            taxonomy_s = taxonomy_s.view(-1, N * K, hidden_size)
            taxonomy_q = taxonomy_q.view(-1, N * Q, hidden_size)
            support = support.view(-1, N * K, hidden_size) # (B, N * K, D)
            query = query.view(-1, N * Q, hidden_size) # (B, N * Q, D)
            
            support = self.attention(taxonomy_s, support, support, hidden_size)
            query = self.attention(taxonomy_q, query, query, hidden_size)
        
            
        
        
        support = support.view(-1, N * K, hidden_size) # (B, N * K, D)
        query = query.view(-1, N * Q, hidden_size) # (B, N * Q, D) 
        #support = support.view(-1, N, K, hidden_size).unsqueeze(1)

        support_labels = support_label.view(-1, N * K, self.max_length)
        anchor = anchor.view(-1, N, hidden_size)
        taxonomy_s = taxonomy_s.view(-1, N, hidden_size)
        

        # Prototypical Networks      
        support_proto_raw = self.prototype_calculation(support, anchor, support_labels) # (B, N, D)
        support_i = support_proto_raw.unsqueeze(2).repeat(1, 1, K, 1) # (B, N, K, D)
        
        
        
        
        if self.att == 'cat':
            anchor_i = taxonomy_s.unsqueeze(2).repeat(1, 1, K, 1) # (B, N, K, D)
        else:
            anchor_i = anchor.unsqueeze(2).repeat(1, 1, K, 1) # (B, N, K, D)

        
        if self.att == 'fea':
            # feature-level attention
            B = support.size(0)
            fea_att_score = support_is.view(B * N, 1, K, hidden_size) # (B * N, 1, K, D)
            fea_att_score = F.relu(self.conv1(fea_att_score)) # (B * N, 32, K, D) 
            fea_att_score = F.relu(self.conv2(fea_att_score)) # (B * N, 64, K, D)
            fea_att_score = self.drop(fea_att_score)
            fea_att_score = self.conv_final(fea_att_score) # (B * N, 1, 1, D)
            fea_att_score = F.relu(fea_att_score)
            fea_att_score = fea_att_score.view(B, N, hidden_size).unsqueeze(1) # (B, 1, N, D)
            logits = -self.__batch_dist__(support_proto_raw, query, fea_att_score)
        elif self.att == 'ins':
            #instance-level attention
            query_i = query.view(-1, N * Q, hidden_size) # (B, N * Q, D)
            ins_att_score_a = F.cosine_similarity(support_i, anchor_i, dim=3) #(B, N, K) alpha
            ins_att_score_a = ins_att_score_a.unsqueeze(3).repeat(1, 1, 1, hidden_size) #(B, N, K, D)
            support_is = torch.mul(support_i, ins_att_score_a)#(B, N, K, D)

            #support_is = support_is.unsqueeze(1).expand(-1, N*Q, -1, -1, -1) # (B, NQ, N, K, D)
            support_for_att = self.linear(support_is) #(B, N, K, D)
            query_for_att = self.linear(query_i.view(-1, N, Q, hidden_size)) #(B, N, Q, D)
            ins_att_score_b = F.softmax(torch.tanh(support_for_att * query_for_att).sum(-1), dim=-1) # (B, N, K) beta
            support_proto = (support_is * ins_att_score_b.unsqueeze(3).expand(-1, -1, -1, hidden_size)).sum(2) # (B, N, D)
            logits = -self.__batch_dist__(support_proto, query)
        elif self.att == 'all':
            #instance-level attention
            query_i = query.view(-1, N * Q, hidden_size) # (B, N * Q, D)
            ins_att_score_a = F.cosine_similarity(support_i, anchor_i, dim=3) #(B, N, K) alpha
            ins_att_score_a = ins_att_score_a.unsqueeze(3).repeat(1, 1, 1, hidden_size) #(B, N, K, D)
            support_is = torch.mul(support_i, ins_att_score_a)#(B, N, K, D)

            #support_is = support_is.unsqueeze(1).expand(-1, N*Q, -1, -1, -1) # (B, NQ, N, K, D)
            support_for_att = self.linear(support_is) #(B, N, K, D)
            query_for_att = self.linear(query_i.view(-1, N, Q, hidden_size)) #(B, N, Q, D)
            ins_att_score_b = F.softmax(torch.tanh(support_for_att * query_for_att).sum(-1), dim=-1) # (B, N, K) beta
            support_proto = (support_is * ins_att_score_b.unsqueeze(3).expand(-1, -1, -1, hidden_size)).sum(2) # (B, N, D)
            # feature-level attention
            B = support.size(0)
            fea_att_score = support_is.view(B * N, 1, K, hidden_size) # (B * N, 1, K, D)
            fea_att_score = F.relu(self.conv1(fea_att_score)) # (B * N, 32, K, D) 
            fea_att_score = F.relu(self.conv2(fea_att_score)) # (B * N, 64, K, D)
            fea_att_score = self.drop(fea_att_score)
            fea_att_score = self.conv_final(fea_att_score) # (B * N, 1, 1, D)
            fea_att_score = F.relu(fea_att_score)
            fea_att_score = fea_att_score.view(B, N, hidden_size).unsqueeze(1) # (B, 1, N, D)
            logits = -self.__batch_dist__(support_proto, query, fea_att_score) #(B, N*Q, N)
            
            
        else:   
            logits = -self.__batch_dist__(support_proto_raw, query) #(B, N*Q, N)
        #minn, _ = logits.min(-1)
        #logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)
        #contrastive_logits = self._contrastive_learning_(support_proto_raw, query)
        
        #print(logits.size())
        support_label = support_label.view(-1, N, K, self.max_length)
        pred = self.threshold_calculation(support_label, logits, N, Q)
        return logits, pred
        
       
