import sys
sys.path.append('..')
import framework
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class SimpleFSRE(framework.frameworks.MLFSModel):
    
    def __init__(self, encoder, max_length, weight_factor, device_ids, att):
        framework.frameworks.MLFSModel.__init__(self, encoder)
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        self.use_dropout = True
        self.dot = False
        
        
        self.relation_encoder = None
        self.hidden_size = 768
        
        self.max_length = max_length
        self.weight_factor = weight_factor
        self.encoder = encoder
        self.encoders = nn.DataParallel(self.encoder, device_ids=device_ids)
        self.att = att
    
    
    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

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
        
        
        ##get relation
        #if self.relation_encoder:
        #    rel_gol, rel_loc = self.relation_encoder(rel_txt)
        #else:
        rel_gol, rel_loc = self.encoders(anchor, 'SimpleFSRE')
        
        #import pdb
        #pdb.set_trace()
        
        rel_loc = torch.mean(rel_loc, 1) #[B*N, D]
        #rel_rep = (rel_loc + rel_gol) /2
        #rel_rep = rel_loc
       # rel_rep = torch.cat((rel_gol, rel_loc), -1)
        rel_gol = rel_gol * 0.5
        rel_loc = rel_loc * 0.5
        rel_rep = rel_gol.add(rel_loc)
    
        
        
        
        #import pdb
        #pdb.set_trace()
        
        
        #rel_final = torch.cat((rel_gol, rel_loc), -1)
        #import pdb
        #pdb.set_trace()
        #TODO
        
        #support,  s_loc = self.sentence_encoder(support) # (B * N * K, D), where D is the hidden size
        #query,  q_loc = self.sentence_encoder(query) # (B * total_Q, D)
        
        
        support = self.encoders(support) # (B * N * K, D), where D is the hidden size
        query = self.encoders(query) # (B * total_Q, D)
        
        #support = torch.cat((support_h, support_t), -1)
        #query = torch.cat((query_h, query_t), -1)
        
        
        
        support = support.view(-1, N, K, self.hidden_size) # (B, N, K, D)
        query = query.view(-1, Q, self.hidden_size) # (B, total_Q, D)
        
       

        # Prototypical Networks 
        # Ignore NA policy
        support = torch.mean(support, 2) # Calculate prototype for each class
        ##
       
        ###add relation into this this add a up relation dimension
        rel_rep = rel_rep.view(-1, N, rel_gol.shape[1])
  
        #rel_rep = self.linear(rel_rep)
        support = support + rel_rep
        #support = torch.permute(support, (1, 0, 2))
        query = query.permute((1,0,2))
        

        
        logits = self.__batch_dist__(support, query) # (B, total_Q, N)
      #  minn, _ = logits.min(-1)
      #  logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2) # (B, total_Q, N + 1)
        
       # print(logits.size())
        support_label = support_label.view(-1, N, K, self.max_length)
        pred = self.threshold_calculation(support_label, logits, N, Q)
       # _, pred = torch.max(logits.view(-1, N + 1), 1)
        return logits, pred

    
    
    
