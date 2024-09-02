import os
import sklearn.metrics
import numpy as np
import sys
import time
from . import encoder
from . import data_loader
from . import anchor_encoder
from framework.data_loader import get_attribute_information
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from torchmetrics import Precision, Recall, F1Score

class MLFSModel(nn.Module):
    def __init__(self, encoder):
        '''
        encoder: Tohoku encoder
        
        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.encoder = encoder
        #self.anchor_encoder = anchor_encoder
        self.cost = nn.BCEWithLogitsLoss()
        self.cost1 = nn.CrossEntropyLoss(reduction='none')
        self.gamma = 1
        self.reduction = 'mean'
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        labels = torch.narrow(label, 1, 0, N)
        logit = logits.view(-1, N)
        return self.cost(logit, labels)
    
    def l2norm(self, X):
        norm = torch.pow(X, 2).sum(dim=-1, keepdim=True).sqrt()
        X = torch.div(X, norm)
        return X
    
    def loss_1(self, logits, label, weight=None):
        """
        logits: Logits with the size (..., class_num)
        label: Label with whatever size.
        return: [Loss] (A single value)
        """
        N = logits.size(-1)

        # focal weights
        logits_ = torch.softmax(self.l2norm(logits), dim=-1)
        logits_ = logits_.view(-1, N)
        #logits, label = logits.view(-1, N), label.view(-1)
        logits = logits.view(-1, N)
        
        #probs = torch.stack([logits_[i, t] for i, t in enumerate(label)])
        focal_weight = torch.pow(1 - logits, self.gamma)

        # task adaptive weights
        if weight is not None:
            focal_weight = focal_weight * weight.view(-1)

        # add weights to cross entropy
        #label = label.type(torch.LongTensor).cuda()
        ce_loss = self.cost1(logits, label)  # (B*totalQ)
        tf_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            tf_loss = (tf_loss / focal_weight.sum()).sum()
        elif self.reduction == 'sum':
            tf_loss = tf_loss.sum()

        return tf_loss

    def evaluation(self, pred, label, B, N, Q):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        label = label.view(B*N*Q, N) #(B, NQ, N)
        pred = pred.view(B*N*Q, N)
        label = label.type(torch.LongTensor).to('cuda')
        precision_macro = Precision(num_classes=N, multiclass=False, average='macro').to('cuda')
        precision_micro = Precision(num_classes=N, multiclass=False, average='micro').to('cuda')
        recall_macro = Recall(num_classes=N, multiclass=False, average='macro').to('cuda')
        recall_micro = Recall(num_classes=N, multiclass=False, average='micro').to('cuda')
        f1_macro = F1Score(num_classes=N, multiclass=False, average='macro').to('cuda')
        f1_micro = F1Score(num_classes=N, multiclass=False, average='micro').to('cuda')
        precision_macro_score = precision_macro(pred, label)
        precision_micro_score = precision_micro(pred, label)
        recall_macro_score = recall_macro(pred, label)
        recall_micro_score = recall_micro(pred, label)
        f1_macro_score = f1_macro(pred, label)
        f1_micro_score = f1_micro(pred, label)
        return precision_macro_score, precision_micro_score, recall_macro_score, recall_micro_score, f1_macro_score, f1_micro_score


class MLFSFramework:
    def __init__(self, train_data_loader, val_data_loader, test_data_loader, attribute_group_data, attribute_value_data, encoder, anchor_encoder, generator, using_generator):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.attribute_group_data = attribute_group_data
        self.attribute_value_data = attribute_value_data
        self.encoder = encoder
        self.anchor_encoder = anchor_encoder
        self.generator = generator
        self.using_generator = using_generator
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.lamda = 1

    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        if int(torch.__version__.split('.')[1]) < 4:
            return x[0]
        else:
            return x.item()
    
    def get_prompts(self, classes, anchor_length):
        prompt_set = {'text': np.zeros((len(classes), anchor_length), dtype=int), 'mask': np.zeros((len(classes), anchor_length), dtype=int)}
        for i in range(len(classes)):
            attribute_group_text = self.attribute_group_data.get(int(classes[i].split(':')[0]))[0]
            attribute_value_text = self.attribute_value_data.get(int(classes[i].split(':')[1]))[2]
            prompt = attribute_group_text + 'is' + attribute_value_text
            if self.using_generator == 'True':
                generated_text = prompt + '. ' + self.generator(prompt)
            else:
                generated_text = prompt + '. '
            text, mask = self.encoder.anchor_tokenize(generated_text, anchor_length)
            prompt_set['text'][i] = text
            prompt_set['mask'][i] = mask
        prompt_set['text'] = torch.IntTensor(prompt_set['text'])
        prompt_set['mask'] = torch.IntTensor(prompt_set['mask'])
        
        return prompt_set

    def get_threshold(self, logits, label, B, N, Q):
        logits = logits.view(B*N*Q, N)
        label = label.view(B*N*Q, N)
        value = torch.mul(logits, label)
        value_new = value.clone()
        value_new[value == 0] = -999
        max_value,_ = torch.max(value_new, 1)
        result = torch.mean(max_value)
        return result

    def train(self,
              model,
              encoder,
              anchor_encoder,
              B, N_for_train, N_for_eval, K, Q,
              anchor_length,
              ckpt_dir,
              test_result_dir,
              model_name,
              learning_rate,
              weight_decay,
              train_iter,
              val_iter,#1000
              val_step,#2000
              test_iter,
              warmup_step,
              threshold_choice,
              pretrain_model=None,
              cuda=True,
              warmup=True,
              grad_iter=1,
              optimizer=optim.SGD,
              enable_amp=True):
        '''
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        test_result_dir: Directory of test results
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        cuda: Use CUDA or not
        pretrain_model: Pre-trained checkpoint path
        '''
        print("Start training...")
        #torch.autograd.set_detect_anomaly(True)
        best_val = []
        
        #parameters_to_optimize = list(model.named_parameters())
        #no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        #parameters_to_optimize = [
        #    {'params': [p for n, p in parameters_to_optimize 
        #        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #    {'params': [p for n, p in parameters_to_optimize
        #        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        #]    
        #parameters_to_optimize = filter(lambda x:x.requires_grad, model.parameters())
        parameters_to_optimize = filter(lambda x:x.requires_grad, model.parameters())
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, correct_bias=False)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 

        if pretrain_model:
            checkpoint = self.__load_model__(pretrain_model)
            model.load_state_dict(checkpoint['state_dict'])
            start_iter = checkpoint['iter'] + 1
        else:
            start_iter = 0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        encoder.to(device)
        anchor_encoder.to(device)
        model.train()

        best_acc = 0
        iter_loss = 0.0
        iter_p_macro = 0.0
        iter_p_micro = 0.0
        iter_r_macro = 0.0
        iter_r_micro = 0.0
        iter_f_macro = 0.0
        iter_f_micro = 0.0
        iter_sample = 0.0
        thresholds = []

        _,_,_,classes= next(self.train_data_loader)
        anchor = self.get_prompts(classes[0], anchor_length)

        if torch.cuda.is_available():
            for an in anchor:
                anchor[an] = anchor[an].to(device)
        #with torch.cuda.amp.autocast(enabled=enable_amp):
            #anchors = anchor  
        anchors = anchor

        for it in range(start_iter, start_iter + train_iter):
            support, query, label, classes = next(self.train_data_loader)
            label = torch.narrow(label, 1, 0, N_for_train)
            label = label.type(torch.FloatTensor)
            #print(label.size()) #N*Q, N
            if torch.cuda.is_available():
                for k in support:
                    support[k] = support[k].to(device)
                for k in query:
                    query[k] = query[k].to(device)
                label = label.to(device)
            # with torch.autocast(device_type='cuda', dtype=torch.float16): 
                with torch.cuda.amp.autocast(enabled=enable_amp):
                    if (model.__class__.__name__ == 'HCRP' or model.__class__.__name__ == 'FAEA'):
                        logits, pred, logits_proto, labels_proto, sim_task = model(support, query, N_for_train, K, Q, anchors, support['label'])
                        pred = pred.to(device)
                        logits = logits.to(device)
                        threshold = self.get_threshold(logits, label, B, N_for_train, Q)
                        loss = model.loss(logits, label) + self.lamda * self.bce_loss(logits_proto, labels_proto)
                    else:
                        logits, pred = model(support, query, N_for_train, K, Q, anchors, support['label'])
                        pred = pred.to(device)
                        logits = logits.to(device)
                        threshold = self.get_threshold(logits, label, B, N_for_train, Q)
                
                        loss = model.loss(logits, label)

            precision_macro_score, precision_micro_score, recall_macro_score, recall_micro_score, f1_macro_score, f1_micro_score = model.evaluation(pred, label, B, N_for_train, Q)
            loss.backward(retain_graph=True)

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            if it % grad_iter == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            iter_loss += self.item(loss.data)
            iter_p_macro += self.item(precision_macro_score.data)
            iter_p_micro += self.item(precision_micro_score.data)
            iter_r_macro += self.item(recall_macro_score.data)
            iter_r_micro += self.item(recall_micro_score.data)
            iter_f_macro += self.item(f1_macro_score.data)
            iter_f_micro += self.item(f1_micro_score.data)
            iter_sample += 1
            sys.stdout.write('step: {0:4} | loss: {1:2.6f}, p_macro: {2:3.2f}%, p_micro: {3:3.2f}%, r_macro: {4:3.2f}%, r_micro: {5:3.2f}%, f_macro: {6:3.2f}%, f_micro: {7:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_p_macro / iter_sample, 100 * iter_p_micro / iter_sample, 100 * iter_r_macro / iter_sample, 100 * iter_r_micro / iter_sample, 100 * iter_f_macro / iter_sample, 100 * iter_f_micro / iter_sample) +'\r')
            sys.stdout.flush()

            if (it + 1) % 500 == 0:
                print(iter_loss / iter_sample)
                print(threshold)

            if (it + 1) % val_step == 0:
                precision_macro_score, precision_micro_score, recall_macro_score, recall_micro_score, f1_macro_score, f1_micro_score = self.eval(model, anchor_encoder, B, N_for_eval, K, Q, anchor_length, threshold, val_iter, threshold_choice)
                model.train()
                if f1_micro_score > best_acc:
                    print('Best checkpoint')
                    best_val = [precision_macro_score, precision_micro_score, recall_macro_score, recall_micro_score, f1_macro_score, f1_micro_score]
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    save_path = os.path.join(ckpt_dir, model_name + ".pth.tar")
                    torch.save({'state_dict': model.state_dict()}, save_path)
                    best_acc = f1_micro_score
                iter_loss = 0.
                iter_p_macro = 0.0
                iter_p_micro = 0.0
                iter_r_macro = 0.0
                iter_r_micro = 0.0
                iter_f_macro = 0.0
                iter_f_micro = 0.0
                iter_sample = 0.
                
        print("\n####################\n")
        print("Finish training " + model_name)
        macro_p, micro_p, macro_r, micro_r, macro_f1, micro_f1 = self.eval(model, anchor_encoder, B, N_for_eval, K, Q, anchor_length, threshold, test_iter, threshold_choice, ckpt=os.path.join(ckpt_dir, model_name + '.pth.tar'))
        print(best_val)
        print("Test macro_p: {0:2.2f}%".format(macro_p*100))
        print("Test micro_p: {0:2.2f}%".format(micro_p*100))
        print("Test macro_r: {0:2.2f}%".format(macro_r*100))
        print("Test micro_r: {0:2.2f}%".format(micro_r*100))
        print("Test macro_f1: {0:2.2f}%".format(macro_f1*100))
        print("Test micro_f1: {0:2.2f}%".format(micro_f1*100))

    def get_prediction(self, logits, threshold):
        new_logits = logits.clone()
        new_logits[logits >= threshold] = 1.
        new_logits[logits < threshold] = 0.
        return new_logits


    def eval(self,
        model,
        anchor_encoder,
        B, N, K, Q,
        anchor_length,
        threshold,
        eval_iter,
        threshold_choice,
        ckpt=None,
        enable_amp=True):
        
        
        model.eval()
        if ckpt is None:
            eval_dataset = self.val_data_loader
        else:
            checkpoint = self.__load_model__(ckpt)
            model.load_state_dict(checkpoint['state_dict'])
            eval_dataset = self.test_data_loader


        iter_p_macro = 0.0
        iter_p_micro = 0.0
        iter_r_macro = 0.0
        iter_r_micro = 0.0
        iter_f_macro = 0.0
        iter_f_micro = 0.0
        iter_sample = 0.0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        anchor_encoder.to(device)

        _,_,_,classes = next(eval_dataset)
        anchor = self.get_prompts(classes[0], anchor_length)
        if torch.cuda.is_available():
            for an in anchor:
                anchor[an] = anchor[an].to(device)
        #with torch.cuda.amp.autocast(enabled=enable_amp):
         #   anchors = anchor_encoder(anchor)
        anchors = anchor   
        with torch.no_grad():
            for it in range(eval_iter):
                support, query, label, classes = next(eval_dataset)
                label = torch.narrow(label, 1, 0, N)
                label = label.type(torch.FloatTensor)
                if torch.cuda.is_available():
                    for k in support:
                        support[k] = support[k].to(device)
                    for k in query:
                        query[k] = query[k].to(device)
                    label = label.to(device)
                with torch.cuda.amp.autocast(enabled=enable_amp):
                    if (model.__class__.__name__ == 'HCRP' or model.__class__.__name__ == 'FAEA'):
                        logits, pred, logits_proto, labels_proto, sim_task = model(support, query, N, K, Q, anchors, support['label'])
                        if threshold_choice:
                            pred_t = self.get_prediction(logits, threshold)
                            pred = pred_t.to(device)
                        else:
                            pred = preds.to(device)
                        logits = logits.to(device)
                    else:
                        logits, preds = model(support, query, N, K, Q, anchors, support['label'])
                        if threshold_choice:
                            pred_t = self.get_prediction(logits, threshold)
                            pred = pred_t.to(device)
                        else:
                            pred = preds.to(device)
                        logits = logits.to(device)
                #logits, pred = model(support, query, N, K, Q * N + Q * na_rate)
                    
                precision_macro_score, precision_micro_score, recall_macro_score, recall_micro_score, f1_macro_score, f1_micro_score = model.evaluation(pred, label, B, N, Q)
                iter_p_macro += self.item(precision_macro_score.data)
                iter_p_micro += self.item(precision_micro_score.data)
                iter_r_macro += self.item(recall_macro_score.data)
                iter_r_micro += self.item(recall_micro_score.data)
                iter_f_macro += self.item(f1_macro_score.data)
                iter_f_micro += self.item(f1_micro_score.data)
                iter_sample += 1

                #sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) + '\r')
                sys.stdout.write('[EVAL] step: {0:4} | p_macro: {1:3.2f}%, p_micro: {2:3.2f}%, r_macro: {3:3.2f}%, r_micro: {4:3.2f}%, f_macro: {5:3.2f}%, f_micro: {6:3.2f}%'.format(it + 1, 100 * iter_p_macro / iter_sample, 100 * iter_p_micro / iter_sample, 100 * iter_r_macro / iter_sample, 100 * iter_r_micro / iter_sample, 100 * iter_f_macro / iter_sample, 100 * iter_f_micro / iter_sample) +'\r')

                sys.stdout.flush()
            print("")
        return iter_p_macro / iter_sample, iter_p_micro / iter_sample, iter_r_macro / iter_sample, iter_r_micro / iter_sample, iter_f_macro / iter_sample, iter_f_micro / iter_sample
