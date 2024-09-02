import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
import pandas as pd
from framework.tags_store_int import TagStore, TagMaster

class RakutenDataset(data.Dataset):
    """
    Rakuten Dataset
    """
    def __init__(self, name, encoder, N, K, Q, root):
        '''
        name: data file name
        encoder: japanese bert encoder
        N: ways
        K: shots
        Q: queries
        root: data file path
        '''
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.encoder = encoder

    def __getraw__(self, item):
        text, mask, labels, taxonomy, mask_t = self.encoder.tokenize(item['title & description'], item['labels'], item['category_text'], item['taxonomy_text'])
        return text, mask, labels, taxonomy, mask_t

    def __additem__(self, d, text, mask, label, taxonomy, mask_t):
        d['text'].append(text)
        d['mask'].append(mask)
        d['label'].append(label)
        d['taxonomy'].append(taxonomy)
        d['mask_t'].append(mask_t)

    def __getitem__(self, index):
        #target_classes = random.sample(self.classes, self.N)
        target_classes = self.classes # N is all labels
        support_set = {'text': [], 'mask': [], 'label': [], 'taxonomy':[], 'mask_t':[]}
        query_set = {'text': [], 'mask': [], 'label': [], 'taxonomy':[], 'mask_t':[]}
        query_label = []
        count_dict = dict.fromkeys(target_classes, 0)

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False) # get random indexes of instances for each class
            
            count = 0
            for j in indices:
                text, mask, labels, taxonomy, mask_t = self.__getraw__(
                        self.json_data[class_name][j])
                text = torch.tensor(text).long()
                mask = torch.tensor(mask).long()
                labels = torch.tensor(labels).long()
                #category = torch.tensor(category).long()
                taxonomy = torch.tensor(taxonomy).long()
                mask_t = torch.tensor(mask_t).long()

                label = self.json_data[class_name][j]['labelID']
                
                if count < self.Q:
                    self.__additem__(query_set, text, mask, labels, taxonomy, mask_t)
                    #query_label.append(label)
                else:
                    for k in range(len(label)):
                        if count_dict[label[k]] < self.K:
                            self.__additem__(support_set, text, mask, labels, taxonomy, mask_t)
                            count_dict[label[k]] = count_dict[label[k]] + 1
                        else:
                            continue
                count += 1
      
        
        return support_set, query_set, self.classes
    
    def __len__(self):
        return 1000000000

def collate_fn(data):
    batch_support = {'text': [], 'mask': [], 'label': [], 'taxonomy':[], 'mask_t':[]}
    batch_query = {'text': [], 'mask': [], 'label': [], 'taxonomy':[], 'mask_t':[]}
    #batch_label = []
    support_sets, query_sets, classes = zip(*data)
   
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        #batch_label += query_labels[i]
      
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)

    
    #batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_query['label'], classes

def get_loader(name, encoder, N, K, Q, batch_size, num_workers=8, collate_fn=collate_fn, root='./data'):
    dataset = RakutenDataset(name, encoder, N, K, Q, root)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)

def get_attribute_information(data, root='./data'):
    tag_master_file = os.path.join(root, data)
    attribute_data = pd.read_csv(tag_master_file, sep='\t')
    attribute_group_data = attribute_data.set_index('タググループID').T.to_dict('list') #タググループ名
    attribute_value_data = attribute_data.set_index('タグID').T.to_dict('list') #タグ名  
    return attribute_group_data, attribute_value_data

'''
def get_category_to_attribute(genre_master_file, tag_master_file, category_data):
    category_data = os.getcwd()+'/data/' + category_data
    tag_master_file = os.getcwd()+'/data/' + tag_master_file
    genre_master_file = os.getcwd()+'/data/' + genre_master_file
    g2tg, tg2g = TagStore.get_tag_genre_dicts(tag_master_file=tag_master_file,genre_master_file=genre_master_file,genre_tag_file=category_data)
    return g2tg
'''    