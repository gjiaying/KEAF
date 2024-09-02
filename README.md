Knowledge-Enhanced Multi-Label Few-Shot Product Attribute-Value Extraction (CIKM 2023)



# KEAF
# mlfs_mave
A MLFS repository for MAVE related script
Training a Model:

- --train: training file, default = 'training_file'
- --val: validation file, default = 'testing_file'
- --test: testing file, default = 'testing_file'
- --attribute: attribute information file, default = 'masterTags.mave.tsv'
- --trainN: N in N-way K-shot. trainN is the specific N in training process. In multi-label settings, we set trainN to 45 for MAVE Dataset. default=45
- --N: N in N-way K-shot. In multi-label settings, we set N to 29 for Rakuten Dataset. default = 17
- --K: K in N-way K-shot. default = 5
- --Q: Sample Q query instances for each class. default = 1
- --batch_size: batch size, default = 1
- --gpu_num: number of gpu used for training the model. default = 1
- --train_iter: number of training iterations, default = 30000
- --val_iter: number of validating iterations, default = 1000
- --test_iter: number of testing iterations, default = 10000
- --val_step: validating after training how many iterations, default = 1000
- --max_length: text maximum length, default = 256
- --anchor_length: label description length, default = 32
- --anchor_weight_factor: weight factor of label description when construction prototypes, default = 0.5 (If no label description is needed, anchor_weight_factor can be set to zero). Range from 0.0 to 1.0
- --lr: learning rate, default = 1e-1
- --weight_decay: weight decay, default = 1e-5
- --dropout: droupout rate, default = 0.2
- --grad_iter: accumulate gradient every x iterations, default = 1
- --attention: taxonomy-aware attention, choices from ['True', 'False'], default = 'False'
- --category: category information

Example to run a multi-label 5-shot model without attention if want to use other default values: python train_demo.py --K 5 --Q 5 --max_length 256
