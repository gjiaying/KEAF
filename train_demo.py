import os
from framework.data_loader import get_loader
from framework.data_loader import get_attribute_information
from framework.encoder import BERTJapaneseEncoder
from framework.anchor_encoder import BERTJapaneseEncoder2
from framework.generator import JapaneseGenerator
from framework.frameworks import MLFSFramework
from framework.model import MLFSModel
from framework.siamese import Siamese
from framework.faea import FAEA
from framework.hcrp import HCRP
from framework.mtb import Mtb
from framework.SimpleFS import SimpleFSRE
import torch
import pprint
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--train', default='training_file',
            help='train file')
    parser.add_argument('--val', default='testing_file',
            help='val file')
    parser.add_argument('--test', default='testing_file',
            help='test file')
    parser.add_argument('--attribute', default='masterTags.mave.tsv',
            help='attribute information')
    parser.add_argument('--model', default='mlfs',
            help='model name')
    parser.add_argument('--trainN', default=45, type=int,
            help='N in train')
    parser.add_argument('--N', default=17, type=int,
            help='N way')
    parser.add_argument('--K', default=1, type=int,
            help='K shot')
    parser.add_argument('--Q', default=1, type=int,
            help='Num of query per class')
    parser.add_argument('--batch_size', default=1, type=int,
            help='batch size')
    parser.add_argument('--gpu_num', default=1, type=int,
            help='gpu number')
    parser.add_argument('--train_iter', default=20000, type=int,
            help='num of iters in training')
    parser.add_argument('--val_iter', default=1000, type=int,
            help='num of iters in validation')
    parser.add_argument('--test_iter', default=1000, type=int,
            help='num of iters in testing')
    parser.add_argument('--val_step', default=2000, type=int,
           help='val after training how many iters')
    parser.add_argument('--max_length', default=256, type=int,
           help='text max length')
    parser.add_argument('--anchor_length', default=32, type=int,
           help='anchor max length')
    parser.add_argument('--anchor_weight_factor', default=0.1, type=float,
           help='anchor weight factor')
    parser.add_argument('--lr', default=1e-5, type=float,
           help='learning rate')
    parser.add_argument('--weight_decay', default=1e-6, type=float,
           help='weight decay')
    parser.add_argument('--dropout', default=0.2, type=float,
           help='dropout rate')
    parser.add_argument('--grad_iter', default=1, type=int,
           help='accumulate gradient every x iterations')
    parser.add_argument('--label_attention', default='No', choices=('ins', 'fea', 'all', 'No', 'cat', 'cross'),
           help='attention mechanisms')
    parser.add_argument('--attention', default='False', choices=('True', 'False'),
           help='taxonomy-aware attention')
    parser.add_argument('--generator', default='False', choices=('True', 'False'),
           help='using generator or not')
    parser.add_argument('--checkpoint', default='./checkpoint',
            help='folder to save models')
    parser.add_argument('--result', default='./test_result',
            help='folder to save testing result')
    parser.add_argument('--name', default='MLFS',
            help='model name')
    parser.add_argument('--warmup', default=5000, type=int, 
            help='warm up steps')
    parser.add_argument('--threshold', default='True', choices=('True', 'False'),
                        help='learnt threshold or not')
    parser.add_argument('--category', default='no', choices=('no', 'category', 'taxonomy'),
                        help='using category information or not')
    parser.add_argument('--output', help='output root folder')
    #parser.add_argument('--anchor_template', default=':', type=str,
          # help='anchor template')
    

    opt = parser.parse_args()
    model = opt.model
    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    max_length = opt.max_length
    anchor_length = opt.anchor_length
    weight_factor = opt.anchor_weight_factor
    data_root = opt.data_root
    training_file = opt.train
    validation_file = opt.val
    testing_file = opt.test
    testing_iter = opt.test_iter
    attribute_information = opt.attribute
    att = opt.attention
    label_att = opt.label_attention
    using_generator = opt.generator
    output = opt.output
    checkpoint = os.path.join(output, opt.checkpoint)
    model_name = opt.name
    warmup = opt.warmup
    test_result_dir = os.path.join(output, opt.result)
    learning_rate = opt.lr
    weight_decay = opt.weight_decay
    train_iter = opt.train_iter
    val_iter = opt.val_iter
    val_step = opt.val_step
    test_iter = opt.test_iter
    threshold = opt.threshold
    category = opt.category
    # checkpoint = opt.checkpoint
    # test_result_dir = opt.result
    #num_of_gpu = opt.gpu_num
    device_ids = list(range(opt.gpu_num))
    pprint.pprint(vars(opt))

    if not os.path.exists(output):
        os.mkdir(output)
    if not os.path.exists(checkpoint):
        os.mkdir(checkpoint)
    if not os.path.exists(test_result_dir):
        os.mkdir(test_result_dir)

    

    print("{}-way-{}-shot Multi-Label Few-Shot Attribute-Value Extraction".format(N, K))
    print("Start to use {} GPUs".format(opt.gpu_num))
    encoder = BERTJapaneseEncoder(max_length, att, category)
    anchor_encoder = BERTJapaneseEncoder2(max_length)
    generator = JapaneseGenerator()
    train_data_loader = get_loader(training_file, encoder, N=trainN, K=K, Q=Q, batch_size=batch_size, root=data_root)
    val_data_loader = get_loader(validation_file, encoder, N=N, K=K, Q=Q, batch_size=batch_size, root=data_root)
    test_data_loader = get_loader(testing_file, encoder, N=N, K=K, Q=Q, batch_size=batch_size, root=data_root)
    attribute_group_data, attribute_value_data = get_attribute_information(attribute_information, root=data_root)

    framework = MLFSFramework(train_data_loader, val_data_loader, test_data_loader, attribute_group_data, attribute_value_data, encoder, anchor_encoder, generator, using_generator)
    if model == 'siamese':
        model = Siamese(encoder, max_length, weight_factor, device_ids, att)
    elif model =='gnn':
        model = GNN(encoder, max_length, weight_factor, device_ids, trainN)
    elif model =='snail':
        model = Snail(encoder, max_length, weight_factor, device_ids, att)
    elif model == 'mtb':
        model = Mtb(encoder, max_length, weight_factor, device_ids, att)
    elif model == 'simplefs':
        model = SimpleFSRE(encoder, max_length, weight_factor, device_ids, att)
    elif model == 'faea':
        model = FAEA(encoder, max_length, weight_factor, device_ids, att)
    elif model == 'hcrp':
        model = HCRP(encoder, max_length, weight_factor, device_ids, att)
    else:
        model = MLFSModel(encoder, max_length, weight_factor, device_ids, att, label_att, K)
    framework.train(model, encoder, anchor_encoder, batch_size, trainN, N, K, Q, anchor_length, checkpoint,
                    test_result_dir, model_name, learning_rate, weight_decay, train_iter, val_iter, val_step, test_iter,
                    warmup, threshold)
    # macro_p, micro_p, macro_r, macro_r, macro_f1, micro_f1 = framework.eval(model, anchor_encoder, batch_size, N, K, Q, anchor_length, testing_iter)
    print("Finish.......")

if __name__ == "__main__":
    main()
