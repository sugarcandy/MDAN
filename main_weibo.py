
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='MDAN')
parser.add_argument('--dataset', default='weibo')  # weibo21 %% weibo
# parser.add_argument('--model_name', default='domain_ple6')
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--max_len', type=int, default=197)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--early_stop', type=int, default=3)
parser.add_argument('--bert_vocab_file', default='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt')
# parser.add_argument('--root_path', default='./Weibo_21/')
parser.add_argument('--root_path', default='./data/')
parser.add_argument('--bert', default='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--seed', type=int, default=3074)
parser.add_argument('--gpu', default='2')
parser.add_argument('--bert_emb_dim', type=int, default=768)
parser.add_argument('--w2v_emb_dim', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.0001)
# parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--emb_type', default='bert')
parser.add_argument('--w2v_vocab_file', default='./pretrained_model/w2v/Tencent_AILab_Chinese_w2v_model.kv')
parser.add_argument('--save_param_dir', default='./param_model_weibo')
parser.add_argument('--num_experts', default=6)
parser.add_argument('--num_heads', default=4)
parser.add_argument('--beta', default=0.1)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from run import Run
import torch
import numpy as np
import random

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = True

if args.emb_type == 'bert':
    emb_dim = args.bert_emb_dim
    vocab_file = args.bert_vocab_file
elif args.emb_type == 'w2v':
    emb_dim = args.w2v_emb_dim
    vocab_file = args.w2v_vocab_file

print('lr: {}; model name: {}; emb_type: {}; batchsize: {}; epoch: {}; gpu: {}; emb_dim: {}'.format(args.lr,
                                                                                                    args.model_name,
                                                                                                    args.emb_type,
                                                                                                    args.batchsize,
                                                                                                    args.epoch,
                                                                                                    args.gpu, emb_dim))

config = {
    'use_cuda': True,
    'batchsize': args.batchsize,
    'max_len': args.max_len,
    'early_stop': args.early_stop,
    'num_workers': args.num_workers,
    'vocab_file': vocab_file,
    'emb_type': args.emb_type,
    'bert': args.bert,
    'root_path': args.root_path,
    'weight_decay': 5e-5,
    'model':
        {
            'mlp': {'dims': [384], 'dropout': 0.2}
        },
    'emb_dim': emb_dim,
    'lr': args.lr,
    'epoch': args.epoch,
    'model_name': args.model_name,
    'seed': args.seed,
    'save_param_dir': args.save_param_dir,
    'dataset': args.dataset,
    'num_experts': args.num_experts,
    'num_heads': args.num_heads,
    'beta': args.beta
}

if __name__ == '__main__':
    Run(config=config).main()
