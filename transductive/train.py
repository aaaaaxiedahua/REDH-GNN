import os
import argparse
import torch
import numpy as np
from load_data import DataLoader
from base_model import BaseModel
from utils import select_gpu

parser = argparse.ArgumentParser(description="Parser for RED-GNN")
parser.add_argument('--data_path', type=str, default='data/family/')
parser.add_argument('--seed', type=str, default=1234)
parser.add_argument('--model_name', type=str, default='redgnn', choices=['redgnn', 'sheaf_momentum'])
parser.add_argument('--topk_nodes', type=int, default=0)
parser.add_argument('--gamma', type=float, default=0.7)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--lambda_sheaf', type=float, default=0.0)
parser.add_argument('--lambda_dyn', type=float, default=0.0)
parser.add_argument('--fact_ratio', type=float, default=None)


args = parser.parse_args()

class Options(object):
    pass

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]
   
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    opts = Options
    opts.perf_file = os.path.join(results_dir,  dataset + '_perf.txt')
    opts.mem_file  = os.path.join(results_dir,  dataset + '_mem.txt')

    gpu = select_gpu()
    torch.cuda.set_device(gpu)
    print('gpu:', gpu)

    if dataset == 'family':
        opts.fact_ratio = 0.90
        opts.lr = 0.0036
        opts.decay_rate = 0.999
        opts.lamb = 0.000017
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.29
        opts.act = 'relu'
        opts.n_batch = 20
        opts.n_tbatch = 50
    elif dataset == 'umls':
        opts.fact_ratio = 0.90
        opts.lr = 0.0012
        opts.decay_rate = 0.9917
        opts.lamb = 0.000115
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 4
        opts.dropout = 0.0024
        opts.act = 'relu'
        opts.n_batch = 20
        opts.n_tbatch = 50
    elif dataset == 'WN18RR':
        opts.fact_ratio = 0.96
        opts.lr = 0.0021
        opts.decay_rate = 0.9962
        opts.lamb = 0.000037
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 5
        opts.dropout = 0.0067
        opts.act = 'tanh'
        opts.n_batch = 100
        opts.n_tbatch = 50
    elif dataset == 'fb15k-237':
        opts.fact_ratio = 0.99
        opts.lr = 0.0009
        opts.decay_rate = 0.9938
        opts.lamb = 0.000080
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 4
        opts.dropout = 0.0391
        opts.act = 'relu'
        opts.n_batch = 5
        opts.n_tbatch = 1
    elif dataset == 'nell':
        opts.fact_ratio = 0.95
        opts.lr = 0.0011
        opts.decay_rate = 0.9938
        opts.lamb = 0.000089
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 5
        opts.dropout = 0.2593
        opts.act = 'relu'
        opts.n_batch = 5
        opts.n_tbatch = 1
    elif dataset == 'YAGO':
        opts.fact_ratio = 0.995
        opts.lr = 0.0003
        opts.decay_rate = 0.997
        opts.lamb = 0.000111
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.2131
        opts.act = 'relu'
        opts.n_batch = 3
        opts.n_tbatch = 1

    if args.fact_ratio is not None:
        opts.fact_ratio = args.fact_ratio

    loader = DataLoader(args.data_path, fact_ratio=opts.fact_ratio)
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel
    opts.model_name = args.model_name
    opts.topk_nodes = args.topk_nodes
    opts.gamma = args.gamma
    opts.beta = args.beta
    opts.lambda_sheaf = args.lambda_sheaf
    opts.lambda_dyn = args.lambda_dyn

    config_str = '%.4f, %.4f, %.6f, %.2f, %d, %d, %d, %d, %.4f,%s,%s,%d,%.4f,%.4f,%.6f,%.6f\n' % (
        opts.lr, opts.decay_rate, opts.lamb, opts.fact_ratio, opts.hidden_dim, opts.attn_dim,
        opts.n_layer, opts.n_batch, opts.dropout, opts.act, opts.model_name,
        opts.topk_nodes, opts.gamma, opts.beta, opts.lambda_sheaf, opts.lambda_dyn
    )
    print(config_str)
    with open(opts.perf_file, 'a+') as f:
        f.write(config_str)

    model = BaseModel(opts, loader)

    best_mrr = 0
    for epoch in range(50):
        mrr, out_str = model.train_batch(epoch=epoch)
        with open(opts.perf_file, 'a+') as f:
            f.write(out_str)
        if mrr > best_mrr:
            best_mrr = mrr
            best_str = out_str
            print(str(epoch) + '\t' + best_str)
    print(best_str)
