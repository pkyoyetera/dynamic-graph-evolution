#
# Patrick Kyoyetera, Vijay Gochinghar
# 2023
#

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train a recommender model with either NGCF or LightGCN architecture.")
    # parser.add_argument('--data_dir', type=str,
    #                     default='data/',
    #                     help='Input data path.')
    parser.add_argument('--arch', type=str, default='ngcf',
                        help='Architecture to use: ngcf or lightgcn')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Store model to path.')
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='Number of epoch.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='dimension of embeddings.')
    parser.add_argument('--layers', type=str, default=2,
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout for NGCF architecture.')
    parser.add_argument('--k', type=str, default=20,
                        help='k order of metric evaluation (e.g. Recall@k)')
    parser.add_argument('--eval_N', type=int, default=2,
                        help='Evaluate every N epochs')
    parser.add_argument('--save_results', type=int, default=1,
                        help='Save model and results')

    return parser.parse_args()
