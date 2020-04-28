"""
Module for argument parcer
"""
import argparse

args = argparse.ArgumentParser(description='Transformer for Reviews')

# misc
args.add_argument('--seed', type=int, default=42,
                  help='random seed')
args.add_argument('--local', type=bool, default=False,
                  help='Boolean for printing log info if local')
args.add_argument('--results_dir', type=str,
                  help='Where to write results')
args.add_argument('--exp_name', type=str, default='no exp name',
                  help='name of experiment')

# data related information
args.add_argument('--data_dir', type=str, default='data',
                  help='directory storing all data')
args.add_argument('--data_name', type=str, default='reviews_UIC',
                  help='name of dataset.')
args.add_argument('--label_name', type=str, default='flagged',
                  help='name of label column header.')
args.add_argument('--content', type=str, default='reviewContent',
                  help='name of content column headers separated by `,`')
args.add_argument('--review_key', type=str, default='reviewContent',
                  help='name of review column header')
args.add_argument('--no_shuffle', action='store_true',
                  help='turns off shuffling of training data')
args.add_argument('--no_cache', action='store_true',
                  help='turns off caching of data')

args.add_argument('--batch_size', type=int, default=8,
                  help='batch size')
args.add_argument('--input_length', type=int, default=512,
                  help='longest length of a token')
args.add_argument('--val_split', type=float, default=0.1,
                  help='percent of data used for validation as decimal')
args.add_argument('--test_split', type=float, default=0.1,
                  help='percent of data used for test as decimal')
args.add_argument('--do_lower_case', action='store_true',
                  help='whether to do lower case for tokenizer')

# model related information
args.add_argument('--model', type=str, default='bert-base-uncased',
                  help='name of RLN network. default is BERT',
                  choices={'bert-base-uncased',
                           'bert-large-uncased',
                           'roberta-base',
                           'roberta-large',
                           'albert-base-v2',
                           'albert-xxlarge-v2'
                           'albert-base-v1',
                           'albert-xxlarge-v1'
                           })
args.add_argument('--n_labels', type=int, default=2,
                  help='number of label types')
args.add_argument('--n_others', type=int, default=0,
                  help='number of other statistics')
args.add_argument('--n_class_hidden', type=int, default=0,
                  help='number of hidden layers in in classifier')
args.add_argument('--preload_emb', action='store_true',
                  help='whether to preload embeddings')
args.add_argument('--preload_emb_name', type=str, default='',
                  help='name of preloaded embedding weights')

# training related information
args.add_argument('--save_dir', type=str, default='results',
                  help='directory to save results')
args.add_argument('--max_epochs', type=int, default=10,
                  help='number of epochs for fine-tuning')
args.add_argument('--lr', type=float, default=0.00003,
                  help='initial learning rate')
args.add_argument('--patience', type=int, default=5,
                  help='number of epochs without improvement before early stopping')
args.add_argument('--early_check', type=str, default='f1',
                  help='evaluation metric to check for early stopping')
args.add_argument('--no_early_stop', action='store_true',
                  help='boolean whether to use early stopping')
args.add_argument('--grad_accum', type=int, default=1,
                  help='number of gradient steps to accumulate for each step')
args.add_argument('--max_grad_norm', type=float, default=1.0,
                  help='maximum gradient norm for clipping')
args.add_argument('--pct_start', type=float, default=0.0,
                  help='percentage of the cycle spent increasing the learning rate')
args.add_argument('--weight_decay', type=float, default=0.0,
                  help='weight decay percentage')
args.add_argument('--cycle_momentum', action='store_true',
                  help='whether to cycle momentum for scheduler')
args.add_argument('--anneal_strategy', type=str, default='linear',
                  help='annealing strategy. default is linear.',
                  choices={'linear', 'cos'})
args.add_argument('--log_int', type=int, default=1,
                  help='number of epochs for logging information')
args.add_argument('--early_stop_criteria', type=str, default='loss',
                  help='early stopping criteria',
                  choices={'loss', 'acc', 'f1'})