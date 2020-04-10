"""
Module for argument parcer
"""
import argparse

args = argparse.ArgumentParser(description='Transformer for Reviews')

# misc
args.add_argument('--seed', type=int, default=1111,
                  help='random seed')
args.add_argument('--local', type=bool, default=False,
                  help='Boolean for printing log info if local')

# data related information
args.add_argument('--data_dir', type=str, default='data',
                  help='directory storing all data')
args.add_argument('--data_name', type=str, default='reviews_UIC',
                  help='name of dataset.')
args.add_argument('--label_names', type=str, default='rating,flagged',
                  help='name of label column headers separated by `,`.')
args.add_argument('--batch_size', type=int, default=32,
                  help='batch size')
args.add_argument('--input_length', type=int, default=512,
                  help='longest length of a token')
args.add_argument('--val_split', type=float, default=0.1,
                  help='percent of data used for validation as decimal')
args.add_argument('--test_split', type=float, default=0.1,
                  help='percent of data used for test as decimal')

# model related information
args.add_argument('--model', type=str, default='bert-base-uncased',
                  help='name of RLN network. default is BERT',
                  choices={'bert-base-uncased',
                           'bert-base-cased',
                           'albert-base-v2',
                           'albert-xxlarge-v2'
                           })
args.add_argument('--label_numbers', type=str, default='5,2',
                  help='number of arguments per label separated by `,`.')

# training related information
args.add_argument('--save_dir', type=str, default='results',
                  help='directory to save results')
args.add_argument('--max_epochs', type=int, default=100000,
                  help='number of epochs for fine-tuning')
args.add_argument('--lr', type=float, default=0.001,
                  help='initial learning rate')
args.add_argument('--label_weights', type=str, default='1,1',
                  help='weights of labels separated by `,`.')
args.add_argument('--patience', type=int, default=10,
                  help='number of epochs without improvement before early stopping')
args.add_argument('--early_check', type=str, default='loss',
                  help='evaluation metric to check for early stopping')
args.add_argument('--early_stop', type=str, default='True',
                  help='boolean whether to use early stopping',
                  choices={'True','False'})