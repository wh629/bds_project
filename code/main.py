"""
Main run script to execute experiments and analysis

NOT TESTED
"""

import torch

import transformers
import os
import sys
import logging as log

# =============== Self Defined ===============
import myio          # module for handling import/export of data
import learner       # training object
import model         # module to define model architecture
import args          # module to store arguments

def main():
    label_order = {'rating':0,
                   'flagged':1}

    # parse arguments
    parser = args.args

    # Set devise to CPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device is {}".format(device))
    
    # set data directory
    wd = os.getcwd()
    if os.path.isdir(parser.data_dir):
        data_path = parser.data_dir
    else:
        data_path = os.path.join(wd, parser.data_dir)
            
    # import data    
    task_names = [parser.data_name]
    tokenizer = transformers.AutoTokenizer.from_pretrained(parser.model)
    label_names = parser.label_names.split(',')
    data_handler = myio.IO(data_dir    = data_path,
                           task_names  = task_names,
                           tokenizer   = tokenizer,
                           max_length  = parser.input_length,
                           batch_size  = parser.batch_size,
                           label_names = label_names
                           )
    
    data_handler.read_task()

    # define model
    number_labels = [int(n) for n in parser.label_numbers.split(',')]
    config = transformers.AutoConfig.from_pretrained(parser.model)
    classifier = model.Model(config=config, 
                             nrating = number_labels[label_order.get('rating')],
                             nflag = number_labels[label_order.get('flagged')]
                             )
    
    # define trainer
    weights = [float(w) for w in parser.label_weights.split(',')]
    
    train_data = data_handler.tasks.get(parser.data_name).get('train')
    val_data = data_handler.tasks.get(parser.data_name).get('dev')
    test_data = data_handler.tasks.get(parser.data_name).get('test')
    
    if os.path.isdir(parser.save_dir):
        save_path = parser.save_dir
    else:
        save_path = os.path.join(wd, parser.save_dir)
            
    trainer = learner.Learner(model=classifier,
                              device=device,
                              train_data=train_data,
                              val_data=val_data,
                              test_data=test_data,
                              rating_w = weights[label_order.get('rating')],
                              flag_w = weights[label_order.get('flagged')],
                              max_epochs = parser.max_epochs,
                              save_path = save_path,
                              lr = parser.lr,
                              buffer_break= (parser.early_stop == 'True'),
                              break_int = parser.patience
                              )
            
    # train model
    best_path = trainer.learn(model_name = parser.model,
                              verbose = True,
                              early_check = parser.early_stop
                              )
            
    sys.exit(0)