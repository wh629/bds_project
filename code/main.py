"""
Main run script to execute experiments and analysis

NOT TESTED
"""

import torch
import transformers
import os
import sys
import logging as log
import random
import numpy as np
from datetime import datetime as dt

# =============== Self Defined ===============
import myio          # module for handling import/export of data
import learner       # training object
import model         # module to define model architecture
import args          # module to store arguments

def main():
    
    # parse arguments
    parser = args.args
    
    # get working directory
    wd = os.getcwd()
    
    # set up logger
    log_fname = os.path.join(wd, "logs", "log_{}.log".format(
    dt.now().strftime("%Y%m%d_%H%M")))
    
    root = log.getLogger()
    while len(root.handlers):
        root.removeHandler(root.handlers[0])
    log.basicConfig(filename=log_fname,
            format='%(asctime)s: %(name)s || %(message)s',
            level=log.INFO)
    root.addHandler(log.StreamHandler())
    
# =============================================================================
#     start
# =============================================================================
    log.info("="*40 + " Start Program " + "="*40)
    
# =============================================================================
#     misc stuff    
# =============================================================================
    label_order = {'rating':0,
                   'flagged':1}
    
    # Set devise to CPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device is {}".format(device))
    
    # set random seeds
    random.seed(parser.seed)
    np.random.seed(parser.seed)
    torch.manual_seed(parser.seed)
    if device == "cuda":
        torch.cuda.manual_seed(parser.seed)
        torch.cuda.manual_seed_all(parser.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    # set data directory
    if os.path.isdir(parser.data_dir):
        data_path = parser.data_dir
    else:
        data_path = os.path.join(wd, parser.data_dir)
    
    log.info("Data Directory is {}.".format(data_path))
    
# =============================================================================
#     import data
# =============================================================================
    task_names = [parser.data_name]
    tokenizer = transformers.AutoTokenizer.from_pretrained(parser.model)
    label_names = parser.label_names.split(',')
    data_handler = myio.IO(data_dir    = data_path,
                           task_names  = task_names,
                           tokenizer   = tokenizer,
                           max_length  = parser.input_length,
                           val_split   = parser.val_split,
                           test_split  = parser.test_split,
                           batch_size  = parser.batch_size,
                           label_names = label_names
                           )
    
    data_handler.read_task()

# =============================================================================
#     define model
# =============================================================================
    log.info("="*40 + " Defining Model " + "="*40)
    number_labels = [int(n) for n in parser.label_numbers.split(',')]
    config = transformers.AutoConfig.from_pretrained(parser.model)
    classifier = model.Model(config=config, 
                             nrating = number_labels[label_order.get('rating')],
                             nflag = number_labels[label_order.get('flagged')]
                             )
    
# =============================================================================
#     define trainer
# =============================================================================
    weights = [float(w) for w in parser.label_weights.split(',')]
    
    train_data = data_handler.tasks.get(parser.data_name).get('train')
    val_data = data_handler.tasks.get(parser.data_name).get('dev')
    test_data = data_handler.tasks.get(parser.data_name).get('test')
    
    if os.path.isdir(parser.save_dir):
        save_path = parser.save_dir
    else:
        save_path = os.path.join(wd, parser.save_dir)
    
    log.info("Save Directory is {}.".format(save_path))
    log.info("="*40 + " Defining Trainer " + "="*40)
    
    # create trainer object
    trainer = learner.Learner(model        = classifier,
                              device       = device,
                              train_data   = train_data,
                              val_data     = val_data,
                              test_data    = test_data,
                              rating_w     = weights[label_order.get('rating')],
                              flag_w       = weights[label_order.get('flagged')],
                              max_epochs   = parser.max_epochs,
                              save_path    = save_path,
                              lr           = parser.lr,
                              buffer_break = (parser.early_stop == 'True'),
                              break_int    = parser.patience
                              )
            
    # train model
    best_path, best_emb_path = trainer.learn(model_name  = parser.model,
                                             verbose     = True,
                                             early_check = parser.early_stop
                                             )
    
    log.info("="*40 + " Program Complete " + "="*40)
    log.info("Best Total Weights in {}".format(best_path))
    log.info("Best Embedding Weights in {}".format(best_emb_path))
    
    # release logs
    for handler in log.getLogger().handlers:
        handler.close()
    
    # exit python
    sys.exit(0)

if __name__ == "__main__":
    main()