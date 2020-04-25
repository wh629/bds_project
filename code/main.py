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
import json

# =============== Self Defined ===============
import myio            # module for handling import/export of data
import learner         # training object
import model           # module to define model architecture
from args import args  # module to store arguments

def main():
    
    # parse arguments
    parser = args.parse_args()
    
    # get working directory
    wd = os.getcwd()
    
    # set up logger
    log_fname = os.path.join(wd, "logs", "log_{}.log".format(
    dt.now().strftime("%Y%m%d_%H%M")))
    
    log.basicConfig(filename=log_fname,
            format='%(asctime)s: %(name)s || %(message)s',
            level=log.INFO)
    
# =============================================================================
#     start
# =============================================================================
    log.info("="*40 + " Start Program " + "="*40)
    
# =============================================================================
#     misc stuff    
# =============================================================================
    
    # Set devise to CPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info("Device is {}".format(device))
    
    # set random seeds
    random.seed(parser.seed)
    np.random.seed(parser.seed)
    torch.manual_seed(parser.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(parser.seed)
    
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
    tokenizer = transformers.AutoTokenizer.from_pretrained(parser.model,
                                                           do_lower_case=parser.do_lower_case)
    label_names = parser.label_names.split(',')
    data_handler = myio.IO(data_dir    = data_path,
                           model_name  = parser.model,
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
    config = transformers.AutoConfig.from_pretrained(parser.model)
    classifier = model.Model(model=parser.model,
                             config = config,
                             n_hidden = parser.n_class_hidden,
                             n_flag = parser.n_labels
                             )
    
# =============================================================================
#     define trainer
# =============================================================================
    
    if os.path.isdir(parser.save_dir):
        save_path = parser.save_dir
    else:
        save_path = os.path.join(wd, parser.save_dir)
    
    log.info("Save Directory is {}.".format(save_path))
    log.info("="*40 + " Defining Trainer " + "="*40)
    
    # create trainer object
    trainer = learner.Learner(model          = classifier,
                              device         = device,
                              myio           = data_handler,
                              max_epochs     = parser.max_epochs,
                              save_path      = save_path,
                              lr             = parser.lr,
                              pct_start      = parser.pct_start,
                              anneal_strategy= parser.anneal_strategy,
                              log_int        = parser.log_int,
                              buffer_break   = not parser.no_early_stop,
                              break_int      = parser.patience,
                              accumulate_int = parser.grad_accum,
                              max_grad_norm  = parser.max_grad_norm,
                              )
            
    # train model
    best = trainer.learn(model_name  = parser.model,
                         early_check = parser.early_stop_criteria
                         )
    
    best['experiment'] = parser.exp_name
        
    #write results to "results.jsonl"
    results_name = os.path.join(parser.results_dir, "val_results.jsonl")
    with open(results_name, 'a') as f:
        f.write(json.dumps(best)+"\n")
    
    log.info("="*40 + " Program Complete " + "="*40)
    log.info("="*40 + " Results written to {} ".format(results_name) + "="*40)
    
if __name__ == "__main__":
    main()