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
    
    # set up logger
    log_path = os.path.join(parser.save_dir, "logs")
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    
    log_fname = os.path.join(log_path, "{}_log_{}.log".format(
    parser.exp_name,dt.now().strftime("%Y%m%d_%H%M")))
    
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
    log.info("Data Directory is {}.".format(parser.data_dir))
    
# =============================================================================
#     import data
# =============================================================================
    task_names = parser.data_name.split(',')
    content_headers = parser.content.split(',')
    tokenizer = transformers.AutoTokenizer.from_pretrained(parser.model,
                                                           do_lower_case=parser.do_lower_case)
    data_handler = myio.IO(data_dir    = parser.data_dir,
                           model_name  = parser.model,
                           task_names  = task_names,
                           tokenizer   = tokenizer,
                           max_length  = parser.input_length,
                           content     = content_headers,
                           review_key  = parser.review_key,
                           label_name  = parser.label_name,
                           val_split   = parser.val_split,
                           test_split  = parser.test_split,
                           batch_size  = parser.batch_size,
                           shuffle     = not parser.no_shuffle,
                           cache       = not parser.no_cache,
                           )
    
    data_handler.read_task()

# =============================================================================
#     define model
# =============================================================================
    log.info("="*40 + " Defining Model " + "="*40)
    config = transformers.AutoConfig.from_pretrained(parser.model)
    classifier = model.Model(model     = parser.model,
                             config    = config,
                             n_others  = parser.n_others,
                             n_hidden  = parser.n_class_hidden,
                             n_flag    = parser.n_labels,
                             load      = parser.preload_emb,
                             load_name = parser.preload_emb_name,
                             )
    
# =============================================================================
#     define trainer
# =============================================================================
    
        
    log.info("Save Directory is {}.".format(parser.save_dir))
    log.info("="*40 + " Defining Trainer " + "="*40)
    
    # create trainer object
    trainer = learner.Learner(model           = classifier,
                              device          = device,
                              myio            = data_handler,
                              max_epochs      = parser.max_epochs,
                              save_path       = parser.save_dir,
                              lr              = parser.lr,
                              weight_decay    = parser.weight_decay,
                              pct_start       = parser.pct_start,
                              anneal_strategy = parser.anneal_strategy,
                              cycle_momentum  = parser.cycle_momentum,
                              log_int         = parser.log_int,
                              buffer_break    = not parser.no_early_stop,
                              break_int       = parser.patience,
                              accumulate_int  = parser.grad_accum,
                              max_grad_norm   = parser.max_grad_norm,
                              n_others        = parser.n_others,
                              batch_size      = parser.batch_size,
                              check_int       = parser.check_int,
                              save            = parser.save,
                              test            = parser.test,
                              )
            
    # train model
    best = trainer.learn(model_name  = parser.model,
                         task_name   = task_names[0],
                         early_check = parser.early_check
                         )
    
    best['experiment'] = parser.exp_name
        
    #write results to "results.jsonl"
    if not os.path.exists(parser.save_dir):
        os.mkdir(parser.save_dir)
    
    results_name = os.path.join(parser.save_dir, "results.jsonl")
    with open(results_name, 'a') as f:
        f.write(json.dumps(best)+"\n")
    
    log.info("="*40 + " Program Complete " + "="*40)
    log.info("="*40 + " Results written to {} ".format(results_name) + "="*40)
    
if __name__ == "__main__":
    main()