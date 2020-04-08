"""
Testing for implementations - Will
"""

import torch
import numpy as np
from datetime import datetime as dt
import os
import transformers

import myio
import model

# ============================= Testing Data Loading ==============================
if True:
    
    # set parameters for IO object
    wd = os.getcwd()
    data_dir = os.path.join(wd, r'cleaned')
    task_names = ['tester']
    tokenizer = transformers.AutoTokenizer.from_pretrained('albert-base-v2')
    max_length = 512
    
    # read in 'tester' data in both train and dev directories
    # only do batch_size of 2
    data_handler = myio.IO(data_dir, task_names, tokenizer, max_length, batch_size = 2)
    data_handler.read_task()  
    
    # see that it works
    if False:
        for use in ['train','dev']:
            # get training data_loader
            dl = data_handler.tasks.get('tester').get(use)
            for i,(data, labels) in enumerate(dl):
                print(r'{} batch {} data size is: {}'.format(use, i, data.size()))
                print(r'{} batch {} data is: {}'.format(use, i, data))
                
                for k, obs in enumerate(data):
                    print(r'{} batch {} obs {} decoded: {}'.format(use, i, k, tokenizer.decode(obs.tolist())))
                    
                    print(r'{} batch {} size is: {}'.format(use, i, labels.size()))
                    print(r'{} batch {} is: {}'.format(use, i, labels))

# ============================= Test Model ==============================
if True: 
    rep_name = 'albert-base-v2'
    
    
    config = transformers.AutoConfig.from_pretrained(rep_name)
    learner = model.Model(config, 5, 2)
    
    val_dl = data_handler.tasks.get('tester').get('dev')
    val_data, val_labels = next(iter(val_dl))
    rating, flag = learner(val_data)
    
    print(rating)
    print(flag)
    
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            