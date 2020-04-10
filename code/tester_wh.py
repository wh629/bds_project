"""
Testing for implementations - Will
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from datetime import datetime as dt
import os
import transformers
import logging as log
import sys

import myio
import model

# ============================= Testing Data Loading ==============================
wd = os.getcwd()


# for Spyder to get logging to work
root = log.getLogger()
while len(root.handlers):
    root.removeHandler(root.handlers[0])

# define logger
log_fname = os.path.join(wd, "logs", "log_{}.log".format(
    dt.now().strftime("%Y%m%d_%H%M")))
log.basicConfig(filename=log_fname,
    format='%(asctime)s - %(name)s - %(message)s',
    level=log.INFO)
root.addHandler(log.StreamHandler())

log.info('Start')
if True:
    
    # set parameters for IO object
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
    
    loss = nn.CrossEntropyLoss()
    
    config = transformers.AutoConfig.from_pretrained(rep_name)
    learner = model.Model(config, 5, 2)
    
    test_dl = data_handler.tasks.get('tester').get('test')
    
    with torch.no_grad():
        test_data, val_labels = next(iter(test_dl))
        rating, flag = learner(test_data)
    
        print(rating)
        print(flag)
        
        r_val, r_idx = torch.max(rating, dim=1)
        f_val, f_idx = torch.max(flag, dim=1)
        
        r_labels = val_labels[:,0]
        f_labels = val_labels[:,1]
        
        
        r_acc = metrics.accuracy_score(r_labels, r_idx)
        r_f1 = metrics.f1_score(r_labels, r_idx, average='macro')
        
        f_acc = metrics.accuracy_score(f_labels, f_idx)
        f_f1 = metrics.f1_score(f_labels, f_idx, average='macro')
        
        print(r_idx)
        print(r_labels)
        print(r_acc)
        print(r_f1)
        print('='*40)
        print(f_idx)
        print(f_labels)
        print(f_acc)
        print(f_f1)
        print('='*40)
        
        print(loss(rating, r_labels))
        print(loss(flag, f_labels))
        
log.info('End')   

# release logs from Python
handlers = log.getLogger().handlers
for handler in handlers:
    handler.close()
    

        
        
        
    
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            