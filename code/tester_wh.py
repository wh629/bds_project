"""
Testing for implementations - Will
"""

import torch
import analyze
import numpy as np
from datetime import datetime as dt
import os
import json
import transformers
import myio

# ============================= Testing Analyze ==============================
# Generate test data
if False:
    btrivia = {"iter":np.arange(11),"val":np.arange(11)}
    bsquad = {"iter":np.arange(11),"val":0.5*np.arange(11)}
    mtrivia = {"iter":np.arange(11),"val":2*np.arange(11)}
    msquad = {"iter":np.arange(11),"val":1.5*np.arange(11)}

    data = {
            "BERT Trivia":btrivia,
            "BERT SQuAD":bsquad,
            "Meta-BERT Trivia":mtrivia,
            "Meta-BERT SQuAD":msquad
            }

    # Test plotting
    plot = analyze.plot_learning(data, iterations=10, max_score=20, x_tick_int=2, y_tick_int=10)

    # Tryout displaying and saving plot
    #
    # Datetime string formatting:
    # %Y = year
    # %m = month
    # %d = day
    # %H = hour
    # %M = minute
    plot.show
    plot.savefig("./results/test_fig_{}.png".format(dt.now().strftime("%Y%m%d_%H%M")))
# ============================= Testing Analyze ==============================
    
# ============================= Testing Data Loading ==============================
if True:
    
    # set parameters for IO object
    wd = os.getcwd()
    data_dir = os.path.join(wd, r'test_data')
    task_names = ['tester']
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
    max_length = 512
    
    # read in 'tester' data in both train and dev directories
    # only do batch_size of 2
    data_handler = myio.IO(data_dir, task_names, tokenizer, max_length, batch_size = 2)
    data_handler.read_task(True)  
    
    # see that it works
    for use in ['train','dev']:
        # get training data_loader
        dl = data_handler.tasks.get('tester').get(use)
        for i,(data, labels) in enumerate(dl):
            print(r'{} batch {} data size is: {}'.format(use, i, data.size()))
            print(r'{} batch {} data is: {}'.format(use, i, data))
            
            for k, obs in enumerate(data):
                print(r'{} batch {} obs {} decoded: {}'.format(use, i, k, tokenizer.decode(obs.tolist())))
            
            for j, answer in enumerate(labels):
                a = torch.tensor(answer)
                print(r'{} batch {} answer {} size is: {}'.format(use, i, j, a.size()))
                print(r'{} batch {} answer {} is: {}'.format(use, i, j, a))

            
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            