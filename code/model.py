"""
Module to define our model
"""

import torch
import torch.nn as nn
import transformers
import os
import logging as log

class Model(nn.Module):
    """
    Define the architecture of the model.
    """
    def __init__(self,
                 model=None,        # model name
                 config=None,       # Huggingface model configuration
                 n_others=0,        # number of other statistics
                 n_flag=2,          # number of classifications for flagging
                 n_hidden = 0,      # hidden dimension for MLP classifier
                 load=False,        # whether to preload embedding weights
                 load_name=None,    # name of weights state dictionary file
                 loss=None,         # type of loss. default is CrossEntropyLoss()
                 ):
        super(Model, self).__init__()
        self.representation = transformers.AutoModel.from_pretrained(model, config=config)
        
        if n_hidden == 0:
            n_hidden = config.hidden_size+n_others
        
        # false review classifier
        self.flag = nn.Sequential(
            nn.Linear(config.hidden_size+n_others, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_flag)
            )
        
        if load:
            assert os.path.exists(load_name), "preloaded embeddings don't exist: {}".format(load_name)
            self.representation.load_state_dict(torch.load(load_name))
        
        self.loss = loss
        if loss is None:
            self.loss = nn.CrossEntropyLoss()
        
    def forward(self,
                reviews=None,
                other_data=None,
                labels=None,
                ):
        # pass through embedding network
        embeddings = self.representation(reviews)[0]
        
        # get [CLS] embedding
        cls_embeddings = embeddings[:,0,:]
        
        cls_embeddings = torch.squeeze(cls_embeddings)
        
        if other_data is None:
            inputs = cls_embeddings
        else:
            inputs = torch.cat((cls_embeddings,other_data), dim=1)
        
        # compute fake review logits
        logits = self.flag(inputs)
        
        # compute loss
        try:
            loss = self.loss(logits, labels)
        except:
            log.info("Failed loss calculation. Logits shape is {}".format(logits.shape))
            log.info("Logits {}".format(logits))
            log.info("Labels {}".format(labels))
        
        return loss, logits
        