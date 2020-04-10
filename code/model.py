"""
Module to define our model
"""

import torch
import torch.nn as nn
import transformers

class Model(nn.Module):
    """
    Define the architecture of the model.
    """
    def __init__(self, config, nrating, nflag, load=False, load_name=None):
        super(Model, self).__init__()
        self.representation = transformers.AutoModel.from_config(config)
        
        # rating classifier
        self.rating = nn.Sequential(
            nn.Linear(config.hidden_size, nrating)
            )
        
        # false review classifier
        self.flag = nn.Sequential(
            nn.Linear(config.hidden_size,nflag)
            )
        
        if load:
            self.representation.load_state_dict(torch.load(load_name))
        
    def forward(self,data):
        
        # get shape of data
        batch_size, length = data.size()
        
        # generate position ids of tokens
        positions = torch.arange(length)        
        position_ids = positions.repeat(batch_size,1)
        
        # pass through embedding network
        embeddings = self.representation(
                data,
                position_ids=position_ids
                )[0]
        
        # get [CLS] embedding
        cls_embeddings = embeddings[:,0,:]
        
        # do tasks
        rating_logits = self.rating(cls_embeddings)
        flag_logits = self.flag(cls_embeddings)
        
        return rating_logits, flag_logits
        