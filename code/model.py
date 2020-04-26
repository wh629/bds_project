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
    def __init__(self,
                 model=None,
                 config=None, 
                 n_others=0, 
                 n_flag=None,
                 n_hidden = None,
                 load=False, 
                 load_name=None,
                 loss=None,
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
            self.representation.load_state_dict(torch.load(load_name))
        
        
        self.loss = loss
        if loss == None:
            self.loss = nn.CrossEntropyLoss()
        
    def forward(self,
                reviews=None,
                other_data=None,
                labels=None,
                ):
        
        # get shape of data
        batch_size, length = reviews.size()
        
        # generate position ids of tokens
        positions = torch.arange(length)        
        position_ids = positions.repeat(batch_size,1)
        
        # pass through embedding network
        embeddings = self.representation(
                reviews,
                position_ids=position_ids
                )[0]
        
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
        loss = self.loss(logits, labels)
        
        return loss, logits
        