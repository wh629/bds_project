"""
Module to define our model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        print(type(embeddings))
        cls_embeddings = embeddings[:,0,:]
        
        # do tasks
        rating_logits = self.rating(cls_embeddings)
        flag_logits = self.flag(cls_embeddings)
        
        return rating_logits, flag_logits

# ============================ Tester Model ============================

class test_model(nn.Module):
    """
        Small model for utils, io, continual learning testing
        
        Copied from HW1 Part 2
    """
    
    def __init__(self,
                 ndim,
                 nout,
                 nhid,
                 embeddings,
                 meta,
                 dropout_prob=0.5                 
                 ):
        """
            Constructor of test model. Will use small LSTM
        """
        self.embedding_layer = self.load_pretrained_emb(embeddings)
        self.lstm = nn.LSTM(input_size=embeddings.shape[1], hidden_size=nhid, nlayers=1, batch_first=True)
        self.clf = nn.Linear(in_features=nhid, out_features=nout)
        self.dropout = nn.Dropout(p=dropout_prob)
    
    def load_pretrained_emb(self, embeddings):
        """
            Code for loading embeddings
        """
        layer = nn.Embedding(embeddings.shape[0], embeddings.shape[1], padding_idx=0)
        layer.weight.data = torch.Tensor(embeddings).float()
        return layer
    
    def forward(self,
                data
                ):
        """
            Forward for small model from HW1 Part 2
        """
        
        # Embedding
        embedded = self.embedding_layer(data)
        
        # Dropout
        kept = self.dropout(embedded)
        
        # LSTM
        out_lstm, = self.lstm(kept)
        
        # Mean pooling
        pooled = out_lstm.mean(dim=1)
        
        # Apply ReLU
        nonlin = nn.ReLU(pooled)
        
        return self.clf(logits)
        