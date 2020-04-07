"""
    Module to define our model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers as hug

class model(nn.Module):
    """
        Define the architecture of the model
    
        Need to check architecture with Phu
    """
    def __init__(self, ndim, nout, vocab_size, meta):
        # PLN definition task
        self.classification = nn.Sequential(
                nn.Linear(ndim, nout),
                nn.ReLU
                )
        
        self.MLM = nn.Softmax(vocab_size)
        
        # RLN definiton as BERT or meta-BERT
        self.representation = temp #BERT idk syntax
        
        # Load weights
        if meta:
            #load meta BERT weights to self.representation
        else:
            #load regular BERT weights
        
    def forward(self, data, mask_idx):
        embeddings = self.representation(data)
        
        # do task
        out = self.classification(embeddings)
        
        #do MLM
        masked_embedding = embeddings[mask_idx]
        mlm_out = self.MLM(masked_embedding)
        return out, mlm_out

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
                 dropout_prob=0.5,
                 meta
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
        