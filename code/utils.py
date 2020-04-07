"""
Module for general utility methods like generalized training loop, evaluation,
fine-tuning, and getting data.

Mainly for continual learning.
"""

import torch
import tqdm
import torch.nn as nn
import torch.optim as opt
import numpy as np
from sklearn import metrics

def train(
        model,          # model to train
        train_loader,   # DataLoader with training data
        loss,           # loss function object
        optimizer,      # optimizer object
        device,         # device to run training
        ):
    """
        Training loop
    
        Will look something like this.
        
        NOT TESTED YET
    """    
    for i, (data, label) in enumerate(tqdm(train_loader)):        
        # put model in train mode
        model.train()
        
        # send data and labels to device
        data, label = data.to(device), label.to(device)
        
        # zero gradients related to optimizer
        optimizer.zero_grad()
        
        # send data through model forward
        out = model(data)
        
        # calculate loss
        l = loss(out, label)
        
        # calculate gradients through back prop
        l.backward()
        
        #take a step in gradient descent
        optimizer.step()
    
def evaluate(
        model,        # model to train
        data,         # validation data
        label,        # validation labels
        loss,         # loss function object
        epoch,        # epoch number for tracking
        device        # device to perform calculations
        ):
    """
        Evaluation model on data. (May need to use dataloader as well)
        
        Returns: Loss, F1
        
        NOT TESTED YET
    """
    # puts model in evaluation mode
    model.eval()
    
    # don't need to track gradient
    with torch.no_grad():
        data, label = data.to(device), data.to(device)
        out = model(data)
        l = loss(out,label)
        
        # prepare out and label for F1 calculation
        if device == 'cuda':
            out, label = out.to('cpu'), label.to('cpu')
        
        f1 = metrics.f1_score(label,out)
    
    return l, f1
    
def fine_tune(
        model,                 # model to be fine tuned
        train_data,            # train dataloader
        val_data,              # validation data
        val_label,             # validation labels
        loss,                  # loss function object
        optimizer,             # optimizer object
        device,                # device to train and evaluate model
        n_epochs,              # number of epochs to train
        anneal_threshold,      # buffer for when to anneal
        anneal_r,              # rate to cut learning rate
        save_path,             # directory storing saved model weights
        model_name,            # name of model used
        verbose = True,        # flag whether to print checking
        log_int = 10000,       # number of epochs between printing logs
        save_int = 10000,      # number of intervals to save parameters
        check_best_int = 1000, # number of intervals for saving best parameters
        best = False           # whether to record only best weights
        ):
    """
        Fine-tune model on task with train and val data
        
        Return: List of Saved Weight Paths
        
        NOT TESTED YET
    """
    losses =  np.array([])
    best = 0
    rln_paths = []
    best_path = save_path + r'\\' + model_name + r'_best.pt'
    not_improved = 0
    
    for epoch in tqdm(range(n_epochs)):
        train(model, train_data, loss, optimizer, device)
        loss, f1 = evaluate(model, val_data, loss)
        
        # track validation
        losses.append([loss])
        
        # keep track of best F1
        if f1 >= best:
            best = f1
            not_improved = 0
        else:
            not_improved += 1
            
            # anneal learning rate if stops improving
            if not_improved >= anneal_threshold:
                optimizer.param_group['lr'] *= anneal_r
                print('Annealed. LR is {:.2f}'.format(optimizer.param_group['lr']))
        
        # print results every interval
        if epoch % log_int == 0 and verbose:
            print('\nTest set: Best F1: {:.4f}, Average Loss: {:.4f}'.format(
                    best, np.sum(losses)/epoch))
        
        # save best weights
        if epoch % check_best_int == 0 and f1 == best:
            torch.save(model.representation, best_path)
        
        # save weights
        if epoch % save_int == 0:
            temp_path = save_path + r'\\' + model_name + r'.pt'
            rln_paths.append(temp_path)
            
            if best:
                best_model = torch.load(best_path)
                torch.save(best_model, temp_path)
            else:
                torch.save(model.representation, temp_path)
    
    return rln_paths