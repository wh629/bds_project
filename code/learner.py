"""
Class for learning
"""

import torch
#from tqdm import tqdm
import torch.nn as nn
import torch.optim as opt
import sklearn
import logging as log
import math
import os
#import pickle
#import numpy as np

class Learner():
    def __init__(self,
                 model,                     # model to train
                 device,                    # device to run on
                 train_data,                # training dataloader
                 val_data,                  # validation dataloader
                 test_data,                 # test dataloader
                 rating_w,                  # weight for rating loss
                 flag_w,                    # weight for flag loss
                 max_epochs,                # maximum number of epochs
                 save_path,                 # path to save best weights
                 rating_loss=None,          # loss object for rating objective
                 flag_loss=None,            # loss object for flag objective
                 optimizer=None,            # optimizer for training
                 scheduler=None,            # scheduler for optimizer
                 lr=0.001,                  # optimizer learning rate
                 weight_decay=0.01,         # optimizer weight decay
                 pct_start = 0.3,           # scheduler warm-up percentage
                 anneal_strategy='linear',  # scheduler annealing strategy
                 cycle_momentum=False,      # scheduler whether to alternate momentums
                 log_int=1e4,               # logging interval
                 buffer_break=False,        # whether to do early breaking
                 break_int=10,              # buffer before early breaking
                 f1_average='macro'         # averaging method for multi-class F1
                 ):
        """
        Class for learning
        """
        
        # set attributes
        self.model = model.to(device)
        self.device = device
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.rating_w = rating_w
        self.flag_w = flag_w
        self.max_epochs = max_epochs
        self.save_path = save_path
        self.rating_loss = rating_loss
        self.flag_loss = flag_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_log_int = math.min(len(train_data.dataset)-1, log_int)
        self.val_log_int = math.min(len(val_data.dataset)-1, log_int)
        self.buffer_break = buffer_break
        self.break_int = break_int
        self.f1_avg = f1_average
        
        
        if rating_loss == None:
            self.rating_loss = nn.CrossEntropyLoss()
        
        if flag_loss == None:
            self.flag_loss = nn.CrossEntropyLoss()
            
        if optimizer == None:
            self.optimizer = opt.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=0.01
                )
        
        if scheduler == None:
            self.scheduler = opt.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=lr,
                epochs=max_epochs,
                steps_per_epoch=len(train_data.dataset),
                pct_start=pct_start,
                anneal_strategy=anneal_strategy,
                cycle_momentum=cycle_momentum
                )
        
        # if directory for best weights does not exist, create it
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
    
# =============================================================================
#     Helper methods
# =============================================================================
    
    def accumulate_metrics(self,
                           metrics,  # dictionary of metrics
                           loss,     # loss value
                           r_logits, # rating logits
                           r_label, # rating labels
                           f_logits, # flagged logits
                           f_label  # flagged labels
                           ):
        """
        Helper method to accumulate metrics.
        
        Metrics should have keys: values as
        
        {
            "loss"      : <Cumulative Loss>
            "rating_acc": <Cumulative Rating Accuracy>
            "flag_acc"  : <Cumulative Flag Accuracy>
            "rating_f1" : <Cumulative Rating F1>
            "flag_f1"   : <Cumulative Flag F1>
            }
        """
        
        with torch.no_grad():
            # accumulate measures
            metrics['loss'] += loss
            
            _, r_idx = torch.max(r_logits, dim=1)
            _, f_idx = torch.max(f_logits, dim=1)
                
            # prepare data for numpy metrics
            if self.device == 'cuda':
                r_label, f_label = r_label.to('cpu'), f_label.to('cpu')
                r_idx, f_idx = r_idx.to('cpu'), f_idx.to('cpu')
                
            metrics['rating_acc'] += sklearn.metrics.accuracy_score(r_label, r_idx)
            metrics['flag_acc'] += sklearn.metrics.accuracy_score(f_label, f_idx)
            metrics['rating_f1'] += sklearn.metrics.f1_score(r_label, r_idx, average=self.f1_avg)
            metrics['flag_f1'] += sklearn.metrics.f1_score(f_label, f_idx, average=self.f1_avg)
    
    def average_metrics(self,
                        metrics,
                        interval
                        ):
        """
        Helper method to average metrics.
        
        Metrics should have keys: values as
        
        {
            "loss"      : <Cumulative Loss>
            "rating_acc": <Cumulative Rating Accuracy>
            "flag_acc"  : <Cumulative Flag Accuracy>
            "rating_f1" : <Cumulative Rating F1>
            "flag_f1"   : <Cumulative Flag F1>
            }
        
        ======================================================================
        Modifies metrics to get by adding last 2 entries:
        
        {
            "loss"       : <Average Loss>,
            "rating_acc" : <Average Rating Accuracy>,
            "flag_acc"   : <Average Flag Accuracy>,
            "rating_f1"  : <Average Rating F1>,
            "flag_f1"    : <Average Flag F1>,
            "comb_acc"   : <Weighted Average Accuracy>,
            "comb_f1"    : <Weighted Average F1>
            }
        
        """
        
        # average metrics
        for metric in metrics.keys():
            metrics[metric] /= (interval)
        
        # weighted average of task metrics
        metrics['comb_acc'] = (self.rating_w*metrics['rating_acc'] + 
                               self.flag_w*metrics['flag_acc'])/(self.rating_w + self.flag_w)
        metrics['comb_f1'] = (self.rating_w*metrics['rating_f1'] + 
                               self.flag_w*metrics['flag_f1'])/(self.rating_w + self.flag_w)
    
    def log_results(self,
                    results,
                    log_type,
                    early_check = None,
                    epoch = None
                    ):
        """
        Helper method for logging information
        """
        
        # make sure logging is valid
        assert log_type in ['Training', 'Validation', 'Testing'], 'Learner logging type {} not supported'.format(log_type)
        assert log_type == 'Training' and type(epoch) == int, 'Need integer epoch with log_type "Training".'
        assert log_type == 'Validation' and (type(early_check) == str and type(epoch) == int), 'Need string early_check and integer epoch with log_type "Validation".'
        
        # build string for logging
        logging_string = '{} Information: | Loss: {:.4f} | Rating Accuracy: {:.4f} | Rating F1: {:.4f}'\
            ' | Flag Accuracy: {:.4f} | Flag F1: {:.4f} | Combined Accuracy: {:.4f}'\
                ' | Combined F1: {:.4f}'.format(
                    log_type,
                    results['loss'],
                    results['rating_acc'],
                    results['rating_f1'],
                    results['flag_acc'],
                    results['flag_f1'],
                    results['comb_acc'],
                    results['comb_f1']
                    )
        
        if log_type == 'Training':
            logging_string += ' | Current Epoch: {}'.format(epoch)
        elif log_type == 'Validation':
            logging_string += ' | Best Epoch: {} | Stop Check Type: {}'.format(
                epoch,
                early_check
                )
        
        log.info(logging_string)
    
# =============================================================================
#     Training helper methods
# =============================================================================
    
    def train(self, 
              epoch # integer of epoch
              ):
        """
        Training loop for a single epoch.
        
        ======================================================================
        
        RETURNS: Results as dictionary
        
        {
            "loss"       : <Average Loss>,
            "rating_acc" : <Average Rating Accuracy>,
            "flag_acc"   : <Average Flag Accuracy>,
            "rating_f1"  : <Average Rating F1>,
            "flag_f1"    : <Average Flag F1>,
            "comb_acc"   : <Weighted Average Accuracy>,
            "comb_f1"    : <Weighted Average F1>
            }
        
        NOT TESTED YET
        """
        log.info('Training | Epoch {} out of {}'\
                 .format(epoch+1, self.max_epochs+1))
        
        metrics = {
            "loss"      : 0,
            "rating_acc": 0,
            "flag_acc"  : 0,
            "rating_f1" : 0,
            "flag_f1"   : 0
            }
        
        for i, (data, label) in enumerate(self.train_data):        
            # put model in train mode
            self.model.train()

            # send data and labels to device
            data, labels = data.to(self.device), label.to(self.device)
            r_label = labels[:,0] # collate has rating labels 1st
            f_label = labels[:,1] # collate has flag labels 2nd
    
            # zero gradients related to optimizer
            self.optimizer.zero_grad()
    
            # send data through model forward
            r_logits, f_logits = self.model(data)
    
            # calculate loss
            l = self.rating_loss(r_logits, r_label)*self.rating_w \
                + self.flag_loss(f_logits, f_label)*self.flag_w
            # calculate gradients through back prop
            l.backward()
    
            #take a step in optimizer and scheduler
            self.optimizer.step()
            self.scheduler.step()
            
            # accumulate metrics
            self.accumulate_metrics(metrics,
                                    l,
                                    r_logits,
                                    r_label,
                                    f_logits,
                                    f_label)
            
        
        # average metrics
        self.average_metrics(metrics, i+1)
        
        return metrics

    def evaluate(self,
                 data_loader,  # dataloader to evaluate on
                 epoch = None, # integer of epoch number if val = True
                 val = True    # boolean whether evaluating validation data
                ):
        """
        Evaluation model on data. (May need to use dataloader as well)
    
        ======================================================================
        
        RETURNS: Results as dictionary
        
        {
            "loss"       : <Average Loss>,
            "rating_acc" : <Average Rating Accuracy>,
            "flag_acc"   : <Average Flag Accuracy>,
            "rating_f1"  : <Average Rating F1>,
            "flag_f1"    : <Average Flag F1>,
            "comb_acc"   : <Weighted Average Accuracy>,
            "comb_f1"    : <Weighted Average F1>
            }
    
        NOT TESTED YET
        """
        if val:
            log.info('Validation | Epoch {} out of {}'\
                     .format(epoch+1, self.max_epochs+1))
        
        metrics = {
            "loss"      : 0,
            "rating_acc": 0,
            "flag_acc"  : 0,
            "rating_f1" : 0,
            "flag_f1"   : 0
            }
        
        # puts model in evaluation mode
        self.model.eval()
        
        # don't need to track gradient
        with torch.no_grad():
            for i, (data, labels) in enumerate(data_loader):
                # send data and labels to device
                data, labels = data.to(self.device), labels.to(self.device)
                r_label = labels[:,0] # collate has rating labels 1st
                f_label = labels[:,1] # collate has flag labels 2nd
                
                # send data through forward
                r_logits, f_logits = self.model(data)
                
                # calculate loss
                l = self.rating_loss(r_logits, r_label)*self.rating_w \
                    + self.flag_loss(f_logits, f_label)*self.flag_w
                
                # accumulate metrics
                self.accumulate_metrics(metrics,
                                        l,
                                        r_logits,
                                        r_label,
                                        f_logits,
                                        f_label)
        
        # average metrics
        self.average_metrics(metrics, i+1)
        
        return metrics

# =============================================================================
#     Learning method
# =============================================================================

    def learn(self,
              model_name,            # name of model used
              verbose = True,        # flag whether to print checking
              early_check = 'loss'   # key for early break check
              ):
        """
        Train model
        
        ======================================================================
        
        RETURNS: Path of best model weights and best embedding weights
        
        NOT TESTED YET
        """
        best = {
            'loss'       : None,
            'rating_acc' : None,
            'flag_acc'   : None,
            'rating_f1'  : None,
            'flag_f1'    : None,
            'comb_acc'   : None,
            'comb_f1'    : None
            }
        best_path = os.path.join(self.save_path, model_name + r'_best_full.pt')
        best_emb_path = os.path.join(self.save_path, model_name + r'_best_emb.pt')
        best_epoch = 0
        stop = False
        
        log.info('='*40 + ' Start Training ' + '='*40)
        for epoch in range(self.max_epochs):
            train_results = self.train(epoch)
            val_results = self.evaluate(self.val_data, epoch=epoch)
            
            # check for best validation score for each metric
            for result_type in val_results.keys():
                temp_val = val_results[result_type]
                
                if ((temp_val < best[result_type] and result_type.find('loss') != -1)
                    or temp_val > best[result_type]):
                    
                    # save best score
                    best[result_type] = temp_val
                    
                    # record embedding weights if it's the metric we care about
                    if result_type == early_check:
                        # total model weights
                        torch.save(self.model.state_dict(),
                                   best_path)
                        
                        # embedding weights
                        torch.save(self.model.representation.state_dict(),
                                   best_emb_path)
                        
                        best_epoch = epoch
                
                # check if early breaking requirements satisfied
                elif (result_type == early_check
                      and epoch-best_epoch > self.break_int
                      and self.buffer_break
                      ):
                    stop = True
                    break
                
            if stop:
                log.info('='*40 + ' Early Stop at Epoch: {} '.format(epoch) + '='*40)
                break
    
            # log results every interval
            if epoch % self.log_int == 0 and verbose:
                self.log_results(train_results,
                                 log_type = 'Training',
                                 epoch = epoch
                                 )
                self.log_results(best,
                                 log_type = 'Validation',
                                 early_check = early_check,
                                 epoch = best_epoch
                                 )
        
        # log best validation results
        self.log_results(best,
                         log_type = 'Validation',
                         early_check = early_check,
                         epoch = best_epoch
                         )
        
        # test best model on test data
        self.model.load_state_dict(torch.load(best_path))
        test_results = self.evaluate(self.test_data, val=False)
        
        self.log_results(test_results,
                         log_type = 'Testing'
                         )
        
        return best_path, best_emb_path