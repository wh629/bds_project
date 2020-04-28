"""
Class for learning
"""

import torch
from tqdm import tqdm, trange
import torch.nn as nn
import torch.optim as opt
import sklearn
import logging as log
import math
import os
import pickle
import numpy as np

class Learner():
    def __init__(self,
                 model=None,                # model to train
                 device=None,               # device to run on
                 myio=None,                 # myio object for data loading
                 max_epochs=None,           # maximum number of epochs
                 save_path=None,            # path to save best weights
                 optimizer=None,            # optimizer for training
                 lr=0.001,                  # optimizer learning rate
                 weight_decay=0.0,          # optimizer weight decay
                 pct_start = 0.00,          # scheduler warm-up percentage
                 anneal_strategy='linear',  # scheduler annealing strategy
                 cycle_momentum=False,      # scheduler whether to alternate momentums
                 log_int=1,                 # logging interval
                 buffer_break=False,        # whether to do early breaking
                 break_int=10,              # buffer before early breaking
                 f1_average='macro',        # averaging method for multi-class F1
                 accumulate_int=1,          # interval for accumulating gradients
                 max_grad_norm=1,           # maximum gradient norm
                 n_others=0,                # number of other statistics
                 batch_size=8,              # batch size
                 ):
        """
        Class for learning
        """
        
        # set attributes
        self.model = model.to(device)
        self.device = device
        self.IO = myio
        self.max_epochs = max_epochs
        self.save_path = save_path
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.buffer_break = buffer_break
        self.break_int = break_int
        self.f1_avg = f1_average
        self.accum_int = accumulate_int
        self.max_grad_norm = max_grad_norm
        self.n_others = n_others
        self.batch_size = batch_size
        
        # for multi-gpu
        if torch.cuda.is_avalailable() and torch.cuda.device_count() > 1 and not isinstance(self.model, nn.DataParallel):
            self.model = nn.DataParallel(self.model)
            self.model.to(self.device)
            
        if optimizer == None:
            self.optimizer = opt.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=self.weight_decay
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
                           logits, # rating logits
                           label, # rating labels
                           ):
        """
        Helper method to accumulate metrics.
        
        Metrics should have keys: values as
        
        {
            "loss" : <Cumulative Loss>
            "acc"  : <Cumulative Accuracy>
            "f1"   : <Cumulative F1>
            }
        """
        
        with torch.no_grad():
            # accumulate measures
            metrics['loss'] += loss
            
            _, idx = torch.max(logits, dim=1)
                
            # prepare data for numpy metrics
            if torch.cuda.is_available():
                idx, label = idx.to('cpu'), label.to('cpu')
                
            metrics['acc'] += sklearn.metrics.accuracy_score(label, idx)
            metrics['f1'] += sklearn.metrics.f1_score(label, idx, average=self.f1_avg)
    
    def average_metrics(self,
                        metrics,
                        interval
                        ):
        """
        Helper method to average metrics.
        
        Metrics should have keys: values as
        
        {
            "loss"      : <Cumulative Loss>
            "acc"       : <Cumulative Accuracy>
            "f1"        : <Cumulative F1>
            }
        
        ======================================================================
        Modifies metrics to get by adding last 2 entries:
        
        {
            "loss"       : <Average Loss>,
            "acc"        : <Average Accuracy>,
            "f1"         : <Average F1>
            }
        
        """
        
        # average metrics
        for metric in metrics.keys():
            metrics[metric] /= (interval)
    
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
        logging_string = '{} Information: | Loss: {:.4f} | Accuracy: {:.4f} | F1: {:.4f}'.format(
            log_type,
            results['loss'],
            results['acc'],
            results['f1'])
        
        if log_type == 'Training':
            logging_string += ' | Current Epoch: {}'.format(epoch)
        elif log_type == 'Validation':
            logging_string += ' | Best Epoch: {} | Stop Check Type: {}'.format(
                epoch,
                early_check
                )
        
        log.info(logging_string)
    
    def pack_inputs(self, reviews, data, labels):
        """
        Packs data into a dictionary for the model
        """
        if self.n_others > 0:
            others = data
        else:
            others = None
        
        results = {'reviews'    : reviews,
                   'other_data' : others,
                   'labels'     : labels}
        
        return results
        
# =============================================================================
#     Training helper methods
# =============================================================================
    
    def train_step(self, 
              iteration,   # integer of epoch
              accumulated, # number of accumulation steps
              reviews,     # reviews for batch
              data,        # data for batch
              labels,      # labels for batch
              ):
        """
        Training loop for a single step.
        
        ======================================================================
        
        RETURNS: Results as dictionary
        
        {
            "loss"       : <Loss>,
            "acc"        : <Accuracy>,
            "f1"         : <F1>
            }
        """
        log.info('Training | Step {} out of {}'\
                 .format(iteration+1, self.max_steps+1))
        
        metrics = {
            "loss"      : 0,
            "acc"       : 0,
            "f1"        : 0
            }
        
        
        self.model.train()
        
        # send data and labels to device
        data = data.to(self.device)
        reviews = reviews.to(self.device)
        labels = labels.to(self.device)
        input_data = self.pack_inputs(reviews, data, labels)
        
        # zero gradients related to optimizer if new start of accumulation
        if accumulated == 0:
            self.model.zero_grad()
    
        # send data through model forward
        l, logits = self.model(**input_data)
        
        # for multi-gpu
        if isinstance(self.model, nn.DataParallel):
            l = l.mean() # average over multi-gpu loss
        
        
        # calculate gradients through back prop
        l.backward()
        
        accumulated += 1
        
        #take a step in optimizer and scheduler
        if accumulated == self.accumulate_int:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()   
            self.scheduler.step()
            
            accumulated = 0
            
            self.model.zero_grad()
            
        # accumulate metrics
        self.accumulate_metrics(metrics,
                                l,
                                logits,
                                labels)
        
        return metrics, accumulated

    def evaluate(self,
                 data_loader,      # dataloader to evaluate on
                 iteration = None, # integer of epoch number if val = True
                 val = True        # boolean whether evaluating validation data
                ):
        """
        Evaluation model on data. (May need to use dataloader as well)
    
        ======================================================================
        
        RETURNS: Results as dictionary
        
        {
            "loss"       : <Average Loss>,
            "acc"        : <Average Accuracy>,
            "f1"         : <Average Rating F1>
            }
        """
        if val:
            log.info('Validation | Step {} out of {}'\
                     .format(iteration+1, self.max_steps+1))
        
        metrics = {
            "loss"      : 0,
            "acc"       : 0,
            "f1"        : 0
            }
        
        # puts model in evaluation mode
        self.model.eval()
        
        # don't need to track gradient
        with torch.no_grad():
            for i, (reviews, data, labels) in enumerate(data_loader):
                # send data and labels to device
                data = data.to(self.device)
                reviews = reviews.to(self.device)
                labels = labels.to(self.device)
                input_data = self.pack_inputs(reviews, data, labels)
                
                # send data through forward
                l, logits = self.model(**input_data)
                
                # for multi-gpu
                if isinstance(self.model, nn.DataParallel):
                    l = l.mean() # average over multi-gpu loss
                
                # accumulate metrics
                self.accumulate_metrics(metrics,
                                        l,
                                        logits,
                                        labels)
        
        # average metrics
        self.average_metrics(metrics, i+1)
        
        return metrics

# =============================================================================
#     Learning method
# =============================================================================

    def learn(self,
              model_name = None,          # name of model used
              task_name = 'reviews_UIC',  # name of data set to use
              scheduler = None,           # learning rate scheduler
              verbose = True,             # flag whether to print checking
              early_check = 'loss'        # key for early break check
              ):
        """
        Train model
        
        ======================================================================
        
        RETURNS: Dictionary of Results as
        {
            'val_loss'     : <best validation loss>
            'val_acc'      : <best validation accuracy>
            'val_f1'       : <best validation f1>
            'best_path'    : <file of best model state_dict>
            'best_step'    : <step of best model>
            'total_steps'  : <total number of update steps>
            'total_epochs' : <total number of epochs>
            'test_loss'    : <test loss>
            'test_acc'     : <test accuracy>
            'test_f1'      : <test f1>
            }
        """
        
        best_path = os.path.join(self.save_path, model_name + '_best.pt')
        best_rln_path = os.path.join(self.save_path, model_name + '_best_rln.pt')
        best_step = 0
        best_epoch = 0
        stop = False
        
        train_data = self.IO.tasks[task_name]['train']
        val_data = self.IO.tasks[task_name]['dev']
        test_data = self.IO.tasks[task_name]['test']
        
        # set max steps
        self.max_steps = ((self.max_epochs*len(train_data.dataset))//self.batch_size)//self.accum_int
        
        if scheduler == None:
            self.scheduler = opt.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.lr,
                total_steps=self.max_steps,
                pct_start = self.pct_start,
                anneal_strategy = self.anneal_strategy,
                cycle_momentum = self.cycle_momentum
                )
        
        
        best = {
            'val_loss'     : float("inf"),
            'val_acc'      : 0.0,
            'val_f1'       : 0.0,
            'best_path'    : best_path,
            'best_step'    : 0,
            'total_steps'  : self.max_steps,
            'total_epochs' : self.max_epochs
            }        
        
        global_step = 0
        accumulated = 0
        
        log.info('='*40 + ' Start Training ' + '='*40)
        log.info('='*40 + ' Max {} Epochs, Max {} Steps'.format(self.max_epochs, self.max_steps) + '='*40)
        log.info('='*40 + ' Using {} GPUs with {} Accumulation Steps'.format(torch.cuda.device_count(), self.accum_int) + '='*40)
        for epoch in trange(0, int(self.max_epochs), desc = 'Epoch', mininterval = 30):
            
            # train
            for i, (reviews, data, labels) in enumerate(tqdm(train_data, desc='Epoch Iteration', mininterval=30)):
                train_results, accumulated = self.train_step(i, accumulated, reviews, data, labels)
                global_step += 1
            
            # evaluate every epoch
            val_results = self.evaluate(val_data, iteration=global_step)
                
            # check for best validation score for each metric
            for result_type, result_val in val_results.items():                    
                if ((result_val < best['val_{}'.format(result_type)] and result_type.find('loss') != -1)
                    or (result_val > best['val_{}'.format(result_type)] and result_type.find('loss')== -1)):
                        
                    # save best score
                    best['val_{}'.format(result_type)] = result_val
                    
                    # record embedding weights if it's the metric we care about
                    if result_type == early_check:
                        # total model weights
                        if isinstance(self.model, nn.DataParallel):
                            save_state = self.model.module.state_dict()
                            save_rln_state = self.model.module.representation.state_dict()
                        else:
                            save_state = self.model.state_dict()
                            save_rln_state = self.model.representation.state_dict()
                        
                        torch.save(save_state, best_path)
                        torch.save(save_rln_state, best_rln_path)
                        best_step = global_step
                        best_epoch = epoch
                    
                # check if early breaking requirements satisfied
                elif (result_type == early_check
                      and epoch-best_epoch > self.break_int
                      and self.buffer_break
                      ):
                    stop = True
                    
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
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(torch.load(best_path))
        else:
            self.model.load_state_dict(torch.load(best_path))
        test_results = self.evaluate(test_data, val=False)
        self.log_results(test_results,
                         log_type = 'Testing'
                         )
        
        best['best_step'] = best_step
        for key, value in test_results.items():
            best['test {}'.format(key)] = value
        
        return best
