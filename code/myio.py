"""
Module with class io containing methods for importing and exporting data
"""

import logging as log
import math
from tqdm import tqdm
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import UICDataset

class IO:
    def __init__(self,
                 data_dir=None,                     # name of the directory storing all tasks
                 model_name='default',              # embedding model name
                 task_names=['reviews_UIC'],        # task name
                 tokenizer=None,                    # tokenizer to use
                 max_length=None,                   # maximum number of tokens
                 content=['reviewContent'],         # col name of review text
                 review_key='reviewContent',        # key for reviews
                 label_name='flagged',              # list of label col names
                 val_split=0.001,                   # percent of data for validation
                 test_split=0.001,                  # percent of data for test
                 batch_size=32,                     # batch size for training
                 shuffle=True,                      # whether to shuffle train sampling
                 cache=True,                        # whether to cache data if reading for first time
                 ):
        self.data_dir = data_dir
        self.model_name = model_name
        self.task_names = task_names
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token = tokenizer.pad_token_id
        self.content = content
        self.review_key = review_key
        self.label_name = label_name
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cache = cache

        self.tasks = {}
        
        for task in task_names:
            self.tasks[task] = None
        
        cache_dir_pre = os.path.join(data_dir,'cached')
        if not os.path.exists(cache_dir_pre):
            os.mkdir(cache_dir_pre)
        
        self.cache_dir = os.path.join(cache_dir_pre, model_name)
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
    """
    Object to store:
        Import/Output Methods
    
    Task data in dictionary 'tasks' as DataLoader objects keyed by file name.
    
    Data is returned as PyTorch tensors.
    
    Labels are returned as list of lists because MRQA can have multiple
        acceptable answers. The list is not uniform, so cannot convert to 
        tensor. At loss evaluation, should convert label to PyTorch tensor and 
        take best loss over all acceptable answers.
    
    """
        
    def collate(self, batch):
        """
        Customized function for DataLoader that dynamically pads batch so
        that all data have the same length. Needed for parallelization
        """
        
        review_list = []  # store padded sequences
        other_list = []   # other observational data
        label_list = []   # store labels
        max_batch_seq_len = min(self.max_length,
                                max(pd.DataFrame(batch).iloc[:,0].apply(len)))
                            # minimum of the longest sequence in batch
                            # and self.max_length
        
        # make batch tokens uniform length
        for entry in batch:
            other_list.append(entry[1])
            label_list.append(entry[2])
            data = entry[0] # first observation is always review
            
            diff = max_batch_seq_len - len(data)
            if (diff) > 0:
                data += [self.pad_token]*diff
            review_list.append(data)

        review_list = torch.tensor(review_list)
        other_list = torch.tensor(other_list)
        label_list = torch.tensor(label_list)

        return [review_list, other_list, label_list]
                
    
    def read_task(self):
        """
        Method to read in data for reviews task. Splits between train and
        validation with `self.split` percentage kept as validation.
        
        Data is saved in a dictionary of DataLoader objects for train and
        validation keyed by 'train' and 'dev' respectively.
        """
        log.info("="*40+" Reading Tasks "+"="*40)
        task_data = {
                'train': None, # dataloader for training data
                'dev'  : None, # dataloader for validation data
                'test' : None  # dataloader for test data
                }
        # print (self.task_names)
        for task in self.task_names:
            cache_file = os.path.join(self.cache_dir,
                                      "cached_{}_{}.pt".format(
                                          task,
                                          self.max_length))
            if os.path.exists(cache_file):
                log.info('Loading {} from cached file: {}'.format(
                    task, cache_file))
                loaded = torch.load(cache_file)
                train_set, val_set, test_set = (
                                                    loaded['train'],
                                                    loaded['dev'],
                                                    loaded['test']
                                                )
            else:
                train_set, val_set, test_set = self.read_from_csv(task)
                
                if self.cache:
                    log.info('Saving {} processed data into cached file: {}'.format(task, cache_file))
                    torch.save({'train' : train_set, 'dev' : val_set, 'test' : test_set}, cache_file)

            # create DataLoader object. Shuffle for training.
            task_data['train'] = DataLoader(dataset=train_set,
                                             batch_size=self.batch_size,
                                             collate_fn=self.collate,
                                             shuffle=True)

            task_data['dev'] = DataLoader(dataset=val_set,
                                             batch_size=self.batch_size,
                                             collate_fn=self.collate)

            task_data['test'] = DataLoader(dataset=test_set,
                                             batch_size=self.batch_size,
                                             collate_fn=self.collate)

            # add task to `self.tasks`
            self.tasks[task] = task_data

    def read_from_csv(self, file_name):
        # lists to store data and labels for a given task
        reviews = []
        other_data = []
        labels = []
        input_data_df = pd.read_csv(os.path.join(self.data_dir,"{}.csv".format(file_name)))

        # for each review
        skipped = []
        for i, entry in tqdm(input_data_df.iterrows(), desc = 'Data Loading'):
            
            try:
                for content_label in self.content:
                    others = []
                    if content_label == self.review_key:
                        review = self.tokenizer.encode(entry[content_label],
                                               add_special_tokens=True,
                                               max_length=self.max_length)
                    else:
                        others.append(entry[content_label])
    
                # add review and labels to lists
                if len(self.content)>1:
                    # if there are other statistics than just reviews
                    other_data.append(others)
                else:
                    # if only using reviews
                    other_data.append([False])
                
                reviews.append(review)
                labels.append(int(entry[self.label_name]))
            except:
                skipped.append(i)
        
        log.info("Skipped {} examples because of encoding errors. Indices: {}".format(
            len(skipped), skipped))
        
        # create a dataset object for the dataloader
        dataset = UICDataset.UICDataset(reviews, other_data, labels)

        # split to train and validation sets with `self.val_split` and `self.test_split`
        val_size = int(len(dataset) * self.val_split)
        test_size = int(len(dataset) * self.test_split)
        train_size = len(dataset) - val_size - test_size

        return torch.utils.data.random_split(dataset,
                                                [train_size, val_size, test_size])