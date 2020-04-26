"""
Module with class io containing methods for importing and exporting data
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
#from tqdm import tqdm
import pandas as pd
from zipfile import ZipFile
import io
import csv
import math
import logging as log
import constants, UICDataset

label_converter = {
        'rating':{'1':0,'2':1,'3':2,'4':3,'5':4},
        'flagged':{True:True, False:False}
        }

class IO:
    def __init__(self,
                 data_dir=None,                     # name of the directory storing all tasks
                 task_names=None,                   # task name                                                 Why tasks? from old code didn't want to go and change everything
                 tokenizer=None,                    # tokenizer to use
                 max_length=None,                   # maximum number of tokens
                 content=['reviewContent'],         # col name of review text                                   What is this supposed to be? List of all columns? yes
                 review_key='reviewContent',        # key for reviews
                 label_names=['flagged'],           # list of label col names
                 val_split=0.1,                     # percent of data for validation
                 test_split=0.1,                    # percent of data for test
                 batch_size=32,                     # batch size for training
                 shuffle=True,                      # whether to shuffle train sampling
                 cache=True,                        # whether to cache data if reading for first time
                 ):
        self.data_dir = data_dir
        self.task_names = task_names
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token = tokenizer.pad_token_id
        self.content = content
        self.review_key = review_key
        self.label_names = label_names
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cache = cache

        self.tasks = {
                'reviews_UIC' :None,
                'tester'      :None
                }

        self.cache_dir = os.path.join(data_dir, 'cached')
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
            print(task)
            cache_file = os.path.join(self.cache_dir,
                                      "cached_{}_{}.pt".format(
                                          task,
                                          self.max_length))
# =============================================================================
#             # this won't be activated if you comment out caching
# =============================================================================
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
                train_set, val_set, test_set = self.read_from_csv()
                
# =============================================================================
#                 # why did you remove caching? this is to save time for data loading
# =============================================================================
#                 if self.cache:
#                     log.info('Saving {} processed data into cached file: {}'.format(task, cache_file))
#                     torch.save({'train' : train_set, 'dev' : val_set, 'test' : test_set}, cache_file)
#
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
    
# =============================================================================
#     # can we keep the file name in this module intead of creating a new module just for the file name?
# =============================================================================
    def read_from_csv(self):
        # lists to store data and labels for a given task
        reviews = []
        other_data = []
        labels = []
        
        # why did you remove this and use pandas instead? 
        # with ZipFile(os.path.join(self.data_dir,task+r'.zip')) as zf:
        #     with zf.open(task+r'.csv','r') as file:
        #         reader = csv.reader(io.TextIOWrapper(file, 'utf-8'))
        #         header = [col_name for col_name in next(iter(reader))]
        #         input_data = [dict(zip(header, row)) for row in reader]
        #
        # input_data.pop(0)
        
# =============================================================================
#         # please use os.path.join instead of simple string concatenation to avoid errors
# =============================================================================
        input_data_df = pd.read_csv(self.data_dir + constants.UIC_DATASET_COMBINED_DEBUG)

        # for each review
        for i, entry in input_data_df.iterrows():

            observations = []
            review = self.tokenizer.encode(entry[self.review_key],
                                           add_special_tokens=True,
                                           max_length=self.max_length)

            # collect labels as list
            entry_labels = []
            for label_name in self.label_names:
                entry_labels.append(label_converter.get(label_name) \
                                    .get(entry[label_name]))

            other_data.append(entry.drop(self.review_key).drop(self.label_names).values.tolist())

            # for content_label in self.content:
            #     if content_label == self.review_key:
            #
            #     else:

            # add review and labels to lists
            reviews.append(review)
            labels.append(entry_labels)
        print(reviews)
        
# =============================================================================
#         # still need to fill out other_data
# =============================================================================
        # create a dataset object for the dataloader
        dataset = UICDataset.UICDataset(reviews, other_data, labels)

        # split to train and validation sets with `self.split`
        val_size = int(math.ceil(len(dataset) * self.val_split))
        test_size = int(math.ceil(len(dataset) * self.test_split))
        train_size = len(dataset) - val_size - test_size

        return torch.utils.data.random_split(dataset,
                                                [train_size, val_size, test_size])