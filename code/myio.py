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

label_converter = {
        'rating':{'1':0,'2':1,'3':2,'4':3,'5':4},
        'flagged':{'True':True, 'False':False}
        }

class FlaggedDataset(Dataset):
    """
    Custom Datset class to use Dataloader
    """
    def __init__(self, reviews, other_data, target):
        self.reviews = reviews
        self.other_data = other_data
        self.target = target
    
    def __getitem__(self, index):
        x = self.reviews[index]
        y = self.target[index]
        z = self.other_data[index]
        
        return [x, y, z]
    
    def __len__(self):
        return len(self.data)
    
class IO:
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
    def __init__(self,
                 data_dir=None,                     # name of the directory storing all tasks
                 model=None,                        # Huggingface model name
                 task_names=None,                   # task name
                 tokenizer=None,                    # tokenizer to use
                 max_length=None,                   # maximum number of tokens
                 content=['reviewContent'],         # col name of review text
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
        temp_task = {
                'train': None, # dataloader for training data
                'dev'  : None, # dataloader for validation data
                'test' : None  # dataloader for test data
                }
        
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
                # lists to store data and labels for a given task
                reviews = []
                other_data = []
                labels = []
                
                with ZipFile(os.path.join(self.data_dir,task+r'.zip')) as zf:
                    with zf.open(task+r'.csv','r') as file:
                        reader = csv.reader(io.TextIOWrapper(file, 'utf-8'))
                        header = [col_name for col_name in next(iter(reader))]
                        input_data = [dict(zip(header, row)) for row in reader]
                
                input_data.pop(0)
                    
                # for each review
                for i, entry in enumerate(input_data):
                    observations = []
                    for content_label in self.content:
                        if content_label == self.review_key:
                            review = self.tokenizer.encode(entry.get(self.content),
                                                        add_special_tokens=True,
                                                        max_length=self.max_length)
                        else:
# =============================================================================
#                             # TO DO: Fill other observations
# =============================================================================
                            obs = None
                            observations.append(obs)                       
                    
                    # collect labels as list
                    entry_labels = []
                    for label_name in self.label_names:
                        entry_labels.append(label_converter.get(label_name)\
                                            .get(entry.get(label_name)))
                        
                    # add review and labels to lists
                    reviews.append(review)
                    other_data.append(observations)
                    labels.append(entry_labels)
                    
                # create a dataset object for the dataloader
                dataset = FlaggedDataset(reviews, other_data, labels)
                
                # split to train and validation sets with `self.split`
                val_size = int(math.ceil(len(dataset)*self.val_split))
                test_size = int(math.ceil(len(dataset)*self.test_split))
                train_size = len(dataset)-val_size-test_size
                
                train_set, val_set, test_set = torch.utils.data.random_split(dataset,
                                                                             [train_size,
                                                                              val_size,
                                                                              test_size])
                
                if self.cache:
                    log.info('Saving {} processed data into cached file: {}'.format(task, cache_file))
                    torch.save({'train' : train_set, 'dev' : val_set, 'test' : test_set}, cache_file)
            
            # create DataLoader object. Shuffle for training.
            temp_task['train'] = DataLoader(dataset=train_set,
                     batch_size=self.batch_size,
                     collate_fn=self.collate,
                     shuffle=True)
            
            temp_task['dev'] = DataLoader(dataset=val_set,
                     batch_size=self.batch_size,
                     collate_fn=self.collate)
            
            temp_task['test'] = DataLoader(dataset=test_set,
                     batch_size=self.batch_size,
                     collate_fn=self.collate)
               
            # add task to `self.tasks`
            self.tasks[task] = temp_task