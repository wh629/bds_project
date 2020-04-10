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
    def __init__(self, data, target):
        self.data = data
        self.target = target
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        return [x, y]
    
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
                 data_dir,                          # name of the directory storing all tasks
                 task_names,                        # task name
                 tokenizer,                         # tokenizer to use
                 max_length,                        # maximum number of tokens
                 content='reviewContent',           # col name of review text
                 label_names=['rating', 'flagged'], # list of label col names
                 val_split=0.1,                     # percent of data for validation
                 test_split=0.1,                    # percent of data for test
                 batch_size=32,                     # batch size for training
                 shuffle=True                       # whether to shuffle train sampling
                 ):
        self.data_dir = data_dir
        self.task_names = task_names
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token = tokenizer.pad_token_id
        self.content = content
        self.label_names = label_names
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tasks = {
                'reviews_UIC' :None,
                'tester'      :None
                }
        
    def collate(self, batch):
        """
        Customized function for DataLoader that dynamically pads batch so
        that all data have the same length. Needed for parallelization
        """
        
        data_list = []  # store padded sequences
        label_list = [] # store labels
        max_batch_seq_len = min(self.max_length, 
                                max(pd.DataFrame(batch).iloc[:,0].apply(len)))
                            # minimum of the longest sequence in batch
                            # and self.max_length
        
        # make batch data uniform length
        for entry in batch:
            label_list.append(entry[1])
            assert type(entry[1])== list, "labels not list"
            data = entry[0]
            
            diff = max_batch_seq_len - len(entry[0])
            if (diff) > 0:
                data += [self.pad_token]*diff
            data_list.append(data)
            
        data_list = torch.tensor(data_list)
        label_list = torch.tensor(label_list)
        
        return [data_list, label_list]
                
    
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
            # lists to store data and labels for a given task
            data = []
            labels = []
            
            with ZipFile(os.path.join(self.data_dir,task+r'.zip')) as zf:
                with zf.open(task+r'.csv','r') as file:
                    reader = csv.reader(io.TextIOWrapper(file, 'utf-8'))
                    header = [col_name for col_name in next(iter(reader))]
                    input_data = [dict(zip(header, row)) for row in reader]
            
            input_data.pop(0)
                
            # for each review
            for i, entry in enumerate(input_data):
                obs = self.tokenizer.encode(entry.get(self.content),
                                            add_special_tokens=True,
                                            max_length=self.max_length)
                
                # collect labels as list
                entry_labels = []
                for label_name in self.label_names:
                    entry_labels.append(label_converter.get(label_name)\
                                        .get(entry.get(label_name)))
                    
                # add review and labels to lists
                data.append(obs)
                labels.append(entry_labels)
                
            # create a dataset object for the dataloader
            dataset = FlaggedDataset(data, labels)
            
            # split to train and validation sets with `self.split`
            val_size = int(math.ceil(len(dataset)*self.val_split))
            test_size = int(math.ceil(len(dataset)*self.test_split))
            train_size = len(dataset)-val_size-test_size
            
            train_set, val_set, test_set = torch.utils.data.random_split(dataset,
                                                                         [train_size,
                                                                          val_size,
                                                                          test_size])
                
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
    
    def export_results(self):
        """
        Export results of analysis
        """
