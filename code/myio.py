"""
Module with class io containing methods for importing and exporting data
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import gzip
import json
import pandas as pd
#import logging
#logger = logging.getLogger(__name__)

class QADataset(Dataset):
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
                 data_dir,      # name of the directory storing all tasks
                 task_names,    # list of task directories, should match 'tasks' keys
                 tokenizer,     # tokenizer to use
                 max_length,    # maximum number of tokens 
                 batch_size=32, # batch size for training
                 shuffle=True   # whether to shuffle train sampling
                 ):
        self.data_dir = data_dir
        self.task_names = task_names
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token = tokenizer.pad_token_id
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tasks = {
                'SQuAD'                  : None,
                'TriviaQA-web'           : None,
                'NewsQA'                 : None,
                'SearchQA'               : None,
                'HotpotQA'               : None,
                'NaturalQuestionsShort'  : None,
                'tester'                 : None,
                }


    def find_closest_indices(self, ans, obs, given_span_start):
        """
        Find answer span closest to given MRQA data. Returns single span
        because MRQA looks like it only has 1 span per answer
        """
        
        assert type(ans) == list and len(ans) > 0, 'ans should be a non-empty list'
        
        cont_len = len(obs)
        ans_len = len(ans)
        first = ans[0]
        candidates = []
        current = obs.index(first)
        min_diff = self.max_length
        
        # find all instances of span
        while current + ans_len < cont_len:
            if ans == obs[current: current + ans_len]:
                candidates.append([current, current+ans_len])
            
            # if there are no more instances, break
            try:
                current = obs.index(first, current + 1)
            except ValueError:
                break
        
        # find the candidate closest to original MRQA span by start position
        for candidate in candidates:
            diff = abs(candidate[0]-given_span_start)
            if diff < min_diff:
                min_diff = diff
                result = candidate
        
        return result
    
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
            data = entry[0]
            
            diff = max_batch_seq_len - len(entry[0])
            if (diff) > 0:
                data += [self.pad_token]*diff
            data_list.append(data)
            
        data_list = torch.tensor(data_list)
        
        return [data_list, label_list]
                
    
    def read_task(self, testing):        
        temp_task = {
                'train': None, # dataloader for training data
                'dev'  : None  # dataloader for validation data
                }
        
        for task in self.task_names:
            
            # for train and dev
            for use in temp_task.keys(): 
                
                # lists to store data and labels for a given task
                data = []
                labels = []
                
                # will remove later, but used for determing how to open file
                # set `testing` to false if loading real data
                if testing:
                    
                    # for testing
                    input_file = os.path.join(self.data_dir, use, task + r".jsonl")
                    with open(input_file, "r", encoding="utf-8-sig") as reader:
                        # skip header
                        content = reader.read().strip().split('\n')[1:]
                        input_data = [json.loads(line) for line in content]
                else:
                    # for real data
                    input_file = os.path.join(self.data_dir, use, task + r".jsonl.gz")
                    with gzip.GzipFile(input_file, 'r') as reader:
                        # skip header
                        content = reader.read().decode('utf-8').strip().split('\n')[1:]
                        input_data = [json.loads(line) for line in content]
                
                # for each context
                for i, entry in enumerate(input_data):
                    context = entry.get('context')
                    
                    # create tokenized entries
                    for qa in entry.get('qas'):
                        # tokenize observations to have uniform length
                        obs = self.tokenizer.encode(qa.get('question'), 
                                                    text_pair=context,
                                                    add_special_tokens=True,
                                                    max_length=self.max_length)
                        
                        # need to talk to TAs about how to get answer span
                        # looks like there should only be one per answer
                        # right now getting `closest` to original
                        q_len = len(self.tokenizer.encode(qa.get('question')))
                        
                        # get start and end indices for acceptable answers
                        # looks like potential to have more than 1 acceptable
                        temp_ans = []
                        for answer in qa.get('detected_answers'):
                            ans = self.tokenizer.encode(answer.get('text'))
                            
                            # rethink how to use q_len after TA meeting
                            span = self.find_closest_indices(ans, obs, answer.get('token_spans')[0][0]+q_len+2)
                            
                            assert len(span) > 0, "Answer is cut off for task: {}, use: {}, obs: {}".format(task, use, i)
                            temp_ans.append(span)
                        
                        # add observation and answers to lists
                        data.append(obs)
                        labels.append(temp_ans)
                    
                    # create a dataset object for the dataloader
                    dataset = QADataset(data, labels)
                    
                # create DataLoader object. Shuffle for training.
                if use == 'train':
                    dl = DataLoader(dataset=dataset,
                                    batch_size=self.batch_size,
                                    collate_fn=self.collate,
                                    shuffle=True)
                else:
                    dl = DataLoader(dataset=dataset,
                                    batch_size=self.batch_size,
                                    collate_fn=self.collate,
                                    shuffle=True)
                
                # add to task
                temp_task[use] = dl
                
            # add task to `self.tasks`
            self.tasks[task] = temp_task
    
    def export_results(self):
        """
        Export results of analysis
        """
