import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
import numpy as np
import torch


datasets = {}

class DatasetOnMemory(Dataset):
    """
    A PyTorch dataset where all data lives on CPU memory.
    """

    def __init__(self, X, y, meta_data=None):
        assert len(X) == len(y), "X and y must have the same length."
        assert meta_data is None or len(X) == len(meta_data), "X and meta_data must have the same length."
        self.X = X
        self.y = y
        self.meta_data = meta_data 

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x = self.X[item]
        y = self.y[item]
        if self.meta_data is not None:
            meta_data = self.meta_data[item]
            return x, y, meta_data
        else:
            return x, y, None

    def get_inputs(self):
        return self.X

    def get_labels(self):
        return self.y

class MyDataset:
    def __init__(self, batch_size, num_workers):
        self.sampled_train_tasks = {} 
        self.sampled_val_tasks = {}
        self.sampled_test_tasks = {}
        self.label_sets = {}
        self.batch_size = batch_size
        self.num_workers = num_workers

    def generate_synthetic_data(self, task_dict, noise_var=None, seed=None, *others):
        pass

    def generate_val_data(self, budget = 200):
        """
        Generate a few shot validation data.
        """
        task_name_list = self.sampled_train_tasks.keys()
        val_task_dict = {}
        for task_name in task_name_list:
            if task_name+"_val" not in self.sampled_val_tasks:
                val_task_dict[task_name+"_val"] = (self.sampled_train_tasks[task_name], budget)
        self.generate_synthetic_data(val_task_dict, noise_var=0.0)

    def get_dataset(self, task_name, train=True, val=False, test=False):
        pass

    def get_sampled_train_tasks(self):
        """
        Get the sampled train tasks.
        :return: tasks.
        """
        return self.sampled_train_tasks
    
    def get_sampled_test_tasks(self):
        """
        Get the sampled test tasks.
        :return: tasks.
        """
        return self.sampled_test_tasks
    
    def get_sampled_val_tasks(self):
        """
        Get the sampled validation tasks.
        :return: tasks.
        """
        return self.sampled_val_tasks    
    
    # def delete_dataset(self, task_name):
    #     """
    #     Delete dataset for the task.
    #     :param task_name: name of the task.
    #     """
    #     if task_name in self.input_ind_sets:
    #         del self.input_ind_sets[task_name]
    #     if task_name in self.label_sets:
    #         del self.label_sets[task_name]
    #     if task_name in self.sampled_test_tasks:
    #         del self.sampled_test_tasks[task_name]
    #     if task_name in self.sampled_train_tasks:
    #         del self.sampled_train_tasks[task_name]