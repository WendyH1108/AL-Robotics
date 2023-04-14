from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from copy import deepcopy
import utils
import re
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.nn.parameter import Parameter
import time
import active_model

condition = {
  "nowind" : 0,
  "10wind" : 1.3,
  "20wind" : 2.5,
  "30wind" : 3.7,
  "35wind" : 4.2,
  "40wind" : 4.9,
  "50wind" : 6.1,
  "70wind" : 8.5
  }

class SyntheticData():
  def __init__(self, num_source_task, rep_dim) -> None:
    self.num_source_task = num_source_task
    self.rep_dim = rep_dim
    # generate random w as non-shared variable for source tasks
    random_w = np.random.uniform(low=0, high=1, size=(num_source_task, num_source_task))
    random_w = np.concatenate((random_w,np.zeros((1,num_source_task))),axis = 0 )
    self.B_matrics = np.random.random((rep_dim , num_source_task+1))
    self.Bw = np.dot(self.B_matrics,random_w)
    # target Bw is the on the same direction with source task 0
    self.B_matrics[:,self.B_matrics.shape[1]-1] = self.Bw[:, 0]
    # generate one-hot vector as target task
    w_target = np.zeros((num_source_task+1,1))
    w_target[w_target.shape[0]-1][0] = 1
    self.random_w = np.concatenate((random_w,w_target),axis = 1)
    # generate B_w * w
    self.Bw = np.dot(self.B_matrics,self.random_w)
    
  def update_bw(self, w):
    # change w for different directions to generate new data
    self.Bw = np.dot(self.B_matrics,w)

  def load_data(self,features = ['v', 'q', 'pwm']):
    dim_a = 3
    # features = ['v', 'q', 'pwm']
    label = 'fa'

    # Training data collected from the neural-fly drone
    dataset = 'neural-fly' 
    dataset_folder = 'data/training'

    # Training data collected from an intel aero drone

    modelname = f"{dataset}_dim-a-{dim_a}_{'-'.join(features)}" # 'intel-aero_fa-num-Tsp_v-q-pwm'
    RawData = utils.load_data(dataset_folder)
    # Data = utils.format_data(RawData, features=features, output=label)
    options = {}

    options['features'] = features
    options['dim_a'] = dim_a
    options['loss_type'] = 'crossentropy-loss'

    options['shuffle'] = True # True: shuffle trajectories to data points
    options['K_shot'] = 32 # number of K-shot for least square on a
    options['phi_shot'] = 256 # batch size for training phi

    options['alpha'] = 0.01 # adversarial regularization loss
    options['learning_rate'] = 5e-4
    options['frequency_h'] = 2 # how many times phi is updated between h updates, on average
    options['SN'] = 2. # maximum single layer spectral norm of phi
    options['gamma'] = 10. # max 2-norm of a
    options['num_epochs'] = 1000

    return RawData, options

  def sample_data(self,data, size):
    """randomly sample data

    Args:
        data (np array): data we want to sample from
        size (int): the number of data we want to sample

    Returns:
        list: selected indecies
    """
    number_of_rows = data.shape[0]
    random_indices = np.random.choice(number_of_rows, 
                                      size=size, 
                                      replace=True)
    return random_indices

 
  def extract_features(self,rawdata, features):
    """extract features from all sources tasks, which is x in algorithm

    Args:
        rawdata (dictionary): _description_
        features (list): the list of names of features that are shared around all sources tasks

    Returns:
        list: the list of data that only contians the selected shared features 
    """
    feature_data = []
    hover_pwm_ratio = 1.
    for feature in features:
      if isinstance(rawdata[feature], str):
        condition_list = re.findall(r'\d+', rawdata[feature]) 
        condition = 0 if condition_list == [] else float(condition_list[0])
        feature_data.append(np.tile(condition,(len(rawdata['v']),1)))
        continue
      feature_len = rawdata[feature].shape[1] if len(rawdata[feature].shape)>1 else 1
      if feature == 'pwm':
          feature_data.append(rawdata[feature] / 1000 * hover_pwm_ratio)
      else:
          feature_data.append(rawdata[feature].reshape(rawdata[feature].shape[0],feature_len))
    feature_data = np.hstack(feature_data)
    return feature_data


  def generate_task_sample(self,data_model, 
                          raw_task_data, 
                          shared_features, 
                          idx, 
                          sample_size, 
                          eps=True):
      """generate synthetic data for a task

      Args:
          data_model (MLP): the MLP model to generate data
          raw_task_data (dictionary): raw task data
          shared_features (list): shared features
          idx (int): the index of the task in source tasks list
          sample_size (int): sample size
          eps (bool, optional): if add noise. Defaults to True.

      Returns:
          np array: synthetic input data
          np array: true labels
      """
      
      shared_input = self.extract_features(raw_task_data,shared_features)
      random_indices = self.sample_data(shared_input, size = int(sample_size))
      # sample shared input
      sub_sample = shared_input[random_indices,:]
      shared_y = data_model.forward_shared(torch.Tensor(sub_sample)).detach().numpy()
      # generate non-shared features
      sub_features = self.Bw[:,idx].reshape((1,self.rep_dim))
      
      # generate labels = \phi_x(x) * B_w * w
      mul = np.dot(sub_features, shared_y.T)
      if eps: 
          y = mul + np.random.normal(0,0.01,mul.shape)
      else:
          y = mul
      return sub_sample, y.T
    
  def generate_sample(self, date_model, raw_data, options, sample_size, v, shared_features, eps=True):
    """ generate data for all tasks, mixed

    Args:
        date_model (MLP): the MLP data model
        raw_data (dictionary): dictionary contains all raw data
        options (list): list of information about the dataset
        sample_size (int): sample size
        v (list): sampling weights for tasks 
        shared_features (list): shared features
        eps (bool, optional): if add noises to data generation. Defaults to True.

    Returns:
        np array: input data
        np array: true label

    """
    sample = np.array([]).reshape(0,options["dim_x"])
    ys = np.array([]).reshape(0,1)
    for idx, vi in enumerate(v):
      data = raw_data[idx]
      teak_sample_size = int(sample_size*vi)
      sub_sample, y = self.generate_task_sample(date_model, data, shared_features, idx, teak_sample_size, eps)
      sample = np.vstack([sample, sub_sample])
      ys = np.vstack([ys, y])
    return sample, ys

  def generate_sample_H(self, data_model, raw_data, options, sample_size, v, shared_features, task_data = None, eps=True):
    """generate data for all tasks, group by each task

    Args:
        data_model (MLP): the MLP data model
        raw_data (dictionary): dictionary contains all raw data
        options (list): list of information about raw data
        sample_size (int): total sample size
        v (list): sampling weights of tasks 
        shared_features (list): list of shared features
        task_data (array, optional): if stack new data to old task_data. Defaults to None.
        eps (bool, optional): if add noises to dataset. Defaults to True.

    Returns:
        np array[array]: array of input data for each task
        np array[array]: array of true labels for each task
    """
    if task_data is None:
      task_data_x = {}
      task_data_y = {}
      for idx, vi in enumerate(v):
        task_data_x[idx] = np.array([]).reshape(0,options["dim_x"])
        task_data_y[idx] = np.array([]).reshape(0,1)
      task_data = [task_data_x, task_data_y]
    task_data_x, task_data_y = task_data
    for idx, vi in enumerate(v):
      data = raw_data[idx]
      task_sample_size = int(sample_size*vi)
      sub_sample, y = self.generate_task_sample(data_model, data, shared_features, idx, task_sample_size, eps)
      task_data_x[idx] = np.vstack([task_data_x[idx], sub_sample])
      task_data_y[idx] = np.vstack([task_data_y[idx], y])
    return task_data_x, task_data_y

  def get_dataset(self,data_model, raw_data, options, sample_size, shared_features, v):
    """generate dataset for training and testing

    Args:
        data_model (MLP): the MLP data model
        raw_data (dictionary): a data dictionary contains all raw task data
        options (list): the list of information about raw data
        sample_size (int): total sample size
        shared_features (list): the list of shared features
        v (list): sampling weights of tasks 

    Returns:
        tuple(array[array], array(array)): input data and labels for training
        tuple(array[array], array(array)): input data and labels for testing
    """
    target_idx = 0
    target_data_size = 100
    task_data_x, task_data_y = self.generate_sample_H(data_model, raw_data, options, sample_size, v, shared_features, None, True)
    test_data, test_label = self.generate_task_sample(data_model, raw_data[target_idx], shared_features,target_idx, target_data_size, False)
    return (task_data_x, task_data_y), (test_data, test_label)
