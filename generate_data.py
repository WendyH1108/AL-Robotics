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
num_task = 6
subspace = 2
# random_w=np.random.uniform(low=0, high=1, size=(1,num_task))
# for i in range(num_task):
#   mask = np.ones((1,num_task))
#   mask[0][i] = 0
#   new_w = np.random.uniform(low=0, high=1, size=(1,num_task))
#   new_w = mask * new_w
#   main_dirc = np.random.uniform(low=2, high=4)
#   new_w[0][i] = main_dirc
#   random_w = np.concatenate((random_w, new_w))
# random_w = random_w[1:]

# Continuous
# def generate_matrics(s_type):
  # if s_type == "best":
random_w = np.random.uniform(low=0, high=1, size=(6, 6))
random_w = np.concatenate((random_w,np.zeros((1,6))),axis = 0 )
B_matrics = np.random.random((2 , 7))
Bw = np.dot(B_matrics,random_w)
# target Bw is the on the same direction with source task 0
B_matrics[:,B_matrics.shape[1]-1] = Bw[:, 0]
w_target = np.zeros((7,1))
w_target[w_target.shape[0]-1][0] = 1
random_w = np.concatenate((random_w,w_target),axis = 1)
Bw = np.dot(B_matrics,random_w)
# if s_type == "uniform":
# B_matrics = np.random.random((2 , 7))
# random_w = np.indentity(7)

# norm 1, different direction
# w close to each other
# source finite, 
# discrete and continuous
# 1. finite & discrete, interpolate
# 2. continuous, w continuous, latent space

def load_data(features = ['v', 'q', 'pwm']):
  dim_a = 3
  # features = ['v', 'q', 'pwm']
  label = 'fa'

  # Training data collected from the neural-fly drone
  dataset = 'neural-fly' 
  dataset_folder = 'data/training'

  # # Training data collected from an intel aero drone
  # dataset = 'neural-fly-transfer'
  # dataset_folder = 'data/training-transfer'
  # hover_pwm = 910 # mean hover pwm for neural-fly drone
  # intel_hover_pwm = 1675 # mean hover pwm for intel-aero drone
  # hover_pwm_ratio = hover_pwm / intel_hover_pwm # scaling ratio from system id

  modelname = f"{dataset}_dim-a-{dim_a}_{'-'.join(features)}" # 'intel-aero_fa-num-Tsp_v-q-pwm'
  RawData = utils.load_data(dataset_folder)
  # Data = utils.format_data(RawData, features=features, output=label)
  options = {}
  # options['dim_x'] = Data[0].X.shape[1]
  # options['dim_y'] = Data[0].Y.shape[1]
  # options['num_c'] = len(Data)
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
  # print('dims of (x, y) are', (options['dim_x'], options['dim_y']))
  # print('there are ' + str(options['num_c']) + ' different conditions')
  return RawData, options

def sample_data(data, size):
  number_of_rows = data.shape[0]
  random_indices = np.random.choice(number_of_rows, 
                                    size=size, 
                                    replace=True)
  return random_indices


def extract_features(rawdata, features):
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
    # print(feature_data) 
  feature_data = np.hstack(feature_data)
  return feature_data


def generate_task_sample(data_model, 
                         raw_task_data, 
                         shared_features, 
                         idx, 
                         sample_size, 
                         eps=True, 
                         comb = False):
    shared_input = extract_features(raw_task_data,shared_features)
    random_indices = sample_data(shared_input, size = int(sample_size))
    sub_sample = shared_input[random_indices,:]
    shared_y = data_model.forward_shared(torch.Tensor(sub_sample)).detach().numpy()
    # sub_features = np.zeros((1,6))
    # sub_features[0][idx] = np.mean(raw_task_data["t"])
    # sub_features = np.tile([condition[raw_task_data["condition"]]],(1,6))
    # sub_features[0][idx] = condition[raw_task_data["condition"]]
    if comb:
      sub_features = 0.5* random_w[0].reshape((1,num_task)) + 0.5 * random_w[5].reshape((1,num_task))
    
    sub_features = np.dot(B_matrics, random_w[:,idx]).reshape((1,subspace))
    sub_features = Bw[:,idx].reshape((1,subspace))
    
    
    mul = np.dot(sub_features, shared_y.T)
    if eps: 
        y = mul + np.random.normal(0,0.01,mul.shape)
    else:
        y = mul
    return sub_sample, y.T
  
def generate_sample(NN, raw_data, options, sample_size, v, shared_features, eps=True):
  sample = np.array([]).reshape(0,options["dim_x"])
  ys = np.array([]).reshape(0,1)
  for idx, vi in enumerate(v):
    data = raw_data[idx]
    teak_sample_size = int(sample_size*vi)
    sub_sample, y = generate_task_sample(NN, data, shared_features, idx, teak_sample_size, eps)
    sample = np.vstack([sample, sub_sample])
    ys = np.vstack([ys, y])
  return sample, ys

def generate_sample_H(NN, raw_data, options, sample_size, v, shared_features, task_data = None, eps=True):
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
    sub_sample, y = generate_task_sample(NN, data, shared_features, idx, task_sample_size, eps)
    task_data_x[idx] = np.vstack([task_data_x[idx], sub_sample])
    task_data_y[idx] = np.vstack([task_data_y[idx], y])
  return task_data_x, task_data_y

def get_dataset(data_model, raw_data, options, sample_size, shared_features, v):
  target_idx = 0
  target_data_size = 100
  task_data_x, task_data_y = generate_sample_H(data_model, raw_data, options, sample_size, v, shared_features, None, True)
  test_data, test_label = generate_task_sample(data_model, raw_data[target_idx], shared_features,target_idx, target_data_size, False)
  return (task_data_x, task_data_y), (test_data, test_label)
