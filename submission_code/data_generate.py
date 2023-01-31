from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from copy import deepcopy

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


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def npy_loader(path,type = "feature"):
    sample = torch.from_numpy(np.load(path));
    if type == "feature": 
        sample = sample.permute(0,3,1,2);
        return sample.type(torch.FloatTensor)
    return sample


def generateSourceData(source_tasks, N_ada, train_datasets):
    N_ada_dict = dict(zip(source_tasks, N_ada));
    N_ada_dict["max"] = torch.max(N_ada).item()
    N_ada_dict["sum"] = torch.sum(N_ada).item()  
    
    cur_datasets = {}
    for task_id, task in enumerate(source_tasks):
        data,target = train_datasets[task][0][:int(N_ada_dict[task].item())],train_datasets[task][1][:int(N_ada_dict[task].item())]
        data, target = data.to(device), target.to(device)
        cur_datasets[task] = (data, target)
    
    return N_ada_dict,cur_datasets

def generateTargetData(target_task,N_target,train_datasets):
    data,target = train_datasets[target_task][0][:N_target],train_datasets[target_task][1][:N_target]
    data, target = data.to(device), target.to(device)
    target_dataset = (data, target)
    
    return target_dataset

def get_dataset(TestFolder, noise_classes, noise_class_target, label_target, path = ''):
    # noise_classes = ["brightness","canny_edges","dotted_line","fog","glass_blur","identity","impulse_noise","motion_blur","rotate","scale","shear","shot_noise","spatter","stripe","translate","zigzag"]
    
    tasks = [];
    train_datasets = {}

    filename = "{}/shuffle_idx.pt".format(TestFolder)
    if os.path.exists(filename):
        print(' loading train data INDEX from existing checkpoint...'+ filename)   
        shuffle_idx = torch.load(filename)
    else:
        shuffle_idx = None
    for noise_class in noise_classes:
    #     train_data = npy_loader("../../testOnIminist/mnist_c/%s/train_images.npy"%(noise_class));
    #     train_label = npy_loader("../../testOnIminist/mnist_c/%s/train_labels.npy"%(noise_class),"label");
        train_data = npy_loader(path + "/mnist_c/%s/train_images.npy"%(noise_class));
        train_label = npy_loader(path + "/mnist_c/%s/train_labels.npy"%(noise_class),"label");
        if shuffle_idx is None:
            shuffle_idx = np.random.permutation(range(len(train_label)))
            print('saving the shuffled traning data as ' + filename)
            torch.save(shuffle_idx, filename)
        train_label = F.one_hot(train_label);
        for label in range(10):           
            name = "{}_{}".format(noise_class,label);
            tasks.append(name);
            train_datasets[name] = (train_data,train_label[:,[label]])

    #load the test 
    target_task = '{}_{}'.format(noise_class_target,label_target);
    # test_data = npy_loader("../../testOnIminist/mnist_c/%s/test_images.npy"%(noise_class_target));
    # test_label = npy_loader("../../testOnIminist/mnist_c/%s/test_labels.npy"%(noise_class_target),"label");
    test_data = npy_loader(path + "/mnist_c/%s/test_images.npy"%(noise_class_target));
    test_label = npy_loader(path + "/mnist_c/%s/test_labels.npy"%(noise_class_target),"label");
    test_label = F.one_hot(test_label);
    test_label = test_label[:,[label_target]]
    test_dataset = torch.utils.data.TensorDataset(test_data,test_label);
    test_loader = torch.utils.data.DataLoader(test_dataset)

    source_tasks = deepcopy(tasks);
    for label in range(10):
        source_tasks.remove("{}_{}".format(noise_class_target,label))

    return train_datasets, source_tasks, target_task, test_loader