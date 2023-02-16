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
from torch.nn.utils import spectral_norm

def maybe_sn(m, use_sn):
    return spectral_norm(m) if use_sn else m
  
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, spectral_norms, num_task, datamodel = False, random = True):
        super(MLP, self).__init__()
        assert len(hidden_dims) == len(spectral_norms)
        layers = []
        for dim, use_sn in zip(hidden_dims, spectral_norms):
            x = nn.Linear(input_dim, dim)
            layers += [
                x,
                nn.ReLU(inplace=True),
            ]
            if not random:
                x.weight = nn.Parameter(torch.ones_like(x.weight))
            

            input_dim = dim
        layers += [nn.Linear(input_dim, num_task)]
        self.net = nn.Sequential(*layers)
        
        if not datamodel:
          self.last_layers = nn.ModuleList([])
          self.last_layers.append(nn.Linear(num_task, output_dim))
          self.target_reset()
          for idx in range(num_task):
              self.last_layers.append(nn.Linear(num_task, output_dim))
              nn.init.constant_(self.last_layers[idx+1].weight, 0)
              nn.init.constant_(self.last_layers[idx+1].bias, 0)
        
    def forward_shared(self, x):
        return self.net(torch.Tensor(x))
      
    def forward_nonshared(self, x, task_id):
        output = self.last_layers[task_id+1](torch.Tensor(x))
        return output
      
    def target_reset(self):
        nn.init.constant_(self.last_layers[0].weight, 0)
        nn.init.constant_(self.last_layers[0].bias, 0)
        
    def forward_nonshared_target(self, x):
        output = self.last_layers[0](torch.Tensor(x))
        return output
      
    def predict(self, input, true_y, task_id=None):
        representation = self.forward_shared(input)
        if task_id == None:
          prediction = self.forward_nonshared_target(representation)
        else:
          prediction = self.forward_nonshared(representation, task_id)
        mse = F.mse_loss(prediction.float(), torch.Tensor(true_y).float(),reduction = "sum").item()/input.shape[0]
        return mse
      
class Conv(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, spectral_norms, num_task, datamodel = False, random = True):
        super(Conv, self).__init__()
        assert len(hidden_dims) == len(spectral_norms)
        layers = []
        layers += [nn.Conv2d(1, 32, 3, 1)]
        layers += [nn.Conv2d(32, 64, 3, 1)]
        layers += [nn.Linear(9216, num_task)]
        self.net = nn.Sequential(*layers)
        
        if not datamodel:
          self.last_layers = nn.ModuleList([])
          self.last_layers.append(nn.Linear(num_task, output_dim))
          self.target_reset()
          for idx in range(num_task):
              self.last_layers.append(nn.Linear(num_task, output_dim))
              nn.init.constant_(self.last_layers[idx+1].weight, 0)
              nn.init.constant_(self.last_layers[idx+1].bias, 0)
        
    def forward_shared(self, x):
        return self.net(torch.Tensor(x))
      
    def forward_nonshared(self, x, task_id):
        output = self.last_layers[task_id+1](torch.Tensor(x))
        return output
      
    def target_reset(self):
        nn.init.constant_(self.last_layers[0].weight, 0)
        nn.init.constant_(self.last_layers[0].bias, 0)
        
    def forward_nonshared_target(self, x):
        output = self.last_layers[0](torch.Tensor(x))
        return output
      
    def predict(self, input, true_y, task_id=None):
        representation = self.forward_shared(input)
        if task_id == None:
          prediction = self.forward_nonshared_target(representation)
        else:
          prediction = self.forward_nonshared(representation, task_id)
        mse = F.mse_loss(prediction.float(), torch.Tensor(true_y).float(),reduction = "sum").item()/input.shape[0]
        return mse
def train(model, optimizer, criterion, train_x, train_y, epoch):
  loss_lst = []
  for i in range(epoch):
    optimizer.zero_grad()
    outputs = model(torch.Tensor(train_x))
    loss = criterion(outputs, torch.Tensor(train_y))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    loss_lst.append(loss.item())
  return model, loss_lst

def train_source(model, datasets, num_tasks, optimizer, epoch, need_print):
  model.train()
  total_Loss = 0
  optimizer.zero_grad()
  total_sample_size = 0
  loss_list = []
  optimizer.zero_grad()
  for task_id in range(num_tasks):
      data, target = datasets[0][task_id],torch.Tensor(datasets[1][task_id])
      representation = model.forward_shared(data)
      output = model.forward_nonshared(representation, task_id)
      loss = F.mse_loss(output.float(), target.float(),reduction = "sum")
      if not data.shape[0] == 0:
        total_Loss += loss.item()/data.shape[0]
      loss.backward()
      total_sample_size += data.shape[0]
  torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
  optimizer.step()   
  if need_print:
        print('Train Epoch: {} [total loss on source: {:.6f}]'.format(epoch,total_Loss/num_tasks))        
  return total_Loss/total_sample_size
  
def train_target(model, dataset, optimizer, epoch, need_print = True):
  model.train()
  total_Loss = 0
  optimizer.zero_grad()
  data, target = dataset[0],dataset[1]
  with torch.no_grad():
      representation = model.forward_shared(data)
  # model.target_reset() # reset target task # ok?
  output = model.forward_nonshared_target(representation)
  loss = F.mse_loss(output.float(), target.float(),reduction = "sum")
  total_Loss +=loss.item()
  loss.backward()
  optimizer.step()
  torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
  optimizer.step()   
  if need_print:
        print('Train Epoch: {} [total loss on target: {:.6f}]'.format(epoch,total_Loss))        
  return total_Loss/dataset[1].shape[0]


def test(model, test_loader, need_print = True):
    model.eval()
    with torch.no_grad():
      mse = model.predict(test_loader[0], test_loader[1])

    mse /= len(test_loader.dataset)
    if need_print:
        print('Test set: Average loss: {:.4f}'.format(mse)/test_loader[0].shape[0])
        
    return mse
  

def update_v(ws, best_w):
  new_v = np.linalg.lstsq(ws.detach().numpy().T, best_w.detach().numpy())[0]
  new_v = np.abs(new_v/np.linalg.norm(new_v,2))
  return new_v

def estimate(model, input, true_y, num_task, v):
  results = []
  for task_id in range(num_task):
    result = model.predict(input, true_y, task_id)
    results.append(result)
  best_w_idx = np.argmin(results)
  # print(ws)
  # print(best_w_idx)
  # best_w = ws[best_w_idx]
  ws = []
  for i, layer in enumerate(model.last_layers[1:]):
    ws.append(layer.weight.detach())
    if best_w_idx == i:
      best_w = layer.weight.detach()
  v = 0.8 * v + 0.2 * update_v(ws, best_w)
  # v = np.array([1]*len(RawData))
  # v = v / (len(RawData))
  v = np.apply_along_axis(lambda x: max(0.167, x), 0, [v])
  if not len(v.shape) == 1:
    v = v[0]
  return v, best_w

def findTargetWeight(lr,target_dataset,model,gamma=0.9, need_print = True):
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=gamma)
    model.target_reset()
    lowest = True
    epoch = 0
    prev_loss = np.inf
    # for epoch in range(1, num_target_epochs):
    while lowest:
        loss = train_target(model,target_dataset, optimizer, epoch, need_print)
        lowest = False if prev_loss < loss else True
        epoch += 1
        scheduler.step()    
    w_est_ada = torch.transpose(model.fc2[0].weight.cpu(),1,0)
    return model,w_est_ada
  
# if __name__=="main":
#   model = 