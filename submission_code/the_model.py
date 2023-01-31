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
from calculate_v import *
# from utils import *

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

class LinearNet(nn.Module):
    
    def __init__(self, tasks, input_dim, K=50):
        print(f'in Linear model K = {K}')
        super(LinearNet, self).__init__()
        
        # B_init = torch.zeros((input_dim, K))
        # B_init.normal_()
        self.B = nn.Linear(input_dim, K)

        self.fc2 = nn.ModuleList([]);
        self.fc2.append(nn.Linear(K, 1)) #Final one for the target task
         #Because here we are dealing with unbalanced dataset, so initilize to 0 is good
        self.target_reset()

        # count = 0;
        for idx, task in enumerate(tasks):
            self.fc2.append(nn.Linear(K, 1))
            self.fc2[idx+1].weight = Parameter(self.fc2[0].weight.detach()); # 0 initial
            self.fc2[idx+1].bias = Parameter(self.fc2[0].bias.detach());

    def target_reset(self):
        nn.init.constant_(self.fc2[0].weight, 0)
        nn.init.constant_(self.fc2[0].bias, 0)

    def forward_shared(self, x):
        # print(f'x.shape = {x.shape}') [*, 1, 28, 28]
        x = x.view(-1, 28*28)
        repr = self.B(x)
        return repr
    
    def forward_nonshared_source(self, x, task_id):
        output = self.fc2[task_id+1](x)
        return output

    def forward_nonshared_target(self, x, task_id):
        output = self.fc2[0](x)
        return output

class Net(nn.Module):
    
    def __init__(self, tasks, K=50):
        print(f'in CNN model K = {K}')
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, K)
        self.fc2 = nn.ModuleList([]);
        self.fc2.append(nn.Linear(K, 1)) #Final one for the target task
         #Because here we are dealing with unbalanced dataset, so initilize to 0 is good
        self.target_reset()
        # count = 0;
        for idx, task in enumerate(tasks):
            self.fc2.append(nn.Linear(K, 1))
            self.fc2[idx+1].weight = Parameter(self.fc2[0].weight.detach()); # 0 initial
            self.fc2[idx+1].bias = Parameter(self.fc2[0].bias.detach());

    def target_reset(self):
        nn.init.constant_(self.fc2[0].weight, 0)
        nn.init.constant_(self.fc2[0].bias, 0)

    def forward_shared(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        return x
    
    def forward_nonshared_source(self, x, task_id):
        output = self.fc2[task_id+1](x)
        return output
    def forward_nonshared_target(self, x, task_id):
        output = self.fc2[0](x)
        return output
    
def test(model, test_loader, need_print = True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            representation = model.forward_shared(data);
            output = model.forward_nonshared_target(representation,0)
            pred = output > 0.5;
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    if need_print:
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
    return correct / len(test_loader.dataset);
    
def trainOnSource(args, model, datasets, tasks, optimizer, epoch, N, test_loader, need_print):
    model.train()
    
#     tic = time.time()

    total_Loss = 0;
    optimizer.zero_grad()

    for task_id, task in enumerate(tasks):
        data, target = datasets[task][0],datasets[task][1]
        representation = model.forward_shared(data);
        output = model.forward_nonshared_source(representation,task_id)
        loss = F.mse_loss(output.float(), target.float(),reduction = "sum")
        total_Loss+=loss.item()
        loss.backward()
            
    optimizer.step()   
    
#     print('Train Epoch: {} [total time: {:.4f}]'.format(epoch,time.time()-tic)) 
        
    if args.print_out and need_print:
        # test(model, test_loader)
        print('Train Epoch: {} [total loss on source: {:.6f}]'.format(epoch,total_Loss/N['sum']))        
    return total_Loss/N['sum']



def trainOnTarget(args, model, dataset, optimizer, epoch, need_print = True):
    """
    Train the model on the target task in 1 step
    """
    model.train()
    total_Loss = 0;
    optimizer.zero_grad()
    data, target = dataset[0],dataset[1]
    with torch.no_grad():
        representation = model.forward_shared(data)
    # model.target_reset() # reset target task # ok?
    output = model.forward_nonshared_target(representation,0)
    loss = F.mse_loss(output.float(), target.float(),reduction = "sum")
    total_Loss +=loss.item()
    loss.backward()
    optimizer.step();

    min_print = min(100, max(args.num_target_epochs // 10, 1))

    if args.print_out and epoch%min_print==0 and need_print:
        print('Train Epoch: {} [total loss on target: {:.6f}]'.format(epoch,total_Loss/len(dataset[1])))
    return total_Loss


def cal_v_avg(args, lr, target_dataset, source_tasks, model, gamma, L1_REG, K, M, LType):
    W_est_ada = torch.zeros((K,M))
    for i in range(1,len(source_tasks)+1):
        W_est_ada[:,[i-1]] = torch.transpose(model.fc2[i].weight.cpu(),1,0)

    _,w_est_ada = findTargetWeight(args, lr, target_dataset, model, gamma, need_print = False)
    W_est_ada = W_est_ada.detach()
    w_est_ada = w_est_ada.detach()

    if LType == 'ada_l1':
        v_avg_tmp = CalL1( w_est_ada, W_est_ada, L1_REG, True, M)
    if LType == 'ada_l2':
        v_avg_tmp = CalL2( w_est_ada, W_est_ada, True, M)

    return v_avg_tmp

def cal_v_avg2(args, lr, target_dataset, source_tasks, model, gamma, L1_REG, K, M, LType):
    W_est_ada = torch.zeros((K,M))
    for i in range(1,len(source_tasks)+1):
        W_est_ada[:,[i-1]] = torch.transpose(model.fc2[i].weight.cpu(),1,0)

    _,w_est_ada = findTargetWeight(args, lr, target_dataset, model, gamma, need_print = False)
    W_est_ada = W_est_ada.detach()
    w_est_ada = w_est_ada.detach()

    v_avg_tmp_l1 = CalL1( w_est_ada, W_est_ada, L1_REG, True, M)
    v_avg_tmp_l2 = CalL2( w_est_ada, W_est_ada, True, M)

    # if LType == 'ada_l1':
    #     v_avg_tmp = CalL1( w_est_ada, W_est_ada, L1_REG, True, M)
    # if LType == 'ada_l2':
    #     v_avg_tmp = CalL2( w_est_ada, W_est_ada, True, M)

    # return v_avg_tmp
    return v_avg_tmp_l1, v_avg_tmp_l2


def findRepresentation_and_weight2(args,lr,numOfEpoch,cur_datasets, target_dataset, source_tasks,N_ada_dict, test_loader, gamma = 0.9, acc = 0, K=50, M=150, L1_REG=1e-10, LType = 'ada_l1'):
    if args.is_linear == 0:
        model = Net(source_tasks, K).to(device)
    else:
        model = LinearNet(source_tasks, 28*28 ,K).to(device)
        
    optimizer = optim.Adadelta(model.parameters(), lr=lr) # Adadelta
    scheduler = StepLR(optimizer, step_size=100, gamma=gamma)
    
    v_avg_tmp = None; 
    v_avg_list_l1 = []
    v_avg_list_l2 = []

    for epoch in range(0, numOfEpoch+1):
        loss = trainOnSource(args, model, cur_datasets, source_tasks, optimizer, epoch, N_ada_dict, test_loader, (epoch % (numOfEpoch // 10) == 0));
        # if loss < acc: break
        scheduler.step()
        if (args.vAvg > 0 and epoch > numOfEpoch - args.vAvg) and (LType != 'non_ada'):
            v_avg_tmp_l1, v_avg_tmp_l2 = cal_v_avg2(args, lr, target_dataset, source_tasks, model, gamma, L1_REG, K, M, LType)
            v_avg_list_l1.append(v_avg_tmp_l1.numpy())
            v_avg_list_l2.append(v_avg_tmp_l2.numpy())
  
        # if (args.vAvg == 0 and epoch == numOfEpoch):
    W_est_ada = torch.zeros((K,M))
    for i in range(1,len(source_tasks)+1):
        W_est_ada[:,[i-1]] = torch.transpose(model.fc2[i].weight.cpu(),1,0)
    
    if len(v_avg_list_l1) != 0:
        v_avg_list_l1 = np.array(v_avg_list_l1).squeeze()
        v_avg_list_l1 = torch.tensor(np.mean(v_avg_list_l1, axis = 0))
        v_avg_list_l1 = v_avg_list_l1 / torch.sum(v_avg_list_l1)
        print(f'v_avg_list_l1.shape = {v_avg_list_l1.shape}')
        stats_values(v_avg_list_l1, f'v_avg_{LType}')
    else:
        v_avg_list_l1 = None
    
    if len(v_avg_list_l2) != 0:
        v_avg_list_l2 = np.array(v_avg_list_l2).squeeze()
        v_avg_list_l2 = torch.tensor(np.mean(v_avg_list_l2, axis = 0))
        v_avg_list_l2 = v_avg_list_l2 / torch.sum(v_avg_list_l2)
        print(f'v_avg_list_l2.shape = {v_avg_list_l2.shape}')
        stats_values(v_avg_list_l2, f'v_avg_{LType}')
    else:
        v_avg_list_l2 = None
    # ipdb.set_trace()
    
    
    return model,W_est_ada,v_avg_list_l1, v_avg_list_l2

def findTargetWeight(args,lr,target_dataset,model,gamma=0.9, need_print = True):
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=gamma)
    model.target_reset()
    for epoch in range(1, args.num_target_epochs):
        loss = trainOnTarget(args,model,target_dataset, optimizer, epoch, need_print)
        scheduler.step()    
    w_est_ada = torch.transpose(model.fc2[0].weight.cpu(),1,0)
    return model,w_est_ada





