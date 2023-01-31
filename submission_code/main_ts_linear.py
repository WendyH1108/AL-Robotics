#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


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

import math
import seaborn as sns

import time

from data_generate import generateSourceData, generateTargetData, get_dataset, npy_loader
from the_model import findRepresentation_and_weight2, findTargetWeight, Net, test, LinearNet
from utils import *
from calculate_v import CalSparseSampleV
import random

# In[2]:


print(torch.__version__)
print(torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


# In[3]:

# two stage learning

parser = argparse.ArgumentParser(description='')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--target_label', type=int,
                    help='target one-hot encoded label: target_label vs. others')
parser.add_argument('--target_task', type=str,
                    help='type corruption class')
parser.add_argument('--num_iter', type=int,
                    help='number of iteration round')
parser.add_argument('--L', type=float, default = 3.0,
                    help='base length of each epoch')
parser.add_argument('--epoch_num', type=int,
                    help='number of epochs')
parser.add_argument('--print_out', action='store_false', help='print out the loss during the training')


# add
parser.add_argument('--sparsity', default = 50, type = int,
                    help='task selection sparsity')
parser.add_argument('--result_path', type=str, default = '',
                    help='type corruption class')


parser.add_argument('--default_epoch', type=int, default = 0,
                    help='number of epochs')
parser.add_argument('--N_lowerBound', type=int, default=60,
                    help='number of lower bounds for sample complexity')
parser.add_argument('--RescaleType', type=int, default=0,
                    help='0 upward. 1 downward')
parser.add_argument('--L1_REG', type=float, default=1e-10,
                    help='base length of each epoch')
parser.add_argument('--debug_num', type=int, default=0,
                    help='base length of each epoch')
parser.add_argument('--vAvg', type=int, default=0,
                    help='base length of each epoch')
parser.add_argument('--N_target', type=int, default=200,
                    help='base length of each epoch')
parser.add_argument('--rerun', type=int, default=0,
                    help='rerun the model')
parser.add_argument('--K', type=int, default=50,
                    help='model dimensional')
parser.add_argument('--is_L2N1', type=int, default=0,
                    help='use L2N1 or not')

# add
parser.add_argument('--is_linear', type=int, default=0,
                    help='0 --> conv, 1 --> linear net')
parser.add_argument('--num_target_epochs', type=int, default=400,
                    help='400 --> conv, ? -> linear net')

args = parser.parse_args()
print(args)

def set_seed(seed):
    print(f'set seed to {seed}')
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)

# In[4]:

# get basic information
numOfEpoch = args.epoch_num
numOfIter = args.num_iter
K = args.K
N_target = args.N_target;
L = args.L #base for the exponentially increase length

TestFolder = args.result_path + "TS_label{}_{}_vAvg{}".format(args.target_label,args.target_task,args.vAvg)
if args.seed != 0:
    TestFolder += f'_Seed{args.seed}'
if args.K != 50:
    TestFolder += f'_K{args.K}'
if args.sparsity != 50:
    TestFolder += f'_sparsity{args.sparsity}'
if args.N_lowerBound != 40:
    TestFolder += f'_LB{args.N_lowerBound}'
if args.is_L2N1 != 0:
    TestFolder += f'_is_L2N1'
if args.is_linear != 0:
    TestFolder += '_Linear'
if args.num_target_epochs != 400:
    TestFolder += f'_TargetEpochs{args.num_target_epochs}'

if not os.path.exists(TestFolder):
  # Create a new directory because it does not exist 
  os.makedirs(TestFolder)
  print("The new directory {} is created!".format(TestFolder))


# In[5]:

noise_classes = ["brightness","canny_edges","dotted_line","fog","glass_blur","identity","impulse_noise","motion_blur","rotate","scale","shear","shot_noise","spatter","stripe","translate","zigzag"]
source_noiseClasses_list = deepcopy(noise_classes)
source_noiseClasses_list.remove(args.target_task)
    
train_datasets, source_tasks, target_task, test_loader = get_dataset(TestFolder, noise_classes, args.target_task, args.target_label )

M = len(source_tasks)
print(f'M = {M}')



#set target train dataset
iterNum = 1
target_dataset = generateTargetData(target_task,N_target,train_datasets)


# In[8]:


# test_error_ada = np.zeros((numOfEpoch, numOfIter))
# test_error_nonada = np.zeros((numOfEpoch, numOfIter))
# num_of_sample = np.zeros((numOfEpoch, numOfIter))

test_error_ada_l2 = np.zeros((numOfEpoch, numOfIter))
test_error_ada_l1 = np.zeros((numOfEpoch, numOfIter))
test_error_nonada = np.zeros((numOfEpoch, numOfIter))
train_error_ada_l2 = np.zeros((numOfEpoch, numOfIter))
train_error_ada_l1 = np.zeros((numOfEpoch, numOfIter))
train_error_nonada = np.zeros((numOfEpoch, numOfIter))
num_of_sample_l2 = np.zeros((numOfEpoch, numOfIter))
num_of_sample_l1 = np.zeros((numOfEpoch, numOfIter))
num_of_sample_nonada = np.zeros((numOfEpoch, numOfIter))


# avg_nonmulti_loss = 0;
# avg_multi_loss = 0;
# avg_multi_correctness = 0;




def SampleAndTrain(numOfSourceSample_count, test_error, num_of_sample, test_loader, default_epoch, i, iteration, name = 'ada_l1'):
    set_seed(args.seed)

    N_dict, cur_datasets= generateSourceData(source_tasks, numOfSourceSample_count, train_datasets)
    # print(f"we plan number of samples for {name} as ",numOfSourceSample_count.view((int(M/10),10)));
    if  os.path.exists("{}/{}_{}_iter{}.pt".format(TestFolder,target_task,name,i)) and args.rerun == 0:
        print("***** loading checkpoint from {}/{}_{}_iter{}.pt *****".format(TestFolder,target_task,name,i))
        if args.is_linear == 0:
            model = Net(source_tasks, K).to(device)
        else:
            model = LinearNet(source_tasks, 28*28 ,K).to(device)

        model.load_state_dict(torch.load("{}/{}_{}_iter{}.pt".format(TestFolder,target_task,name,i)))
        v_avg_list_l1 = v_avg_list_l2 = None
        # if args.vAvg != 0:
        #     if  os.path.exists("{}/avg_v_{}_{}_iter{}.pt".format(TestFolder,target_task,name,i)):
        #         print(f'***** read avg from file ******')
        #         v_avg_list = torch.tensor(torch.load("{}/avg_v_{}_{}_iter{}.pt".format(TestFolder,target_task,name,i)))
    else: 
        print("***** start finding the shared representation and the non-shared weights on source task *****")
        lr = 1e-2*(0.75**(i-default_epoch));
#         acc=0.13*np.power(1.15,i-default_epoch-1)
        model,W_est,v_avg_list_l1, v_avg_list_l2 = findRepresentation_and_weight2(args, lr,args.debug_num,cur_datasets, target_dataset, source_tasks,N_dict,test_loader,gamma = 0.9, K = K, M = M, L1_REG=args.L1_REG, LType = name)

        print("***** start finding the non-shared weights on target task *****")
        model,w_est = findTargetWeight(args, lr,target_dataset,model,gamma=0.9)

        print("***** start saving *****")
        torch.save(model.state_dict(), "{}/{}_{}_iter{}.pt".format(TestFolder,target_task,name,i))
        if v_avg_list_l1 != None and name == 'ada_l1':
            torch.save(v_avg_list_l1, "{}/avg_v_{}_{}_iter{}.pt".format(TestFolder,target_task,name,i))
        if v_avg_list_l2 != None and name == 'ada_l2':
            torch.save(v_avg_list_l2, "{}/avg_v_{}_{}_iter{}.pt".format(TestFolder,target_task,name,i))

    print("***** start testing *****")
    num_correctPred = test(model, test_loader);
    test_error[i-default_epoch,iteration]=1-num_correctPred

    num_of_sample[i-default_epoch,iteration]=torch.sum(numOfSourceSample_count)

    return num_correctPred, v_avg_list_l1, v_avg_list_l2



fig1, axes = plt.subplots(1, numOfEpoch-1, figsize=(40,5))

for iteration in range(iterNum):    
    # torch.manual_seed(iteration)
    set_seed(args.seed)

    print("test on %s for the %d iteration"%(target_task,iteration) )
    
    default_epoch = int(math.log(K*M,L));
    if args.default_epoch != 0:
        default_epoch = args.default_epoch;
    

    # print(f'default_epoch = {default_epoch}')
    # w_est_ada_l2 = torch.FloatTensor(np.ones(shape = (K,1)));
    # W_est_ada = torch.FloatTensor(np.clip(np.random.normal(0.5,0.5,(K,M)),0,1));

    w_est_ada_l1 = torch.FloatTensor(np.ones(shape = (K,1)));
    W_est_ada_l1 = torch.FloatTensor(np.clip(np.random.normal(0.5,0.5,(K,M)),0,1));

    w_est_ada_l2 = deepcopy(w_est_ada_l1)#torch.FloatTensor(np.ones(shape = (K,1)));
    W_est_ada_l2 = deepcopy(W_est_ada_l1)#torch.FloatTensor(np.clip(np.random.normal(0.5,0.5,(K,M)),0,1));

    # numOfSourceSample_count = 0;
    numOfSourceSample_count_l1 = 0;
    numOfSourceSample_count_l2 = 0;

    v_avg_list_l2 = None; v_avg_list_l1 = None 

    FirstStageTotalNum = max(L**(default_epoch) * 1.0, args.N_lowerBound * M)

    for i in range(default_epoch,numOfEpoch+default_epoch):
        print("start epoch ", i)
        N_lowerBound = args.N_lowerBound #int(np.sqrt(L**i))
        basic_N_tot = FirstStageTotalNum * L**(i - default_epoch)
        
        is_ada = (i != default_epoch)
        print(f'basic_N_tot = {basic_N_tot}, is_ada = {is_ada}')
        v_bar_ada_l1, v_bar_ada_l2, v_bar_random = CalSparseSampleV(N_lowerBound, basic_N_tot, w_est_ada_l2,W_est_ada_l2, w_est_ada_l1,W_est_ada_l1, args.L1_REG, is_ada, M, v_avg_list_l1, v_avg_list_l2, args.sparsity, args.is_L2N1)
        v_avg_list_l2 = None; v_avg_list_l1 = None # clear

        ###################adaptive L2 ######################################################################################################

        numOfSourceSample_count_l2 = torch.round(v_bar_ada_l2 * basic_N_tot);

        print(f"epoch {i}: we plan number of samples for ada l2 as ",numOfSourceSample_count_l2.view((int(M/10),10)));
        
        num_correctPred_ada_l2, v_avg_list_l1, v_avg_list_l2 = SampleAndTrain(numOfSourceSample_count_l2, test_error_ada_l2, num_of_sample_l2, test_loader, default_epoch, i, iteration, name = 'ada_l2')
        
        ###################adaptive L1 ######################################################################################################
        if i == default_epoch:
            test_error_ada_l1[0, iteration] = 1-num_correctPred_ada_l2
            num_of_sample_l1[0, iteration] = num_of_sample_l2[0, iteration]
            # v_avg_list_l1 = v_avg_list_l2
            continue
        else:
            numOfSourceSample_count_l1 = torch.round(v_bar_ada_l1 * basic_N_tot);

            print(f"epoch {i}: we plan number of samples for ada l1 as ",numOfSourceSample_count_l1.view((int(M/10),10)));

            num_correctPred_l1, v_avg_list_l1, _ = SampleAndTrain(numOfSourceSample_count_l1, test_error_ada_l1, num_of_sample_l1, test_loader, default_epoch, i, iteration, name = 'ada_l1')
        
            ###################plot weight################################################ 
            # sns.heatmap(numOfSourceSample_count_l1.view((int(M/10),10)),xticklabels=[0,1,2,3,4,5,6,7,8,9],yticklabels=source_noiseClasses_list,ax=axes[i-default_epoch-1])
            # axes[i-default_epoch-1].set_title("epoch {}".format(i-default_epoch))

        ################### non ada ######################################################################################################
        if i == default_epoch:
            test_error_nonada[i-default_epoch,iteration]=1-num_correctPred_ada_l2
            num_of_sample_nonada[0,iteration] = num_of_sample_l2[0, iteration]
            continue
        else:
            numOfSourceSample_count_non_ada = torch.round(v_bar_random * basic_N_tot);
            # numOfSourceSample_count_non_ada = torch.ceil(torch.FloatTensor([torch.sum(numOfSourceSample_count_l2)/M]*M));
            print(f"epoch {i}: we plan number of samples for nonada as ",numOfSourceSample_count_non_ada.view((int(M/10),10)));
            num_correctPred_nonada, _, _ = SampleAndTrain(numOfSourceSample_count_non_ada, test_error_nonada, num_of_sample_nonada, test_loader, default_epoch, i, iteration, name = 'non_ada')
            # num_correctPred_nonada = num_correctPred_nonada.int()
   
        
plt.show()


# In[9]:


# fig1.savefig('results/plots/test1_{}_weight.eps'.format(target_task), format='eps',bbox_inches='tight')
# fig1.savefig('results/plots/test1_{}_weight.png'.format(target_task), format='png',bbox_inches='tight')


# final result
# printout
print(f'non: np.mean(num_of_sample_nonada[0:numOfEpoch],axis = 1)= {np.mean(num_of_sample_nonada[0:numOfEpoch],axis = 1)}')
print(f'L1:  np.mean(num_of_sample_l1[0:numOfEpoch],axis = 1) = {np.mean(num_of_sample_l1[0:numOfEpoch],axis = 1)}')
print(f'L2:  np.mean(num_of_sample_l2[0:numOfEpoch],axis = 1) = {np.mean(num_of_sample_l2[0:numOfEpoch],axis = 1)}')

print(f'non: np.sum(test_error_nonada[0:numOfEpoch],axis = 1) = {np.sum(test_error_nonada[0:numOfEpoch],axis = 1)}')
print(f'L1:  np.sum(test_error_ada_l1[0:numOfEpoch],axis = 1) = {np.sum(test_error_ada_l1[0:numOfEpoch],axis = 1)}')
print(f'L2:  np.sum(test_error_ada_l2[0:numOfEpoch],axis = 1) = {np.sum(test_error_ada_l2[0:numOfEpoch],axis = 1)}')


