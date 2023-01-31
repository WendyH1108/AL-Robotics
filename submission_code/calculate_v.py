from email.policy import default
import numpy as np
import torch
from l1regls import l1regls
from cvxopt import normal, matrix
from utils import stats_values, L2Error

def CutNotTopk(vec, sparsity):
    if len(vec.shape) == 1:
        vec = vec.reshape(-1,1)
    vec = torch.tensor(vec, dtype = torch.float32)
    # print(f'vec = {vec}\n')
    vec_len = vec.shape[0]
    abs_vec = torch.abs(vec)
    sort_indices = abs_vec.sort(0, True)[1].squeeze()
    # print(f'type(vec) = {type(vec)}')
    vec[sort_indices[sparsity:]] = torch.zeros((vec_len - sparsity,1), dtype = torch.float32)
    # trunc_vec = torch.where(sort_indices >= (vec_len - sparsity), zero_th_float, vec)
    # print(f'vec = {vec}\nabs_vec = {abs_vec}\n')
    return vec


def CalL2(w_est_ada_l2, W_est_ada_l2, is_ada = True, M = 150, Norm = 2):
    if is_ada:
        # v_est_ada = torch.lstsq(w_est_ada_l2,W_est_ada_l2 ).solution;
        v_est_ada = torch.linalg.lstsq(W_est_ada_l2, w_est_ada_l2 ).solution;
        # v_est_ada = v_est_ada.reshape(M, -1)
        stats_values(v_est_ada, 'L2 minimization')
        # print(f'v_est_ada.shape = {v_est_ada.shape}')
        print(f'Error of L2 : {L2Error(W_est_ada_l2, w_est_ada_l2, v_est_ada)}, v_est_ada.shape = {v_est_ada.shape}')
        
        U,D_2,Vt = np.linalg.svd(np.array(W_est_ada_l2))
        condition_number_2 = np.max(D_2) / np.min(D_2)
        print(f'sigma_min(W_est_ada_l2) = {np.min(D_2)}, sigma_max(W_est_ada_l2) = {np.max(D_2)}, condition_number = {condition_number_2}')
        v_est_ada = torch.abs(v_est_ada)
        v_bar_ada_l2 = torch.pow(v_est_ada,Norm)
        v_bar_ada_l2 = v_bar_ada_l2/torch.sum(v_bar_ada_l2);
        # v_bar_ada = v_bar_ada/torch.norm(v_est_ada,p=2);
    else: # first epoch, v -> uniform
        v_bar_ada_l2 = (torch.ones(M)/M).unsqueeze(1);
    
    return v_bar_ada_l2


def CalSparseL2(w_est_ada_l2, W_est_ada_l2, is_ada = True, M = 150, sparsity = 50):
    v_bar_ada_l2_ori = CalL2(w_est_ada_l2, W_est_ada_l2, is_ada, M)
    if is_ada:
        v_bar_ada_l2_cut = CutNotTopk(v_bar_ada_l2_ori, sparsity)
    else:
        v_bar_ada_l2_cut = v_bar_ada_l2_ori
    return v_bar_ada_l2_cut

def CalL1( w_est_ada_l1, W_est_ada_l1, L1_REG=1e-10, is_ada = True, M = 150):
    if is_ada: # change to L1 solution !
        w_est_ada_m = matrix(np.array(w_est_ada_l1 /np.sqrt(L1_REG), dtype = np.float64)); 
        W_est_ada_m = matrix(np.array(W_est_ada_l1 /np.sqrt(L1_REG), dtype = np.float64)); # |Ax/\sqrt{a} - b/\sqrt{a}|^2 + |x|_1
        # v_est_ada_l2 = torch.lstsq(w_est_ada_l1,W_est_ada_l1 ).solution;
        v_est_ada_l2 = torch.linalg.lstsq(W_est_ada_l1, w_est_ada_l1 ).solution;
        
        print(f'Error of L2N1 : {L2Error(W_est_ada_l1, w_est_ada_l1, v_est_ada_l2)}, v_est_ada_l2.shape = {v_est_ada_l2.shape}')
        v_bar_ada_l2_n1 = torch.abs(v_est_ada_l2)
        v_bar_ada_l2_n1 = v_bar_ada_l2_n1 / torch.sum(v_bar_ada_l2_n1)

        # stats_values(np.fabs(w_est_ada_l1), 'fabs.w_est_ada_l1')
        # stats_values(np.fabs(np.array(w_est_ada_m)), 'fabs.w_est_ada_m')
        # stats_values(np.fabs(W_est_ada_l1), 'fabs.W_est_ada_l1')
        # stats_values(np.fabs(np.array(W_est_ada_m)), 'fabs.W_est_ada_m')

        U,D,Vt = np.linalg.svd(np.array(W_est_ada_l1))
        condition_number = np.max(D) / np.min(D)
        print(f'sigma_min(W_est_ada_l1) = {np.min(D)}, sigma_max(W_est_ada_l1) = {np.max(D)}, condition_number = {condition_number}')
        
        v_bar_ada_l1 = l1regls(W_est_ada_m, w_est_ada_m)
        print(f'Error of L1 : {L2Error(W_est_ada_l1, w_est_ada_l1, torch.tensor(np.array(v_bar_ada_l1), dtype=torch.float32))}')

        v_bar_ada_l1 = np.array(np.fabs(v_bar_ada_l1))
        stats_values(v_bar_ada_l1, 'L1 minimization')
        # v_threshold = np.where(np.fabs(v_bar_ada) < 1e-10, 0, v_bar_ada)
        v_bar_ada_l1 = torch.tensor(v_bar_ada_l1) # fabs sumup

        
        v_bar_ada_l1 = v_bar_ada_l1/torch.sum(v_bar_ada_l1) # magic
        # v_bar_ada = v_bar_ada/torch.sum(torch.pow(v_bar_ada, 2)); # sample rescale

    else: # first epoch, v -> uniform
        v_bar_ada_l1 = (torch.ones(M)/M).unsqueeze(1);
        v_bar_ada_l2_n1 = (torch.ones(M)/M).unsqueeze(1);
    
    return v_bar_ada_l1

def CalSparseL1( w_est_ada_l1, W_est_ada_l1, L1_REG=1e-10, is_ada = True, M = 150, sparsity = 50):
    v_bar_ada_l1_ori = CalL1( w_est_ada_l1, W_est_ada_l1, L1_REG, is_ada, M)
    if is_ada:
        v_bar_ada_l1_cut = CutNotTopk(v_bar_ada_l1_ori, sparsity)
    else:
        v_bar_ada_l1_cut = v_bar_ada_l1_ori
    return v_bar_ada_l1_cut



def RescaleSample( v_bar_ada_l1, v_bar_ada_l2, N_LB_l1,N_LB_l2, basic_N_tot):
    v_bar_ada_l1 = v_bar_ada_l1.to(torch.float32)
    v_bar_ada_l2 = v_bar_ada_l2.to(torch.float32)
    #### lower bound ratio ####
    LBR_l1 = torch.tensor(N_LB_l1 / basic_N_tot, dtype=torch.float32)
    LBR_l2 = torch.tensor(N_LB_l2 / basic_N_tot, dtype=torch.float32)
    zero_th_float = torch.tensor(0.0, dtype=torch.float32)
    # print(f'LBR_l1 = {LBR_l1}, LBR_l2 = {LBR_l2}')

    clip_v_l1 = torch.where(v_bar_ada_l1 <= LBR_l1, LBR_l1, v_bar_ada_l1) 
    clip_v_l2 = torch.where(v_bar_ada_l2 <= LBR_l2, LBR_l2, v_bar_ada_l2)
    L1 = torch.sum(torch.where(clip_v_l1 > LBR_l1, clip_v_l1, zero_th_float))
    S1 = torch.sum(torch.where(clip_v_l1 <= LBR_l1, clip_v_l1, zero_th_float))
    L2 = torch.sum(torch.where(clip_v_l2 > LBR_l2, clip_v_l2, zero_th_float))
    S2 = torch.sum(torch.where(clip_v_l2 <= LBR_l2, clip_v_l2, zero_th_float))

    sum_clip_v_l1 = torch.sum(clip_v_l1)
    sum_clip_v_l2 = torch.sum(clip_v_l2)
    if  sum_clip_v_l1 > sum_clip_v_l2: # L2 need larger
        # L2' + S2 = L1 + S1, L2' = a * L2 ===> a = (L1+S1-S2)/L2
        a2 = (L1+S1-S2)/L2
        # print(f'a2 = {a2}')
        clip_v_l2 = torch.where(clip_v_l2 <= LBR_l2, LBR_l2, clip_v_l2 * a2)
        
    elif sum_clip_v_l1 < sum_clip_v_l2: # L1 need larger
        # L1' + S1 = L2 + S2, L1' = a * L1 ===> a = (L2+S2-S1)/L1
        a1 = (L2+S2-S1)/L1
        clip_v_l1 = torch.where(clip_v_l1 <= LBR_l1, LBR_l1, clip_v_l1 * a1)
        # print(f'a1 = {a1}')
        
    stats_values(clip_v_l1, 'clip_v_l1')
    stats_values(clip_v_l2, 'clip_v_l2')
    return clip_v_l1, clip_v_l2

from copy import deepcopy
def CalSampleV(N_lowerBound, basic_N_tot, w_est_ada_l2,W_est_ada_l2, w_est_ada_l1, W_est_ada_l1, L1_REG=1e-10, is_ada = True, M = 150, v_avg_list_l1=None, v_avg_list_l2=None):
    if v_avg_list_l1 != None and v_avg_list_l2 != None:
        print(f'*** update from v avg ! ***')
        v_bar_ada_l2 = v_avg_list_l2
        v_bar_ada_l1 = v_avg_list_l1
        v_bar_ada_l1, v_bar_ada_l2 = RescaleSample(v_bar_ada_l1, v_bar_ada_l2, N_lowerBound, N_lowerBound, basic_N_tot)
    else:
        v_bar_ada_l2 = CalL2(w_est_ada_l2, W_est_ada_l2, is_ada, M)
        if is_ada:
            v_bar_ada_l1 = CalL1(w_est_ada_l1, W_est_ada_l1, L1_REG, is_ada, M)
            v_bar_ada_l1, v_bar_ada_l2 = RescaleSample(v_bar_ada_l1, v_bar_ada_l2, N_lowerBound, N_lowerBound, basic_N_tot)
        else:
            v_bar_ada_l1 = deepcopy(v_bar_ada_l2)

    return v_bar_ada_l1, v_bar_ada_l2

import random

def CalRandom(sum_value, LB, M, basic_N_tot, sparsity):
    """
    random select a vector that has the same sum value as 'sum_value'
    """
    rand_idx = random.sample(range(M), sparsity)
    not_rand_idx = list(set(range(M)) - set(rand_idx))
    LB_ratio = LB / basic_N_tot
    # print(f'rand_idx = {rand_idx}, not_rand_idx = {not_rand_idx}')
    random_vec = torch.zeros(M)
    # sparsity * x + (M - sparsity) * LB_ratio = sum_value
    random_vec[rand_idx] = (sum_value - (M - sparsity) * LB_ratio ) / sparsity
    random_vec[not_rand_idx] = LB_ratio
    # print(f'random_vec = {random_vec}, sum(random_vec) = {sum(random_vec)}')
    return random_vec



def CalSparseSampleV(N_lowerBound, basic_N_tot, w_est_ada_l2,W_est_ada_l2, w_est_ada_l1, W_est_ada_l1, L1_REG=1e-10, is_ada = True, M = 150, v_avg_list_l1=None, v_avg_list_l2=None, sparsity = 50, is_L2N1 = False):
    
    if v_avg_list_l1 != None and v_avg_list_l2 != None:
        print(f'*** update from v avg ! ***')
        if is_ada:
            v_bar_ada_l2 = CutNotTopk(v_avg_list_l2, sparsity)
            v_bar_ada_l1 = CutNotTopk(v_avg_list_l1, sparsity)
        else:
            v_bar_ada_l2 = v_avg_list_l2
            v_bar_ada_l1 = v_avg_list_l1
        
        v_bar_ada_l1, v_bar_ada_l2 = RescaleSample(v_bar_ada_l1, v_bar_ada_l2, N_lowerBound, N_lowerBound, basic_N_tot)
    else:
        v_bar_ada_l2 = CalSparseL2(w_est_ada_l2, W_est_ada_l2, is_ada, M, sparsity)
        if is_ada:
            if is_L2N1:
                v_bar_ada_l1 = CalL2(w_est_ada_l2, W_est_ada_l2, is_ada, M, 1)
            else:
                v_bar_ada_l1 = CalL1(w_est_ada_l1, W_est_ada_l1, L1_REG, is_ada, M)
            v_bar_ada_l1, v_bar_ada_l2 = RescaleSample(v_bar_ada_l1, v_bar_ada_l2, N_lowerBound, N_lowerBound, basic_N_tot)
        else:
            v_bar_ada_l1 = deepcopy(v_bar_ada_l2)
    
    random_v = CalRandom(torch.sum(v_bar_ada_l2), N_lowerBound, M, basic_N_tot, sparsity)
    stats_values(v_bar_ada_l1, 'v_bar_ada_l1')
    stats_values(v_bar_ada_l2, 'v_bar_ada_l2')
    stats_values(random_v, 'random_v')
    return v_bar_ada_l1, v_bar_ada_l2, random_v

if __name__ == '__main__':
    M = 10
    a = torch.randn(M,1)
    # CutNotTopk(a, M // 2)
    CalRandom(2 * M * 1, 1, M, M // 2)