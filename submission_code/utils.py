import numpy as np
import torch


def stats_values(targets, str = ''):
    if not isinstance(targets, np.ndarray):
        targets_npy = targets.numpy()
    else:
        targets_npy = targets
    mean = np.mean(targets_npy)
    sum = np.sum(targets_npy)
    min = np.min(targets_npy)
    max = np.max(targets_npy)
    std = np.std(targets_npy)
    n1 = np.linalg.norm(targets_npy,ord=1)
    n2 = np.linalg.norm(targets_npy,ord=2)
    sparsity = np.sum(np.where(targets_npy > mean, 1.0, 0.0))
    print(f'{str} stats: mean = {mean}, sum = {sum}, max = {max}, min = {min}, std = {std}, n1 = {n1}, n2 = {n2}, sparsity = {sparsity}')
    return mean, min, max, std, n1, n2

def L2Error(A, b, x):
    # print(f'A.shape = {A.shape}, b.shape = {b.shape}, x.shape = {x.shape}, x = {x}')
    return torch.norm(A@x - b, p=2)
