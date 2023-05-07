import numpy as np
from strategies.strategy_skeleton import Strategy
from metrics.utils import compute_matrix_spectrum

#TODO: Here I seed the exploration base change every epoch, but we could also seed it once and use the same exploration base for all epochs.
class RandomSampling(Strategy):
    strategy_name = "random"

    def __init__(self, target_task_dict, fixed_inner_epoch_num, task_dim, task_aug_dim=None, task_aug_kernel=None):
         super(RandomSampling, self).__init__(target_task_dict, fixed_inner_epoch_num, task_dim, task_aug_dim, task_aug_kernel)

    def select(self, model, budget, outer_epoch, inner_epoch, seed=None, adjustable_budget_ratio=1.0):
        """Select a subset of the task to collect the data.

        Args:
            model (nn.Module): the task embedding model
            budget (int): the number of samples to label)

        Returns:
            np.array: the indices of the selected samples
        """
        # Currently we assume we know that only the first two dim is non-linear.
        non_linear_dim = 2 if self.task_dim < self.task_aug_dim else 0
        linear_dim = self.task_dim - non_linear_dim -1
        n_basis = linear_dim + 5**non_linear_dim
        
        # basis= np.zeros((n_basis, self.task_dim))
        # # For linear dim, we use orthonormal basis.
        # gaus = np.random.normal(0, 1, (linear_dim, linear_dim))
        # _, _, tmp = np.linalg.svd(gaus)
        # basis[0:linear_dim, non_linear_dim:-1] = tmp
        # # For non-linear dim, we use uniform basis.
        # basis[linear_dim:, :non_linear_dim] = np.random.uniform(-1, 1, (n_basis - linear_dim, non_linear_dim)) 

        basis= np.random.uniform(-1, 1, (n_basis, self.task_dim)) 
        basis[:, -1] = 0

        print(f"The spectrum of the basis is {compute_matrix_spectrum(basis, self.task_aug_kernel)}")

        # Format that as a dictionary.
        task_dict = {}
        for i, v in enumerate(basis):
           v = np.expand_dims(v,1)
           task_dict[f"random_base_epoch{outer_epoch}_{i}"] = (v, int(budget//len(basis)))

        return task_dict, inner_epoch == self.inner_epoch_num - 1 if self.inner_epoch_num is not None else True
    
    # def __generate_orthonormal_basis(self, model, budget, outer_epoch, inner_epoch, seed=None, *others):
    #     if seed is not None: np.random.seed(seed)
    #     # The (task_dim - 1, task_dim) orthonormal basis of the source task space
    #     orth = np.zeros((self.task_dim - 1, self.task_dim))
    #     gaus = np.random.normal(0, 1, (self.task_dim-1, self.task_dim-1))
    #     _, _, tmp = np.linalg.svd(gaus)
    #     orth[:, :self.task_dim-1] = tmp

    #     # Format that as a dictionary.
    #     task_dict = {}
    #     for i, v in enumerate(orth):
    #        v = np.expand_dims(v,1)
    #        task_dict[f"random_base_epoch{outer_epoch}_{i}"] = (v, int(budget//len(orth)))

        return task_dict, inner_epoch == self.inner_epoch_num - 1 if self.inner_epoch_num is not None else True
    
        
    

class FixBaseSampling(RandomSampling):
    strategy_name = "fix_base"

    def __init__(self, target_task_dict, fixed_inner_epoch_num,task_dim, task_aug_dim=None, task_aug_kernel=None):
         super(FixBaseSampling, self).__init__(target_task_dict, fixed_inner_epoch_num, task_dim, task_aug_dim, task_aug_kernel)
         self.fixed_basis = None

    def select(self, model, budget, outer_epoch, inner_epoch, seed=None, adjustable_budget_ratio=1.0):
        """Select a subset of the task to collect the data.

        Args:
            model (nn.Module): the task embedding model
            budget (int): the number of samples to label)

        Returns:
            np.array: the indices of the selected samples
        """

        if self.fixed_basis is None:    
            # Currently we assume we know that only the first two dim is non-linear.
            non_linear_dim = 2 if self.task_dim < self.task_aug_dim else 0
            linear_dim = self.task_dim - non_linear_dim -1
            n_basis = linear_dim + 5**non_linear_dim
            
            # basis= np.zeros((n_basis, self.task_dim))
            # # For linear dim, we use orthonormal basis.
            # gaus = np.random.normal(0, 1, (linear_dim, linear_dim))
            # _, _, tmp = np.linalg.svd(gaus)
            # basis[0:linear_dim, non_linear_dim:-1] = tmp
            # # For non-linear dim, we use uniform basis.
            # basis[linear_dim:, :non_linear_dim] = np.random.uniform(-1, 1, (n_basis - linear_dim, non_linear_dim)) 

            basis= np.random.uniform(-1, 1, (n_basis, self.task_dim)) 
            basis[:, -1] = 0

            print(f"The spectrum of the basis is {compute_matrix_spectrum(basis, self.task_aug_kernel)}")

            # Format that as a dictionary.
            task_dict = {}
            for i, v in enumerate(basis):
                v = np.expand_dims(v,1)
                task_dict[f"random_base_epoch{outer_epoch}_{i}"] = (v, int(budget//len(basis)))

            self.fixed_basis = basis
            return task_dict, inner_epoch == self.inner_epoch_num - 1 if self.inner_epoch_num is not None else True
        else:
            task_dict = {}
            for i, v in enumerate(self.fixed_basis):
                v = np.expand_dims(v,1)
                task_dict[f"random_base_epoch{outer_epoch}_{i}"] = (v, int(budget//len(self.fixed_basis)))
            return task_dict, inner_epoch == self.inner_epoch_num - 1 if self.inner_epoch_num is not None else True