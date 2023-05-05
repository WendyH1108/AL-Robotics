import numpy as np
from strategies.strategy_skeleton import Strategy
from metrics.utils import *

def project_to_domain(x, domain):
    """Project the vector x to the unit cube.

    Args:
        x (np.array): the vector to be projected

    Returns:
        np.array: the projected vector
    """
    if domain == "ball":
        return x/np.linalg.norm(x)
    elif domain == "cube":
        return x/np.max(np.abs(x))
    else:
        raise ValueError("The domain should be either 'ball' or 'cube'.")
    


#TODO: Here I seed the exploration base change every epoch, but we could also seed it once and use the same exploration base for all epochs.
class MTALSampling(Strategy):
    strategy_name = "mtal"

    def __init__(self, target_task_dict, fixed_inner_epoch_num, mode="target_awared", domain = "ball", task_dim = None,  task_aug_kernel = None):
        super(MTALSampling, self).__init__(target_task_dict, fixed_inner_epoch_num, task_dim)
        self.mode = mode
        self.domain = domain
        print(f"MTALSampling mode: {self.mode}")
        self.task_aug_kernel = task_aug_kernel

        assert domain in ["ball", "cube"] or self.task_dim is not None, "If the domain is not ball or cube, then the task_dim should be specified."

    def select(self, model, budget, outer_epoch, inner_epoch, seed = 42, adjustable_budget_ratio=1.0):
        """Select a subset of the task to collect the data.

        Args:
            model (nn.Module): the task embedding model
            budget (int): the number of total samples to label (sum of all tasks)
            outer_epoch (int): the outer epoch number
            inner_epoch (int): the inner epoch number
        Returns:
            np.array: the indices of the selected samples
        """
        if outer_epoch == 0:
            # If at the initial epoch, we use the rough exploration phase to explore every direction of the task space.
            np.random.seed(seed)
            return self.rough_exploration_phase(model, int(budget)), True
        else:
            if self.mode == "target_awared":
                # If not at the initial epoch, we first use the fine exploration phase to explore the effective subspace of the task space.
                # Then we use the exploitation phase to sample from the source tasks that are close to the target task.

                # avoid exploit_len to be too small in the initial base
                fine_explore_len = min(int(budget**(3/4)*model.get_embed_dim()) * adjustable_budget_ratio, budget//2)
                exploitation_len = int(budget - fine_explore_len)
                if inner_epoch == 0:
                    return self.fine_exploration_phase(model, fine_explore_len, outer_epoch), False
                else:
                    return self.exploitation_phase(model, exploitation_len, outer_epoch, self.target_task_dict), True
            elif self.mode == "target_agnostic":
                return self.fine_exploration_phase(model, int(budget), outer_epoch), True
            else:
                raise ValueError("The mode should be either 'target_awared' or 'target_agnostic'.")

    def rough_exploration_phase(self, model, budget):
        # In the rough exploration phase, we explore each direction of the $task_dim - 1$-dimensional subspace of the task space.
        # Here we use the one-hot vector to represent the direction of the subspace, any other orthonormal vectors should also be fine.

        task_dim = model.get_output_dim() if self.task_dim is None else self.task_dim
        basis = np.eye(task_dim)
        basis[-1][-1] = 0
        task_dict = {}
        for i in range(task_dim - 1):
            task_dict[f"rough_explore_{i}"] = (basis[:, [i]], int(budget//(task_dim - 1)))
        return task_dict
    
    def fine_exploration_phase(self, model, budget, outer_epoch):
        # In the fine exploration phase, we aims to find the $embed_dim$-dimensional subspace of the task space that is the effective subspace of the task space.
        
        embed_matrix = model.get_restricted_task_embed_matrix()

        # When k ~= d_W, we use the rough exploration phase to explore every direction of the task space.
        # if embed_matrix.shape[0] >=  embed_matrix.shape[1]/2: TODO
        if embed_matrix.shape[0] >=  self.task_dim/2: 
            return self.rough_exploration_phase(model, budget)
        else:
            assert self.domain != "pendulum", "Pendulum domain is not supported in the fine exploration phase."
            _,_,vh = np.linalg.svd(embed_matrix, full_matrices=False)
            task_dict = {}
            for i, v in enumerate(vh):
                v = project_to_domain(np.expand_dims(v,1),self.domain)
                task_dict[f"fine_explore_epoch{outer_epoch}_{i}"] = (v, int(budget//len(vh)))
            return task_dict
    
    def exploitation_phase(self, model, exploitation_len, outer_epoch, target_task_dict):
        if self.domain == "pendulum":
            return self.exploitation_phase_ball(model, exploitation_len, outer_epoch, target_task_dict)
        elif self.domain == "ball" or self.domain == "cube":
            return self.exploitation_phase_general(model, exploitation_len, outer_epoch, target_task_dict)
        else:
            raise ValueError("The domain should be either 'pendulum', 'ball' or 'cube'.")

    def exploitation_phase_ball(self, model, budget, outer_epoch, target_task_dict):
        # In the exploitation phase, we focus on sample from sources tasks that are close to the target.
        # Here we deal with a benign domain (a ball)
        task_dict = {}
        counter = 0
        for _, (target_vector, _) in target_task_dict.items():
            embed_matrx = model.get_full_task_embed_matrix()
            embed_restrict_matrx = model.get_restricted_task_embed_matrix()
            # TODO : might want to add r cond here when target is not single
            v = np.linalg.lstsq(embed_restrict_matrx, embed_matrx @ target_vector, rcond=None)[0]
            v = project_to_domain(v, self.domain)
            task_dict[f"exploit_epoch{outer_epoch}_{counter}"] = (v, int(budget//len(target_task_dict))) #TODO: multiply with np.linalg.norm(v)?
            counter += 1
        return task_dict
    
    def exploitation_phase_general(self, model, budget, outer_epoch, target_task_dict):
        # In the exploitation phase, we focus on sample from sources tasks that are close to the target.
        # Here we use the sampling method to deal with a benign domain

        assert self.task_aug_kernel is not None, "The task_aug_kernel should be provided when the domain is not benign."

        task_dict = {}
        sample_num = 10**self.task_dim
        tmp = np.random.uniform(-1,1, self.task_dim)
        tmp_aug = self.task_aug_kernel(tmp)

        est_input_embed_matrix = model.get_input_embed_matrix()
        est_input_embed_matrix, s, vh = np.linalg.svd(est_input_embed_matrix, full_matrices=False)
        embed_matrix = model.get_full_task_embed_matrix()
        embed_restrict_matrix = model.get_restricted_task_embed_matrix()
        embed_matrix = np.diag(s) @ vh @ embed_matrix
        embed_restrict_matrix = np.diag(s) @ vh @ embed_restrict_matrix
        for _, (target_vector, _) in target_task_dict.items():
            target_vector = self.task_aug_kernel(target_vector.T)
            diff = 1
            iter = 0
            target_vector = self.task_aug_kernel(target_vector.T)
            v = 0
            while diff > 0.05 and iter < 6:
                np.random.seed()
                tmp = np.random.uniform(-1,1, (sample_num, self.task_dim ))*0.5**iter + v
                tmp[:,-1] = 0
                tmp_aug = self.task_aug_kernel(tmp)
                best_ind = np.linalg.norm(embed_matrix @ tmp_aug.T - embed_restrict_matrix @ target_vector.T, axis=0).argmin()
                v = tmp[best_ind]
                diff = np.linalg.norm(embed_matrix @ tmp_aug[[best_ind]].T - embed_restrict_matrix @ target_vector.T, axis=0)
                print(diff)
                iter += 1
            v = project_to_domain(v, "ball")
            task_dict[f"exploit_epoch{outer_epoch}_{counter}"] = (v, int(budget//len(target_task_dict))) #TODO: multiply with np.linalg.norm(v)?
        counter += 1
        return task_dict
    
class MTALSampling_TaskSparse(MTALSampling):
    strategy_name = "mtal_sparse"

    def __init__(self, target_task_dict, fixed_inner_epoch_num, mode="target_awared", domain = "ball", task_dim = None,task_aug_kernel = None):
        super(MTALSampling_TaskSparse, self).__init__(target_task_dict, fixed_inner_epoch_num, mode, domain, task_dim,task_aug_kernel)
        print(f"MTALSampling mode: {self.mode}")
        self.fine_exploration_vh = None

    def fine_exploration_phase(self, model, budget, outer_epoch):
        # In the fine exploration phase, we aims to find the $embed_dim$-dimensional subspace of the task space that is the effective subspace of the task space.
        embed_matrix = model.get_restricted_task_embed_matrix()
        # When k ~= d_W, we use the rough exploration phase to explore every direction of the task space.
        # if embed_matrix.shape[0] >=  embed_matrix.shape[1]/2: TODO
        if embed_matrix.shape[0] >=  self.task_dim/2:
            print("Use rough exploration phase for fine exploration phase.")
            return self.rough_exploration_phase(model, budget)
        else:
            assert self.domain != "pendulum", "Pendulum domain is not supported in the fine exploration phase."
            _,_,vh = np.linalg.svd(embed_matrix, full_matrices=False)

            if self.fine_exploration_vh is None:
                self.fine_exploration_vh = vh
            elif rowspace_dist(vh, self.fine_exploration_vh, metric="avg") < 0.8:
                print("Change exploration base", np.linalg.norm(self.fine_exploration_vh @vh.T, 'fro')**2/len(vh))
                self.fine_exploration_vh = vh

            task_dict = {}
            for i, v in enumerate(self.fine_exploration_vh):
                v = project_to_domain(np.expand_dims(v,1),self.domain)
                task_dict[f"fine_explore_epoch{outer_epoch}_{i}"] = (v, int(budget//len(self.fine_exploration_vh)))
            return task_dict
