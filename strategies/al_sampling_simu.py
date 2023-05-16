import numpy as np
from strategies.strategy_skeleton import Strategy
from metrics.utils import *


#TODO: Here I seed the exploration base change every epoch, but we could also seed it once and use the same exploration base for all epochs.
class MTALSampling(Strategy):
    strategy_name = "mtal"

    def __init__(self, target_task_dict, fixed_inner_epoch_num, mode="target_awared"):
        super(MTALSampling, self).__init__(target_task_dict, fixed_inner_epoch_num)
        self.mode = mode
        print(f"MTALSampling mode: {self.mode}")

    def select(self, task_list, model, budget, outer_epoch, inner_epoch, seed = 42):
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
            return self.rough_exploration_phase(task_list, model, int(budget)), True
        else:
            if self.mode == "target_awared":
                # If not at the initial epoch, we first use the fine exploration phase to explore the effective subspace of the task space.
                # Then we use the exploitation phase to sample from the source tasks that are close to the target task.
                if inner_epoch == 0:
                    return self.fine_exploration_phase(task_list, model, int(budget**(3/4)*model.get_embed_dim()), outer_epoch), False
                else:
                    return self.exploitation_phase(task_list, model, int(budget - budget**(3/4)*model.get_embed_dim()), outer_epoch, self.target_task_dict), True
            elif self.mode == "target_agnostic":
                return self.fine_exploration_phase(task_list, model, int(budget), outer_epoch), True
            else:
                raise ValueError("The mode should be either 'target_awared' or 'target_agnostic'.")

    def rough_exploration_phase(self, task_list, model, budget):
        # In the rough exploration phase, we explore each direction of the $task_dim - 1$-dimensional subspace of the task space.
        # Here we use the one-hot vector to represent the direction of the subspace, any other orthonormal vectors should also be fine.
        
        task_name = list(task_list.keys())
        task_dim = model.get_output_dim()
        basis = np.eye(task_dim)
        basis[-1][-1] = 0
        task_dict = {}
        for i in range(task_dim - 1):
            task_dict[task_name[i]] = (basis[:, [i]].T, int(budget//(task_dim - 1)))
        return task_dict
    
    def fine_exploration_phase(self, task_list, model, budget, outer_epoch):
        # In the fine exploration phase, we aims to find the $embed_dim$-dimensional subspace of the task space that is the effective subspace of the task space.
        task_name = list(task_list.keys())
        embed_matrix = model.get_restricted_task_embed_matrix()
        _,_,vh = np.linalg.svd(embed_matrix, full_matrices=False)
        task_dict = {}
        for i, v in enumerate(vh):
            v = np.expand_dims(v,1)
            task_dict[task_name[i]] = (v.T, int(budget//len(vh)))
        return task_dict
    

    # TODO: or we could use Yingbing's method to gradually change the v. 
    # def exploitation_phase(self, task_list, model, budget, outer_epoch, target_task_dict):
    #     # In the exploitation phase, we focus on sample from sources tasks that are close to the target.

    #     task_dict = {}
    #     counter = 0
    #     task_name = list(target_task_dict.keys())
    #     for _, (target_vector, _) in target_task_dict.items():
    #         embed_matrx = model.get_full_task_embed_matrix()
    #         embed_restrict_matrx = model.get_restricted_task_embed_matrix()
    #         # TODO : might want to add r cond here when target is not single
    #         v = np.linalg.lstsq(embed_restrict_matrx, embed_matrx @ target_vector, rcond=None)[0]
    #         v_norm = np.linalg.norm(v)
    #         v = v/v_norm
    #         task_dict[task_name[counter]] = (v.T, int(budget//len(target_task_dict))) #TODO: multiply with np.linalg.norm(v)?
    #         counter += 1
        # return task_dict
    def exploitation_phase(self, task_list, model, budget, outer_epoch, target_task_dict):
        # In the exploitation phase, we focus on sample from sources tasks that are close to the target.

        task_dict = {}
        counter = 0
        task_name = list(target_task_dict.keys())
        for _, (target_vector, _) in target_task_dict.items():
            embed_matrx = model.get_full_task_embed_matrix()
            embed_restrict_matrx = model.get_restricted_task_embed_matrix()
            # TODO : might want to add r cond here when target is not single
            v = np.linalg.lstsq(embed_restrict_matrx, embed_matrx @ target_vector, rcond=None)[0]
            v[v < 0.05] = 0
            v_norm = np.linalg.norm(v)
            v = v/v_norm
            # min_dist = np.Infinity
            # min_task = ""
            for i, task_name in enumerate(list(task_list.keys())[:-1]):
                # w = task_list[task_name][0]
                # dist = rowspace_dist([v], [w], metric="avg")
                # # TODO: choose multiple ones, clip or not
                # if dist < min_dist:
                #     min_dist = dist
                #     min_task = task_name
                
            # task_dict[task_name[counter]] = (v.T, int(budget//len(target_task_dict))) #TODO: multiply with np.linalg.norm(v)?
                print(task_name, max(int(budget*v[i]**2), 50))
                task_dict[task_name] = (task_list[task_name][0], max(int(budget*v[i]**2), 50))
            counter += 1
        return task_dict
    
class MTALSampling_TaskSparse(Strategy):
    strategy_name = "mtal"

    def __init__(self, target_task_dict, fixed_inner_epoch_num, mode="target_awared"):
        super(MTALSampling_TaskSparse, self).__init__(target_task_dict, fixed_inner_epoch_num)
        self.mode = mode
        print(f"MTALSampling mode: {self.mode}")
        self.fine_exploration_vh = None

    def select(self, task_list, model, budget, outer_epoch, inner_epoch, seed = 42):
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
            return self.rough_exploration_phase(task_list, model, int(budget)), True
        else:
            if self.mode == "target_awared":
                # If not at the initial epoch, we first use the fine exploration phase to explore the effective subspace of the task space.
                # Then we use the exploitation phase to sample from the source tasks that are close to the target task.
                if inner_epoch == 0:
                    return self.fine_exploration_phase(task_list, model, int(budget**(3/4)*model.get_embed_dim()), outer_epoch), False
                else:
                    return self.exploitation_phase(task_list, model, int(budget - budget**(3/4)*model.get_embed_dim()), outer_epoch, self.target_task_dict), True
            elif self.mode == "target_agnostic":
                return self.fine_exploration_phase(task_list, model, int(budget), outer_epoch), True
            else:
                raise ValueError("The mode should be either 'target_awared' or 'target_agnostic'.")

    def rough_exploration_phase(self, task_list, model, budget):
        # In the rough exploration phase, we explore each direction of the $task_dim - 1$-dimensional subspace of the task space.
        # Here we use the one-hot vector to represent the direction of the subspace, any other orthonormal vectors should also be fine.
        task_name = list(task_list.keys())
        task_dim = model.get_output_dim()
        basis = np.eye(task_dim)
        basis[-1][-1] = 0
        task_dict = {}
        for i in range(task_dim - 1):
            # task_dict[f"rough_explore_{i}"] = (basis[:, [i]], int(budget//(task_dim - 1)))
            task_dict[task_name[i]] = (basis[:, [i]].T, int(budget//(task_dim - 1)))
        return task_dict
    
    def fine_exploration_phase(self, task_list, model, budget, outer_epoch):
        # In the fine exploration phase, we aims to find the $embed_dim$-dimensional subspace of the task space that is the effective subspace of the task space.
        embed_matrix = model.get_restricted_task_embed_matrix()
        _,_,vh = np.linalg.svd(embed_matrix, full_matrices=False)
        task_name = list(task_list.keys())
        if self.fine_exploration_vh is None:
            self.fine_exploration_vh = vh
        elif rowspace_dist(vh, self.fine_exploration_vh, metric="avg") < 0.8:
            print("Change exploration base", np.linalg.norm(self.fine_exploration_vh @vh.T, 'fro')**2/len(vh))
            self.fine_exploration_vh = vh

        task_dict = {}
        for i, v in enumerate(self.fine_exploration_vh):
            v = np.expand_dims(v,1)
            task_dict[task_name[i]] = (v.T, int(budget//len(self.fine_exploration_vh)))
        return task_dict
    

    # TODO: or we could use Yingbing's method to gradually change the v. 
    def exploitation_phase(self, task_list, model, budget, outer_epoch, target_task_dict):
        # In the exploitation phase, we focus on sample from sources tasks that are close to the target.

        task_dict = {}
        counter = 0
        task_name = list(target_task_dict.keys())
        for _, (target_vector, _) in target_task_dict.items():
            embed_matrx = model.get_full_task_embed_matrix()
            embed_restrict_matrx = model.get_restricted_task_embed_matrix()
            # TODO : might want to add r cond here when target is not single
            v = np.linalg.lstsq(embed_restrict_matrx, embed_matrx @ target_vector, rcond=None)[0]
            v_norm = np.linalg.norm(v)
            v = v/v_norm
            min_dist = np.infinity
            min_task = ""
            for task_name in list(task_list.keys()):
                w = task_list[task_name][0]
                dist = rowspace_dist(v, w, metric="avg")
                if dist < min_dist:
                    min_dist = dist
                    min_task = task_name
            # task_dict[task_name[counter]] = (v.T, int(budget//len(target_task_dict))) #TODO: multiply with np.linalg.norm(v)?
            task_dict[min_task] = (task_list[min_task][0], int(budget//len(target_task_dict)))
            counter += 1
        return task_dict        
