import numpy as np
from strategies.strategy_skeleton import Strategy


#TODO: Here I seed the exploration base change every epoch, but we could also seed it once and use the same exploration base for all epochs.
class MTALSampling(Strategy):
    strategy_name = "mtal"

    def __init__(self, target_task_dict, fixed_inner_epoch_num):
        super(MTALSampling, self).__init__(target_task_dict, fixed_inner_epoch_num)

    def select(self, model, budget, outer_epoch, inner_epoch, seed = 42):
        """Select a subset of the task to collect the data.

        Args:
            model (nn.Module): the task embedding model
            budget (int): the number of total samples to label (sum of all tasks)

        Returns:
            np.array: the indices of the selected samples
        """
        if outer_epoch == 0:
            # If at the initial epoch, we use the rough exploration phase to explore every direction of the task space.
            np.random.seed(seed)
            return self.rough_exploration_phase(model, int(budget)), True
        else:
            # If not at the initial epoch, we first use the fine exploration phase to explore the effective subspace of the task space.
            # Then we use the exploitation phase to sample from the source tasks that are close to the target task.
            if inner_epoch == 0:
                return self.fine_exploration_phase(model, int(budget), outer_epoch), False
            else:
                return self.exploitation_phase(model, int(budget**(4/3)), outer_epoch, self.target_task_dict), True
            

    def rough_exploration_phase(self, model, budget):
        # In the rough exploration phase, we explore each direction of the $task_dim - 1$-dimensional subspace of the task space.
        # Here we use the one-hot vector to represent the direction of the subspace, any other orthonormal vectors should also be fine.

        task_dim = model.get_output_dim()
        basis = np.eye(task_dim)
        basis[-1][-1] = 0
        task_dict = {}
        for i in range(task_dim - 1):
            task_dict[f"rough_explore_{i}"] = (basis[:, [i]], int(budget//(task_dim - 1)))
        return task_dict
    
    def fine_exploration_phase(self, model, budget, outer_epoch):
        # In the fine exploration phase, we aims to find the $embed_dim$-dimensional subspace of the task space that is the effective subspace of the task space.
        
        embed_matrix = model.get_restricted_task_embed_matrix()
        _,_,vh = np.linalg.svd(embed_matrix, full_matrices=False)
        task_dict = {}
        for i, v in enumerate(vh):
            v = np.expand_dims(v,1)
            task_dict[f"fine_explore_epoch{outer_epoch}_{i}"] = (v, int(budget//embed_matrix.shape[1]))
        return task_dict
    

    # TODO: or we could use Yingbing's method to gradually change the v. 
    def exploitation_phase(self, model, budget, outer_epoch, target_task_dict):
        # In the exploitation phase, we focus on sample from sources tasks that are close to the target.

        task_dict = {}
        counter = 0
        for _, (target_vector, _) in target_task_dict.items():
            embed_matrx = model.get_full_task_embed_matrix()
            embed_restrict_matrx = model.get_restricted_task_embed_matrix()
            # TODO : might want to add r cond here when target is not single
            v = np.linalg.lstsq(embed_restrict_matrx, embed_matrx @ target_vector)[0]
            v_norm = np.linalg.norm(v)
            print(f"estimated direction norm with B : {embed_restrict_matrx @ v}")
            v = v/v_norm
            task_dict[f"exploit_epoch{outer_epoch}_{counter}"] = (v, int(budget//len(target_task_dict))) #TODO: multiply with np.linalg.norm(v)?
            counter += 1
        return task_dict