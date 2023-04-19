import numpy as np
from strategies.strategy_skeleton import Strategy

#TODO: Here I seed the exploration base change every epoch, but we could also seed it once and use the same exploration base for all epochs.
class RandomSampling(Strategy):
    strategy_name = "random"

    def __init__(self, target_task_dict, fixed_inner_epoch_num):
         super(RandomSampling, self).__init__(target_task_dict, fixed_inner_epoch_num)

    def select(self, task_embed_model, budget, outer_epoch, inner_epoch, seed=None):
        """Select a subset of the task to collect the data.

        Args:
            task_embed_model (nn.Module): the task embedding model
            budget (int): the number of samples to label)

        Returns:
            np.array: the indices of the selected samples
        """
        if seed is not None: np.random.seed(seed)
        task_dim = task_embed_model.get_output_dim()
        # The (task_dim - 1, task_dim) orthonormal basis of the source task space
        orth = np.zeros((task_dim - 1, task_dim))
        gaus = np.random.normal(0, 1, (task_dim-1, task_dim-1))
        _, _, tmp = np.linalg.svd(gaus)
        orth[:, :task_dim-1] = tmp

        # Format that as a dictionary.
        task_dict = {}
        for i, v in enumerate(orth):
           v = np.expand_dims(v,1)
           task_dict[f"random_base_epoch{outer_epoch}_{i}"] = (v, int(budget//len(orth)))
        #    print(f"v: {v}")
        #    print(f"{v.T @ v}")

        return task_dict, inner_epoch == self.inner_epoch_num - 1 if self.inner_epoch_num is not None else True
        