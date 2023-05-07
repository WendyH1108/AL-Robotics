# List of available active learning strategies.
strategies = {}


class Strategy:
    """
    Abstract class for active learning strategies.
    """
    def __init__(self, target_task_dict, fixed_inner_epoch_num, task_dim, task_aug_dim=None, task_aug_kernel=None):
        """
        :param Dict target_task_dict: a dictionary of target tasks.
        :param int fixed_inner_epoch_num: the number of inner epochs. default: None. If None, the end of the inner loop will depend on the strategy.
        :param int task_dim: the dimension of the original task embedding.
        :param int task_aug_dim: the dimension of the augmented task embedding. default: None. If None, it will be set the same as task_dim.
        :param func task_aug_kernel: the kernel used to augment the task embedding. default: None. 
        """
        self.inner_epoch_num = fixed_inner_epoch_num
        self.target_task_dict = target_task_dict
        self.task_dim = task_dim
        self.task_aug_dim = task_dim if task_aug_dim is None else task_aug_dim
        self.task_aug_kernel = task_aug_kernel
        if self.task_aug_kernel is not None:
            assert self.task_aug_dim is not None, "task_aug_dim should be specified if task_aug_kernel is not None."

    def __init_subclass__(cls, **kwargs):
        """
        Register strategy by its strategy_name.
        """
        super().__init_subclass__(**kwargs)
        strategies[cls.strategy_name] = cls

    def select(self, model, budget, outer_epoch, inner_epoch, seed=None):
        """
        Selecting a batch of examples based on output from the trainer.

        :param model: the model to be trained.
        :param budget: the number of samples to label.
        :param outer_epoch: the outer epoch number.
        :param inner_epoch: the inner epoch number.
        :param seed: the random seed.
        """
        pass
