# List of available active learning strategies.
strategies = {}


class Strategy:
    """
    Abstract class for active learning strategies.
    """
    def __init__(self, target_task_dict, fixed_inner_epoch_num):
        """
        :param Dict target_task_dict: a dictionary of target tasks.
        :param int fixed_inner_epoch_num: the number of inner epochs. default: None. If None, the end of the inner loop will depend on the strategy.
        """
        self.inner_epoch_num = fixed_inner_epoch_num
        self.target_task_dict = target_task_dict

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
