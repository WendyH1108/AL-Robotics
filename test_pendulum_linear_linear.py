import numpy as np
from torch.nn import functional as F
from dataset.utils import load_and_process_data, generate_orth, generate_fourier_kernel
from dataset.pendulum_simulator import PendulumSimulatorDataset
from model.linear import ModifiedLinear
from model.shallow import ModifiedShallow
from model.bilinear import ModifiedBiLinear, ModifiedBiLinear_augmented
from strategies.al_sampling import MTALSampling, MTALSampling_TaskSparse
from strategies.baseline_sampling import RandomSampling, FixBaseSampling
from trainer.pytorch_passive_trainer import PyTorchPassiveTrainer
from trainer.trainer import *
from metrics.utils import rowspace_dist, rowspace_dist2, most_related_source

import pandas as pd
import seaborn as sns
import json
import argparse
from matplotlib import pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to configuration file.")
    parser.add_argument("--seed", type=int, default=43, help="Random seed.")
    args = parser.parse_args()

    with open(f"configs/pendulum_realLinear/{args.config}.json") as f:
        config = json.load(f)

    task_dim = 5 + 1 # TODO
    embed_dim = config["embed_dim"]
    input_dim = 2
    # This is actual the augmented dimension of the input. 
    # To be consistent with the synthetic, we use input_dim to denote the "dimension of the input to the model"
    aug_dim = config["input_dim"]
    
    #### Generate the synthetic target dataset
    # Set random seed to generate the data.
    data_seed = args.seed
    # Set the actual target that we cannot observe.
    actual_target = [0.5, 0.5, 0, 0, 0, 0] if "actual_target" not in config else config["actual_target"]
    actual_target = np.array(actual_target)
    # Generate a single target task that is perpendicular to the source tasks space.
    # Source tasks are in the first task_dim-1 dimensions, and the target task is in the last dimension.
    target_task_dict = {}
    # This corresponds to w=actual_target after projection. 
    # But we cannot observe, so we use the following instead.
    tmp = np.array([0., 0., 0., 0., 0., 1]) 
    tmp = np.expand_dims(tmp, axis = 1)
    target_task_dict.update({"target1_test": (tmp, 10000)})
    target_task_dict.update({"target1": (tmp, config["num_target_sample"])})
    true_v = actual_target/np.linalg.norm(actual_target)
    true_v = np.expand_dims(true_v, axis = 1)
    print("True target task: ", true_v)
    # Generate a fixed fourier kernel.
    w, b, fourier_kernel = generate_fourier_kernel(input_dim, aug_dim, seed = data_seed)
    # Generate the synthetic data for target tasks
    dataset = PendulumSimulatorDataset(fourier_kernel=fourier_kernel, actual_target=actual_target)
    # data_seed = config["data_seed"] if "data_seed" in config else config["task_embed_matrix_seed"]
    dataset.generate_synthetic_data({'target1_test': target_task_dict['target1_test']}, seed = 43 + 2343, noise_var= 0)
    dataset.generate_synthetic_data({'target1': target_task_dict['target1']}, seed = data_seed, noise_var=0.5)

    total_task_list = list(dataset.get_sampled_train_tasks().keys())
    dataset.get_dataset(total_task_list, mixed=True)

    def update_trainer_config(budget):
        if budget < 5e4:
            trainer_config = {"trainer_name":"pytorch_passive", "max_epoch": 15*embed_dim, "train_batch_size": 512, "lr": 0.1, "num_workers": 10,\
                            "optim_name": "AdamW", "wd":0.05, "scheduler_name": "StepLR", "step_size": 200, "gamma": 0.9,
                            "test_batch_size": 1000}
        else:
            trainer_config = {"trainer_name":"pytorch_passive", "max_epoch": 15*embed_dim, "train_batch_size": 512, "lr": 0.1, "num_workers": 10,\
                            "optim_name": "AdamW", "wd":0.05, "scheduler_name": "StepLR", "step_size": 500, "gamma": 0.9,
                            "test_batch_size": 1000}      
        trainer_config = get_optimizer_fn(trainer_config)
        trainer_config = get_scheduler_fn(trainer_config)
        return trainer_config
    
    train_model = ModifiedBiLinear_augmented(aug_dim, task_dim, embed_dim, ret_emb = False)
    if config["active"]:
        stable_model = ModifiedBiLinear_augmented(aug_dim, task_dim, embed_dim, ret_emb = False)
    else:
        stable_model = train_model
    if config["active"]:
        strategy_mode = "target_agnostic" if not config["target_aware"] else "target_awared"
        if config["saving_task_num"]:
            strategy = MTALSampling_TaskSparse({'target1': target_task_dict['target1']}, fixed_inner_epoch_num=None, mode=strategy_mode)
        else:
            strategy = MTALSampling({'target1': target_task_dict['target1']}, fixed_inner_epoch_num=None, mode=strategy_mode) 
    else:
        if config["saving_task_num"]:
            strategy = FixBaseSampling({'target1': target_task_dict['target1']}, fixed_inner_epoch_num=1)
        else:
            strategy = RandomSampling({'target1': target_task_dict['target1']}, fixed_inner_epoch_num=1)

    exp_base = 1.5 if "exp_base" not in config else config["exp_base"]

    outer_epoch_num = 7 if "outer_epoch_num" not in config else config["outer_epoch_num"]
    base_len_ratio = 1 if "base_len_ratio" not in config else config["base_len_ratio"]
    # Right now it is still exponential increase, but we can change it to linear increase.
    # Change to linear increase requires more careful design of the budget allocation. Pending.
    # Or change it to a fixed budget for each outer epoch. Pending.
    cumulative_budgets = []
    losses = []
    related_source_est_similarities = []
    condi = 1
    for outer_epoch in range(0,outer_epoch_num):
        print("Outer epoch: ", outer_epoch)
        if outer_epoch == 0:
            budget = aug_dim * embed_dim**2 * condi * base_len_ratio
        else:
            budget = aug_dim * embed_dim**2 * condi * base_len_ratio * (exp_base**outer_epoch)
        end_of_epoch = False
        inner_epoch = 0
        trainer = PyTorchPassiveTrainer(update_trainer_config(budget), train_model)
        if config["active"]:
            stable_trainer = PyTorchPassiveTrainer(update_trainer_config(budget), stable_model)
        while not end_of_epoch:
            cur_task_dict, end_of_epoch = strategy.select(stable_model, budget, outer_epoch, inner_epoch, adjustable_budget_ratio=1)
            # print("Current task dict: ", cur_task_dict) #debug
            # cur_task_dict = {"source1": (np.array([0.0266657, -0.1159717, 0.526, 0.380503,  0.4326478, 0.348809, -0.2556309, -0.436159, 0.]), budget)} # debug
            # cur_task_dict = {"source1": (np.array([ 0.0177939, 0.00169856, 0.57302446, 0.13202089, 0.49981853, 0.12407007, -0.34877699, -0.51675585, 0.  ]), budget)} # debug
            # cur_task_dict = {"source1": (true_v, budget)} # debug
            # cur_task_dict = {"source1": (actual_target, budget)} # debug
            # print("Current task dict: ", cur_task_dict) #debug
            dataset.generate_synthetic_data(cur_task_dict, seed = outer_epoch +  data_seed, noise_var=0.5) # noise_var=None
            total_task_list = list(dataset.get_sampled_train_tasks().keys())
            
            if config["active"]:
                if outer_epoch == 0:
                    stable_trainer.train(dataset, total_task_list , freeze_rep = False, shuffle=True, need_print=False)
                    trainer.train(dataset, total_task_list , freeze_rep = False, shuffle=True, need_print=False)
                elif not end_of_epoch:
                    exploration_task_list = []
                    for task in total_task_list:
                        if "exploit" not in task: exploration_task_list.append(task)
                    stable_trainer.train(dataset, exploration_task_list , freeze_rep = False, shuffle=True, need_print=False)
                else:
                    exploitation_task_list = ["target1"]
                    for task in total_task_list:
                        if "exploit" in task: exploitation_task_list.append(task)
                    # Note here we retrain the model from scratch for each outer epoch. Can we do better? (i.e. start from previous model)
                    trainer.train(dataset, total_task_list , freeze_rep = False, shuffle=True, need_print=False)
            else:
                trainer.train(dataset, total_task_list , freeze_rep = False, shuffle=True, need_print=False)
                stable_model = train_model
            


            inner_epoch += 1
        avg_Loss = trainer.test(dataset, dataset.get_sampled_test_tasks().keys())
        losses.append(avg_Loss)
        if cumulative_budgets:
            cumulative_budgets.append(cumulative_budgets[-1] + budget)
        else:
            cumulative_budgets.append(budget)            
            
        # Distance between the ground truth most related source task and the estimation most related source task
        similarity, est_v = most_related_source(stable_model, target_task_dict["target1"][0] ,true_v)
        print(f"Inner epoch {inner_epoch}: The similarity between the estimated and true most related source task is: ", similarity)
        print(est_v.T) #debug
        if inner_epoch ==0: related_source_est_similarities.append(similarity.item())

    print(cumulative_budgets)
    print(losses)
    print(related_source_est_similarities)


    results = pd.DataFrame({"budget": cumulative_budgets, "loss": losses, "related_source_est_similarities": related_source_est_similarities})
                           
    results_name = f"embed_dim{config['embed_dim']}"
    results_name += "_active" if config["active"] else "_passive"
    results_name += "_saving_task_num" if config["saving_task_num"] else "_not_saving_task_num"
    if config["active"]:
        results_name += "_target_aware" if config["target_aware"] else "_target_agnostic"
    results_name += f"_target_sample_num{config['num_target_sample']}"
    results_name += f"_seed{data_seed}"
    results_name += f"_actual_target{config['actual_target']}" if "actual_target" in config else "default"
    results.to_csv(f"results/pendulum_realLinear/{results_name}.csv", index=False)

    # fig, axes = plt.subplots(1,2, figsize=(25,25))
    # axes[0].set_title('Test loss for target')
    # sns.lineplot(x="budget", y="loss", data=results, ax = axes[0])
    # axes[1].set_title('Acc on estimated most related source task')
    # sns.lineplot(x="budget", y="related_source_est_similarities", data=results, ax = axes[1])

    # fig.savefig(f"results/pendulum_ball/{results_name}.pdf")


