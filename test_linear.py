import numpy as np
from torch.nn import functional as F
from dataset.utils import load_and_process_data, generate_orth
from dataset.dataset import SyntheticDataset
from model.linear import ModifiedLinear
from model.shallow import ModifiedShallow
from model.bilinear import ModifiedBiLinear
from strategies.al_sampling import MTALSampling, MTALSampling_TaskSparse
from strategies.baseline_sampling import RandomSampling, FixBaseSampling
from trainer.pytorch_passive_trainer import PyTorchPassiveTrainer
from trainer.trainer import *
from metrics.utils import rowspace_dist, rowspace_dist2

import pandas as pd
import seaborn as sns
import json
import argparse
from matplotlib import pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to configuration file.")
    parser.add_argument("--data_seed", type=int, default=43)
    args = parser.parse_args()

    with open(f"configs/{args.config}.json") as f:
        config = json.load(f)

    config["num_target_sample"] = 800 #DO some hack here

    ## [Generate the synthetic input dataset] ##
    num_unlabeled_sample = 500000
    input_dim = config["input_dim"]
    np.random.seed(args.data_seed)
    input_data = np.random.random((num_unlabeled_sample, input_dim))
    embed_dim = config["embed_dim"] # dim(\phi(x))
    task_dim = embed_dim*config["task_embed_ratio"] # dim(w
    seed = config["task_embed_matrix_seed"] #42, 500, 124321

    # Generate the input embedding matrix B_X as an orthogonal matrix. input_embed_matrix = B_X
    hidden_layers = [input_dim, input_dim, embed_dim]
    input_embed_matrix = generate_orth((input_dim, embed_dim), seed=seed)

    # Generate the task embedding matrix B_W
    # Make sure it is full rank and it embalanced
    np.random.seed(seed)
    task_embed_matrix = np.random.random((embed_dim , task_dim))
    np.random.seed(seed+1000)
    # Let the latter 2/3 columns (except the last one) be similar so it is unbalanced.
    task_embed_matrix[:, task_dim//3:] = task_embed_matrix[:,[-1]] 
    task_embed_matrix += np.random.random((embed_dim , task_dim))*0.1
    # Make the last column equal to the first column
    task_embed_matrix[:, -1] = task_embed_matrix[:, 0]
    # Normalize the task embedding matrix.
    task_embed_matrix = task_embed_matrix / np.linalg.norm(task_embed_matrix, axis=0)
    # Compute the condition number of the task embedding matrix. 
    # The larger the condition number, the more unbalanced the task embedding matrix is. And therefore the more difficult the problem. is.
    condi = np.linalg.cond(task_embed_matrix[:,:-1])
    print(f"condi of task_embed_matrix: {condi}")

    # Build the pytorch model base on the input_embed_model and the task_embed_model.
    synthData_model = ModifiedBiLinear(input_dim, task_dim, embed_dim, ret_emb = False)
    synthData_model.update_input_embedding(input_embed_matrix)
    synthData_model.update_task_embedding(task_embed_matrix)

    #### Generate the synthetic target dataset
    # For each source task, we assume there exsits a few shot samples.
    num_target = input_dim * embed_dim
    # Generate a signle target task that is perpendicular to the source tasks space.
    # Source tasks are in the first task_dim-1 dimensions, and the target task is in the last dimension.
    target_task_dict = {}
    tmp = np.zeros((task_dim,1))
    tmp[task_dim-1][0] = 1
    target_task_dict.update({"target1_test": (tmp, 10000)})
    target_task_dict.update({"target1": (tmp, config["num_target_sample"])})
    # Generate the synthetic data for target tasks
    dataset = SyntheticDataset(input_data, input_embed_model=None, task_embed_model=None,model=synthData_model, noise_var=0)
    data_seed = config["data_seed"] if "data_seed" in config else config["task_embed_matrix_seed"]
    dataset.generate_synthetic_data({'target1_test': target_task_dict['target1_test']}, seed = data_seed, noise_var=0)
    dataset.generate_synthetic_data({'target1': target_task_dict['target1']}, noise_var=1, seed = data_seed)
    task_embed_restrict_matrix = synthData_model.get_restricted_task_embed_matrix()
    v = np.linalg.lstsq(task_embed_restrict_matrix, task_embed_matrix @ target_task_dict["target1_test"][0],rcond=None)[0]
    v_norm = np.linalg.norm(v)
    true_v = v / v_norm
    # print(f"The ground truth most related source task is {true_v} with norm {v_norm}.")

    # Multi-task learning
    trainer_config = {"trainer_name":"pytorch_passive", "max_epoch": 20*embed_dim, "train_batch_size": 600, "lr": 0.1, "num_workers": 4,\
                    "optim_name": "Adam", "wd":0.01, "scheduler_name": "StepLR", "step_size": 50*embed_dim, "gamma": 0.9,
                    "test_batch_size": 1000}
    trainer_config = get_optimizer_fn(trainer_config)
    trainer_config = get_scheduler_fn(trainer_config)
    train_model = ModifiedBiLinear(input_dim, task_dim, embed_dim, ret_emb = False)


    if config["active"]:
        strategy_mode = "target_agnostic" if not config["target_aware"] else "target_awared"
        if config["saving_task_num"]:
            strategy = MTALSampling_TaskSparse(target_task_dict, fixed_inner_epoch_num=None, mode=strategy_mode)
        else:
            strategy = MTALSampling(target_task_dict, fixed_inner_epoch_num=None, mode=strategy_mode) 
    else:
        if config["saving_task_num"]:
            strategy = FixBaseSampling(target_task_dict, fixed_inner_epoch_num=1)
        else:
            strategy = RandomSampling(target_task_dict, fixed_inner_epoch_num=1)

    exp_base = 1.2 if "exp_base" not in config else config["exp_base"]

    outer_epoch_num = 7 if "outer_epoch_num" not in config else config["outer_epoch_num"]
    base_len_ratio = 1 if "base_len_ratio" not in config else config["base_len_ratio"]
    # Right now it is still exponential increase, but we can change it to linear increase.
    # Change to linear increase requires more careful design of the budget allocation. Pending.
    # Or change it to a fixed budget for each outer epoch. Pending.
    culmulative_budgets = []
    losses = []
    related_source_est_similarities = [0]
    input_embed_space_est_similarities = []
    task_embed_space_est_similarities_upper = []
    task_embed_space_est_similarities_lower = []
    for outer_epoch in range(outer_epoch_num):
        print("Outer epoch: ", outer_epoch)
        if outer_epoch == 0:
            budget = input_dim * embed_dim**2 * condi * base_len_ratio
            print(f"budget: {budget}")
        else:
            budget = input_dim * embed_dim**2 * condi * base_len_ratio * (exp_base**outer_epoch)
        end_of_epoch = False
        inner_epoch = 0
        trainer = PyTorchPassiveTrainer(trainer_config, train_model)
        while not end_of_epoch:
            cur_task_dict, end_of_epoch = strategy.select(train_model, budget, outer_epoch, inner_epoch)
            dataset.generate_synthetic_data(cur_task_dict, noise_var=1, seed=(data_seed+154245) if outer_epoch==0 else None) # noise_var=0.2
            dataset.generate_val_data()
            total_task_list = list(dataset.get_sampled_train_tasks().keys())
            # Note here we retrain the model from scratch for each outer epoch. Can we do better? (i.e. start from previous model)
            trainer.train(dataset, total_task_list , freeze_rep = False, need_print=False)
            inner_epoch += 1
        avg_Loss = trainer.test(dataset, dataset.get_sampled_test_tasks().keys())
        losses.append(avg_Loss)
        if culmulative_budgets:
            culmulative_budgets.append(culmulative_budgets[-1] + budget)
        else:
            culmulative_budgets.append(budget)

        # Current estimation
        est_input_embed_matrix = train_model.get_input_embed_matrix()
        est_input_embed_matrix, s, vh = np.linalg.svd(est_input_embed_matrix, full_matrices=False)
        print("Input embed similarity btw est and ground truth:", rowspace_dist(est_input_embed_matrix.T, input_embed_matrix.T))
        input_embed_space_est_similarities.append(rowspace_dist(est_input_embed_matrix.T, input_embed_matrix.T)[1])
        est_task_restrict_embed_matrix = train_model.get_restricted_task_embed_matrix()
        # est_task_restrict_embed_matrix = np.diag(s) @ vh @ est_task_restrict_embed_matrix
        task_embed_restrict_matrix = synthData_model.get_restricted_task_embed_matrix()
        dist_to_upper_bound, dist_to_lower_bound = rowspace_dist2(est_task_restrict_embed_matrix, task_embed_restrict_matrix)
        print(f"Task embed dist compared to desired upper bound is {dist_to_upper_bound} and to lower bound is {dist_to_lower_bound}")
        task_embed_space_est_similarities_upper.append(dist_to_upper_bound)
        task_embed_space_est_similarities_lower.append(dist_to_lower_bound)
        
        # Distance between the ground truth most related source task and the estimation most related source task
        if f"exploit_epoch{outer_epoch}_{0}" in cur_task_dict:
            est_v = cur_task_dict[f"exploit_epoch{outer_epoch}_{0}"][0]
            # Compuete similarity
            print("The similarity betten the estimated and true most related source task is: ", est_v.T @ true_v)
            similarity = est_v.T @ true_v
            related_source_est_similarities.append(similarity[0][0])

    print(culmulative_budgets)
    print(losses)
    print(related_source_est_similarities)
    print(input_embed_space_est_similarities)
    print(task_embed_space_est_similarities_upper)
    print(task_embed_space_est_similarities_lower)


    try:
        results = pd.DataFrame({"budget": culmulative_budgets, "loss": losses, "related_source_est_similarities": related_source_est_similarities, \
                                "input_embed_space_est_similarities": input_embed_space_est_similarities, \
                                "task_embed_space_est_similarities_upper": task_embed_space_est_similarities_upper, \
                                "task_embed_space_est_similarities_lower": task_embed_space_est_similarities_lower})
    except:
        results = pd.DataFrame({"budget": culmulative_budgets, "loss": losses, \
                                "input_embed_space_est_similarities": input_embed_space_est_similarities, \
                                "task_embed_space_est_similarities_upper": task_embed_space_est_similarities_upper, \
                                "task_embed_space_est_similarities_lower": task_embed_space_est_similarities_lower})     
         
    results_name = f"embed_dim{config['embed_dim']}"
    results_name += "_active" if config["active"] else "_passive"
    results_name += "_saving_task_num" if config["saving_task_num"] else "_not_saving_task_num"
    if config["active"]:
        results_name += "_target_aware" if config["target_aware"] else "_target_agnostic"
    results_name += f"_target_sample_num{config['num_target_sample']}"
    results_name += f"_seed{config['task_embed_matrix_seed']}"
    results_name += f"_data_seed{args.data_seed}"
    print(results_name)
    results.to_csv(f"results/{results_name}.csv", index=False)

    # fig, axes = plt.subplots(2,2, figsize=(25,25))
    # axes[0,0].set_title('Test loss for target')
    # sns.lineplot(x="budget", y="loss", data=results, ax = axes[0,0])
    # try:
    #     axes[0,1].set_title('Acc on estimated most related source task')
    #     sns.lineplot(x="budget", y="related_source_est_similarities", data=results, ax = axes[0,1])
    # except:
    #     pass
    # axes[1,0].set_title('Input embed space similarity')
    # sns.lineplot(x="budget", y="input_embed_space_est_similarities", data=results, ax = axes[1,0])
    # axes[1,1].set_title('Task embed space similarity')
    # sns.lineplot(x="budget", y="task_embed_space_est_similarities_lower", data=results, ax = axes[1,1])
    # fig.savefig(f"results/results/{results_name}.pdf")
    



