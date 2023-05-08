import numpy as np
from dataset.utils import load_and_process_data, generate_orth, generate_fourier_kernel, generate_pendulum_specified_kernel
from dataset.pendulum_simulator import PendulumSimulatorDataset, Pendulum
from model.shallow import ModifiedShallow
from model.bilinear import ModifiedBiLinear, ModifiedBiLinear_augmented
from strategies.al_sampling import MTALSampling, MTALSampling_TaskSparse
from strategies.baseline_sampling import RandomSampling, FixBaseSampling
from trainer.pytorch_passive_trainer import PyTorchPassiveTrainer
from trainer.trainer import *
from metrics.utils import rowspace_dist, rowspace_dist2, most_related_source, sim_pendulum

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

    folder = "pendulum_nonlinear_highErrorBar"

    with open(f"configs/{folder}/{args.config}.json") as f:
        config = json.load(f)

    task_dim = 5 + 1 # TODO
    embed_dim = config["embed_dim"]
    input_dim = 2
    # This is actual the augmented dimension of the input. 
    # To be consistent with the synthetic, we use input_dim to denote the "dimension of the input to the model"
    aug_dim = config["input_dim"]
    
    #### Generate the synthetic target dataset
    # Set random seed to generate the data and noise.
    data_seed = args.seed
    noise_var = config["noise_var"] if "noise_var" in config else 0.5
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
    target_task_dict.update({"target1_test": (tmp, 20000)})
    target_task_dict.update({"target1": (tmp, config["num_target_sample"])})
    
    # Generate a fixed fourier kernel.
    w, b, fourier_kernel = generate_fourier_kernel(input_dim, aug_dim, seed = data_seed)    
    # Generate the task_aug_kernel
    task_aug_dim, task_aug_kernel = generate_pendulum_specified_kernel(task_dim, 13, seed = data_seed)
    true_v = np.expand_dims(actual_target, axis = 0)
    true_v = task_aug_kernel(true_v)
    print(true_v)
    true_v = true_v.T/np.linalg.norm(true_v)
    print("True target task after aug: ", true_v)
    # Generate the synthetic data for target tasks
    dataset = PendulumSimulatorDataset(input_aug_kernel=fourier_kernel, actual_target=actual_target)
    dataset.generate_synthetic_data({'target1_test': target_task_dict['target1_test']}, seed = 43 + 2343, noise_var= 0)
    dataset.generate_synthetic_data({'target1': target_task_dict['target1']}, seed = data_seed, noise_var= noise_var)

    total_task_list = list(dataset.get_sampled_train_tasks().keys())
    dataset.get_dataset(total_task_list, mixed=True)

    ## Set the trainer config
    def update_trainer_config(budget):
        if budget < 2e4:
            trainer_config = {"trainer_name":"pytorch_passive", "max_epoch": 10*embed_dim, "train_batch_size": 512, "lr": 0.01, "num_workers": 10,\
                            "optim_name": "AdamW", "wd":0.05, "scheduler_name": "StepLR", "step_size": 500, "gamma": 0.9,
                            "test_batch_size": 1000}
        elif budget < 4e4:
            trainer_config = {"trainer_name":"pytorch_passive", "max_epoch": 12*embed_dim, "train_batch_size": 512, "lr": 0.01, "num_workers": 10,\
                            "optim_name": "AdamW", "wd":0.05, "scheduler_name": "StepLR", "step_size": 900, "gamma": 0.9,
                            "test_batch_size": 1000}
        else:
            trainer_config = {"trainer_name":"pytorch_passive", "max_epoch": 15*embed_dim, "train_batch_size": 512, "lr": 0.01, "num_workers": 10,\
                            "optim_name": "AdamW", "wd":0.05, "scheduler_name": "StepLR", "step_size": 1300, "gamma": 0.9,
                            "test_batch_size": 1000}      
        # trainer_config = {"trainer_name":"pytorch_passive", "max_epoch": 10*embed_dim, "train_batch_size": 512, "lr": 0.01, "num_workers": 10,\
        #                     "optim_name": "SGD", "scheduler_name": "StepLR", "step_size": 500, "gamma": 0.9,
        #                     "test_batch_size": 1000}
        # trainer_config = {"trainer_name":"pytorch_passive", "max_epoch": 2, "train_batch_size": 512, "lr": 0.01, "num_workers": 10,\
        #                 "optim_name": "SGD", "scheduler_name": "StepLR", "step_size": 500, "gamma": 0.9,
        #                 "test_batch_size": 1000}
        trainer_config = get_optimizer_fn(trainer_config)
        trainer_config = get_scheduler_fn(trainer_config)
        return trainer_config
    


    # Generate the model
    torch.manual_seed(43)
    train_model = ModifiedBiLinear_augmented(aug_dim, task_aug_dim, embed_dim, ret_emb = False)
    if config["active"]:
        torch.manual_seed(43)
        stable_model = ModifiedBiLinear_augmented(aug_dim, task_aug_dim, embed_dim, ret_emb = False)
    else:
        stable_model = train_model

    # Train the model using target only for testing
    total_task_list = list(dataset.get_sampled_train_tasks().keys())
    trainer = PyTorchPassiveTrainer(update_trainer_config(0), train_model, task_aug_kernel = task_aug_kernel)
    trainer.train(dataset, total_task_list , freeze_rep = False, shuffle=True, need_print=False)
    trainer.test(dataset, dataset.get_sampled_test_tasks().keys())

    # Set the strategy
    if config["active"]:
        strategy_mode = "target_agnostic" if not config["target_aware"] else "target_awared"
        if config["saving_task_num"]:
            strategy = MTALSampling_TaskSparse({'target1': target_task_dict['target1']}, fixed_inner_epoch_num=None, mode=strategy_mode, task_dim=task_dim, task_aug_dim = task_aug_dim, task_aug_kernel = task_aug_kernel)
        else:
            strategy = MTALSampling({'target1': target_task_dict['target1']}, fixed_inner_epoch_num=None, mode=strategy_mode, task_dim=task_dim, task_aug_dim = task_aug_dim, task_aug_kernel = task_aug_kernel) 
    else:
        if config["saving_task_num"]:
            strategy = FixBaseSampling({'target1': target_task_dict['target1']}, fixed_inner_epoch_num=1, task_dim=task_dim, task_aug_dim = task_aug_dim,task_aug_kernel = task_aug_kernel)
        else:
            strategy = RandomSampling({'target1': target_task_dict['target1']}, fixed_inner_epoch_num=1, task_dim=task_dim, task_aug_dim = task_aug_dim,task_aug_kernel = task_aug_kernel)

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
        

        trainer = PyTorchPassiveTrainer(update_trainer_config(budget), train_model, task_aug_kernel = task_aug_kernel)
        if config["active"]:
            stable_trainer = PyTorchPassiveTrainer(update_trainer_config(budget), stable_model, task_aug_kernel = task_aug_kernel)
        while not end_of_epoch:
            cur_task_dict, end_of_epoch = strategy.select(stable_model, budget, outer_epoch, inner_epoch, adjustable_budget_ratio=1)
            # print("Current task dict: ", cur_task_dict) #debug
            # cur_task_dict = {"source1": (np.array([ 0.26983025, -0.15776465, -0.14325377, -0.83220568,  0.43498737,  0.  ]), budget)} # debug
            # cur_task_dict = {"source1": (np.array([ 0, 0, 1, 0, 0,  0.  ]), budget)} # debug
            # cur_task_dict = {"source1": (true_v, budget)} # debug
            # cur_task_dict = {"source1": (actual_target, budget)} # debug
            # print("Current task dict: ", cur_task_dict) #debug
            dataset.generate_synthetic_data(cur_task_dict, seed = outer_epoch +  data_seed, noise_var=noise_var) # noise_var=None
            total_task_list = list(dataset.get_sampled_train_tasks().keys())
            # print(dataset.get_sampled_train_tasks())
            
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
        if config["active"] and outer_epoch > 0:
            already_compute = cur_task_dict[f"exploit_epoch{outer_epoch}_{0}"][0]
        else:
            already_compute = None
        similarity, est_v = most_related_source(stable_model, target_task_dict["target1"][0] ,true_v, task_dim, domain='pendulum', task_aug_kernel = task_aug_kernel, already_compute = already_compute)
        print(f" The similarity between the estimated and true most related source task is: ", similarity)
        print(est_v.T) #debug
        related_source_est_similarities.append(similarity.item())

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
    results.to_csv(f"results/{folder}/{results_name}.csv", index=False)

    # Save the model
    torch.save(train_model.state_dict(), f"results/{folder}/{results_name}.pt")


    # def predict_from_trainedModel(x):
    #     aug_x = fourier_kernel(x)
    #     aug_x = torch.from_numpy(aug_x).float().cuda()
    #     train_model.eval()
    #     observe_target_w = dataset.get_sampled_test_tasks()["target1_test"]
    #     observe_target_w = task_aug_kernel(observe_target_w.T)
    #     observe_target_w = torch.from_numpy(observe_target_w).float().cuda()
    #     return train_model(aug_x, observe_target_w.T, ret_feat_and_label=False).cpu().detach().numpy()

    # pendulum = Pendulum(duration=20, w=actual_target, policy_type="learned", learned_F_model = predict_from_trainedModel, noise_var=0.05, seed= 43 + 2343)
    # sim_pendulum(pendulum, plot=True, save_path=f"results/{folder}/{results_name}.png")
    # if config["active"]:
    #     pendulum = Pendulum(duration=20, w=actual_target, policy_type="oracle", noise_var=0.05, seed= 43 + 2343)
    #     sim_pendulum(pendulum, plot=True, save_path=f"results/{folder}/{results_name}_gt.png")




