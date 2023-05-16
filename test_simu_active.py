import pandas as pd
import seaborn as sns
import numpy as np
import re
import math
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import numpy as np
from dataset.utils import load_and_process_data, generate_orth, load_data,extract_features
from dataset.dataset import SyntheticDataset, SimuDataset
from model.linear import ModifiedLinear
from model.shallow import ModifiedShallow
from model.bilinear import ModifiedBiLinear
from strategies.al_sampling import MTALSampling
from strategies.al_sampling_simu import MTALSampling
from trainer.pytorch_passive_trainer import PyTorchPassiveTrainer
from trainer.trainer import *
import matplotlib.pyplot as plt
from metrics.utils_simu import rowspace_dist, rowspace_dist2
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int)
  args = parser.parse_args()
  shared_features = ['v', 'q', 'pwm']
  rawdata_nontrans = load_data("./data/training/")
  rawdata_transfer = load_data("./data/training-transfer/")
  # rawdata_transfer = []
  print("finish data loading")
  raw_data = rawdata_nontrans + [0] + rawdata_transfer
  input_dim = 11 # dim(x)
  embed_dim = 4 # dim(\phi(x))
  task_dim = 33 # dim(w)
  source_task_dict = {}
  test_task_dict = {}

  input_data = {}
  input_label = {}
  input_ws = {}
  task_embed_matrix = []
  fa_list = ["x","y","z"]
  transfer = "nontransfer"
  w_matrix = np.identity(task_dim)
  count = -1
  for task_data in raw_data:
    if task_data == 0:
        transfer = "transfer"
        continue
    if task_data["condition"] == "nowind":
        wind_speed = 0
    else:
        wind_speed = re.search(r'\d+', task_data["condition"]).group()
    for dim in range(task_data["fa"].shape[1]):
        # input_data.append(np.array(extract_features(task_data,shared_features)))
        # input_label.append(task_data["fa"][:,dim])
        count += 1
        task_name = f"{transfer}_{wind_speed}_{fa_list[dim]}"
        input_data[task_name] = np.array(extract_features(task_data,shared_features))
        input_label[task_name] = task_data["fa"][:,dim]
        # input_ws[task_name] = np.array([w_matrix[count]] * task_data["fa"].shape[0])
        input_ws[task_name] = np.array([w_matrix[count]])
        source_task_dict[task_name] = (w_matrix[count],50)
        
  print("finish data processing")
  test_task_dict[list(input_ws.keys())[task_dim-1] + "_test"] = (w_matrix[task_dim-1],1000)
  source_task_dict[list(input_ws.keys())[task_dim-1]] = (w_matrix[task_dim-1],500)
  task_dict = source_task_dict.copy()
  task_dict.update(test_task_dict)

  dataset = SimuDataset(input_data, input_label, input_ws)

  outer_epoch_num = 10
  base_len_ratio = 1
  exp_base = 1.2
  condi = 3
  culmulative_budgets = []
  losses = []
  related_source_est_similarities = [0]
  task_embed_space_est_similarities_upper = []
  task_embed_space_est_similarities_lower = []
  avg_training_loss = []

  config = {"trainer_name":"pytorch_passive", "max_epoch": 20, "train_batch_size": 1000, "lr": 0.01, "num_workers": 4,\
                    "optim_name": "AdamW", "scheduler_name": "StepLR", "step_size": 500, "gamma": 0.1,
                    "test_batch_size": 500}
  seed = args.seed
  config = get_optimizer_fn(config)
  config = get_scheduler_fn(config)
  model = ModifiedBiLinear(input_dim, task_dim, embed_dim, ret_emb = False)
  trainer = PyTorchPassiveTrainer(config, model)
  strategy = MTALSampling(test_task_dict, fixed_inner_epoch_num=None, mode="target_awared")
  copy_input_ws = dataset.input_ws.copy()
  copy_source_task_dict = source_task_dict.copy()
  for outer_epoch in range(outer_epoch_num):
      total_training_loss = 0
      print("Outer epoch: ", outer_epoch)
      if outer_epoch == 0:
          budget = input_dim**2 * embed_dim**2 * condi * base_len_ratio #input_dim&**2 count for hidden layer
          print(f"budget: {budget}")
      else:
          budget = input_dim**2 * embed_dim**2 * condi * base_len_ratio * (exp_base**outer_epoch)
      end_of_epoch = False
      inner_epoch = 0
      while not end_of_epoch:
          cur_task_dict, end_of_epoch = strategy.select(source_task_dict, model, budget, outer_epoch, inner_epoch)
          # dataset.generate_synthetic_data(cur_task_dict, noise_var=1, seed=(154245) if outer_epoch==0 else None) # noise_var=0.2
        #   print(len(list(source_task_dict.keys())))
        #   print(source_task_dict.keys())
          # dataset.input_ws.update(cur_task_dict)
          source_task_dict.update(cur_task_dict)
          if not outer_epoch == 0:
            # dataset.input_ws[list(input_ws.keys())[task_dim-1]] = (w_matrix[task_dim-1],0)
            source_task_dict[list(input_ws.keys())[task_dim-1]] = (w_matrix[task_dim-1],0)
          print("this is source_task_dict", source_task_dict)
        #   break
          
          # Note here we retrain the model from scratch for each outer epoch. Can we do better? (i.e. start from previous model)
          total_training_loss += trainer.train(dataset, source_task_dict , freeze_rep = False, need_print=False, seed = seed)
          inner_epoch += 1
          # dataset.input_ws.update(copy_input_ws)
          # source_task_dict.update(copy_source_task_dict)
    #   break
      # print(source_task_dict) 
      avg_training_loss.append(total_training_loss/end_of_epoch)
      avg_Loss = trainer.test(dataset, test_task_dict,device = "cpu")
      losses.append(avg_Loss)
      if culmulative_budgets:
          culmulative_budgets.append(culmulative_budgets[-1] + budget)
      else:
          culmulative_budgets.append(budget)
  est_task_restrict_embed_matrix = model.get_restricted_task_embed_matrix()
  
  task_embed_space_est_similarities_upper = {}
  task_embed_space_est_similarities_lower = {}
  task_name_list = list(source_task_dict.keys())
  for task_idx in range(task_dim-1):
    upper, lower = rowspace_dist([est_task_restrict_embed_matrix[:,task_idx]], [est_task_restrict_embed_matrix[:,:-1]])
    task_embed_space_est_similarities_upper[task_name_list[task_idx]] = upper
    task_embed_space_est_similarities_lower[task_name_list[task_idx]] = lower
    
  upper_bound = pd.DataFrame.from_dict(task_embed_space_est_similarities_upper, orient='index')
  lower_bound = pd.DataFrame.from_dict(task_embed_space_est_similarities_lower, orient='index')
  upper_bound.to_csv(f"active_results/upper_bound_taskdim{task_dim}_seed{seed}.csv", index=False)
  lower_bound.to_csv(f"active_results/lower_bound_taskdim{task_dim}_seed{seed}.csv", index=False)

  # try:
  #     results = pd.DataFrame({"budget": culmulative_budgets, "loss": losses, "related_source_est_similarities": related_source_est_similarities, \
  #                             "task_embed_space_est_similarities_upper": task_embed_space_est_similarities_upper, \
  #                             "task_embed_space_est_similarities_lower": task_embed_space_est_similarities_lower})

  results = pd.DataFrame({"budget": culmulative_budgets, "loss": losses, "training_loss": avg_training_loss})     
        
  # results_name = f"embed_dim{config['embed_dim']}"
  # results_name += "_active" if config["active"] else "_passive"
  # results_name += "_saving_task_num" if config["saving_task_num"] else "_not_saving_task_num"
  results_name = "losses_target_agnostic"
  
  results.to_csv(f"active_results/{results_name}_taskdim{task_dim}_seed{seed}.csv", index=False)

  fig, axes = plt.subplots(2,2, figsize=(25,25))
  axes[0,0].set_title('Test loss for target')
  sns.lineplot(x="budget", y="loss", data=results, ax = axes[0,0])

  # axes[1,0].set_title('Input embed space similarity')
  # axes[1,1].set_title('Task embed space similarity')
  # sns.lineplot(x="budget", y="task_embed_space_est_similarities_lower", data=results, ax = axes[1,1])
  # fig.savefig(f"baseline_results/{results_name}.pdf")
      




      

      