import numpy as np
from dataset.utils import load_and_process_data, generate_orth
from dataset.dataset import SyntheticDataset
from model.linear import ModifiedLinear
from model.shallow import ModifiedShallow
from model.bilinear import ModifiedBiLinear
from strategies.al_sampling import MTALSampling
from strategies.baseline_sampling import RandomSampling
from trainer.pytorch_passive_trainer import PyTorchPassiveTrainer
from trainer.trainer import *

# Generate the synthetic input dataset
num_unlabeled_sample = 50000
input_dim = 100
input_data = np.random.random((num_unlabeled_sample, input_dim))

# Create the input_embed_model and the task_embed_model used to generate the synthetic data
# Here the input_embed_model is a shallow neural network with 2 hidden layers, (\phi)
# and the task_embed_model is a linear model. (B_w)
input_dim = input_data.shape[1] # dim(x)
embed_dim = 5 # dim(\phi(x))
task_dim = 50 # dim(w)

hidden_layers = [input_dim, input_dim, embed_dim]
input_embed_matrix = generate_orth((input_dim, embed_dim))
input_embed_model = ModifiedLinear(input_embed_matrix)
task_embed_matrix = np.random.random((embed_dim , task_dim))
task_embed_model = ModifiedLinear(task_embed_matrix)
# combined_model = ModifiedBiLinear(input_dim, task_dim, embed_dim, ret_emb = False)
# combined_model.update_input_embedding(input_embed_matrix)
# combined_model.update_task_embedding(task_embed_matrix)


#### Generate the synthetic target dataset
# For each source task, we assume there exsits a few shot samples.
num_target = input_dim * embed_dim**2 * 5

# Generate a signle target task that is perpendicular to the source tasks space.
# Source tasks are in the first task_dim-1 dimensions, and the target task is in the last dimension.
target_task_dict = {}
tmp = np.zeros((task_dim,1))
tmp[task_dim-1][0] = 1
# target_task_dict.update({"target1_train": (tmp, num_target)})
target_task_dict.update({"target1_test": (tmp, 2)})

# ## Ground truth
# embed_matrx = task_embed_model.get_full_task_embed_matrix()
# embed_restrict_matrx = task_embed_model.get_restricted_task_embed_matrix()
# v = np.linalg.lstsq(embed_restrict_matrx, embed_matrx @ target_task_dict["target1_train"][0])[0]
# v_norm = np.linalg.norm(v)
# v = v/np.linalg.norm(v)
# # print(f"estimated direction norm with B : {embed_restrict_matrx @ v/np.linalg.norm(embed_restrict_matrx @ v)}")
# # print(f"estimated direction norm : {v}")


###### Test the AL strategy #####

# Generate the synthetic data for target tasks
# dataset = SyntheticDataset(input_data, input_embed_model=None, task_embed_model=None,model=combined_model, noise_var=0.00)
dataset = SyntheticDataset(input_data, input_embed_model=input_embed_model, task_embed_model=task_embed_model,model=None, noise_var=0.00)
dataset.generate_synthetic_data(target_task_dict)

# Initialize the model, trainer
trainer_config = {"trainer_name":"pytorch_passive", "max_epoch": 100, "train_batch_size": 1000, "lr": 0.1, "num_workers": 4,\
                  "optim_name": "SGD", "scheduler_name": "StepLR", "step_size": 500, "gamma": 0.1,
                  "test_batch_size": 500}
trainer_config = get_optimizer_fn(trainer_config)
trainer_config = get_scheduler_fn(trainer_config)
model = ModifiedBiLinear(input_dim, task_dim, embed_dim, ret_emb = False)
model.update_input_embedding(input_embed_matrix)
model.update_task_embedding(task_embed_matrix)
trainer = PyTorchPassiveTrainer(trainer_config, model)

# Target task only training 
# trainer.train(dataset, dataset.get_sampled_train_tasks().keys() , freeze_rep = False, need_print=True)
trainer.test(dataset, dataset.get_sampled_test_tasks().keys())



# # Initialize the strategy
# strategy = RandomSampling(target_task_dict, fixed_inner_epoch_num=None)
# base_len = input_dim * embed_dim**2 * 5

# outer_epoch_num = 10
# # Right now it is still exponential increase, but we can change it to linear increase.
# # Change to linear increase requires more careful design of the budget allocation. Pending.
# # Or change it to a fixed budget for each outer epoch. Pending.
# for outer_epoch in range(outer_epoch_num):
#     budget = base_len
#     end_of_epoch = False
#     inner_epoch = 0
#     while not end_of_epoch:
#         cur_task_dict, end_of_epoch = strategy.select(model, budget, outer_epoch, inner_epoch)
#         dataset.generate_synthetic_data(cur_task_dict)
#         total_task_list = list(dataset.get_sampled_train_tasks().keys())
#         # Note here we retrain the model from scratch for each outer epoch. Can we do better? (i.e. start from previous model)
#         trainer.train(dataset, total_task_list , freeze_rep = False)
#         inner_epoch += 1
#     trainer.test(dataset, dataset.get_sampled_test_tasks().keys())






