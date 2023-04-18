import numpy as np
from dataset.utils import load_and_process_data
from dataset.dataset import SyntheticDataset
from model.linear import ModifiedLinear
from model.shallow import ModifiedShallow
from strategies.al_sampling import MTALSampling
from strategies.baseline_sampling import RandomSampling
from trainer.pytorch_passive_trainer import PyTorchPassiveTrainer
from trainer.trainer import *

# Load the input data from the existing files.
shared_features = ['v', 'q', 'pwm']
dim_a = 3
label = 'fa'
# Training data collected from the neural-fly drone
dataset = 'neural-fly' 
dataset_folder = 'data/training'
# Training data collected from an intel aero drone
modelname = f"{dataset}_dim-a-{dim_a}_{'-'.join(shared_features)}" # 'intel-aero_fa-num-Tsp_v-q-pwm'
input_data = load_and_process_data(dataset_folder, shared_features)

# Create the input_embed_model and the task_embed_model used to generate the synthetic data
# Here the input_embed_model is a shallow neural network with 2 hidden layers, (\phi)
# and the task_embed_model is a linear model. (B_w)
input_dim = input_data.shape[1] # dim(x)
embed_dim = 2 # dim(\phi(x))
task_dim = 7 # dim(w)

hidden_layers = [input_dim, input_dim, embed_dim]
input_embed_model = ModifiedShallow(input_dim, task_dim, hidden_layers, ret_emb = True)
task_embed_matrix = np.random.random((embed_dim , task_dim))
task_embed_model = ModifiedLinear(task_embed_matrix)

# Create the synthetic dataset
dataset = SyntheticDataset(input_data, input_embed_model, task_embed_model)

#### Generate the synthetic target dataset
# For each source task, we assume there exsits a few shot samples.
num_target = 20

# Generate a signle target task that is perpendicular to the source tasks space.
# Source tasks are in the first task_dim-1 dimensions, and the target task is in the last dimension.
target_task_dict = {}
tmp = np.zeros((task_dim,1))
tmp[task_dim-1][0] = 1
target_task_dict.update({"perp_single_target": (tmp, num_target)})

# Set test dataset
tmp = np.zeros((task_dim,1))
tmp[task_dim-1][0] = 1
target_task_dict_test = {"perp_single_target_test": (tmp, 1000)}

## Ground truth
embed_matrx = task_embed_model.get_full_task_embed_matrix()
embed_restrict_matrx = task_embed_model.get_restricted_task_embed_matrix()
v = np.linalg.lstsq(embed_restrict_matrx, embed_matrx @ target_task_dict["perp_single_target"][0])[0]
print(f"estimated direction norm with B : {embed_restrict_matrx @ v}")
print("estimated direction norm with B : {}".format(embed_matrx @ target_task_dict["perp_single_target"][0]))
v_norm = np.linalg.norm(v)
v = v/np.linalg.norm(v)
# print(f"estimated direction: {v} with norm {v_norm}")
# print(f"estimated direction norm with B : {embed_restrict_matrx @ v/np.linalg.norm(embed_restrict_matrx @ v)}")


###### Test the AL strategy #####

# Generate the synthetic data for target tasks
dataset = SyntheticDataset(input_data, input_embed_model, task_embed_model)
dataset.generate_synthetic_data(target_task_dict)
dataset.generate_synthetic_data(target_task_dict_test)

# Initialize the model, trainer and the AL strategy
trainer_config = {"trainer_name":"pytorch_passive", "max_epoch": 100, "train_batch_size": 200, "lr": 0.01, "num_workers": 4,\
                  "optim_name": "SGD", "scheduler_name": "StepLR", "step_size": 100, "gamma": 0.9,
                  "test_batch_size": 500}
trainer_config = get_optimizer_fn(trainer_config)
trainer_config = get_scheduler_fn(trainer_config)
al_strategy = MTALSampling(target_task_dict, fixed_inner_epoch_num=None)
base_len = embed_dim * input_dim
model = ModifiedShallow(input_dim, task_dim, [input_dim, input_dim, embed_dim], ret_emb = False)
trainer = PyTorchPassiveTrainer(trainer_config, model)
outer_epoch_num = 3
# Right now it is still exponential increase, but we can change it to linear increase.
# Change to linear increase requires more careful design of the budget allocation. Pending.
# Or change it to a fixed budget for each outer epoch. Pending.
for outer_epoch in range(outer_epoch_num):
    budget = base_len * (1.1 ** outer_epoch)
    end_of_epoch = False
    inner_epoch = 0
    while not end_of_epoch:
        cur_task_dict, end_of_epoch = al_strategy.select(model, budget, outer_epoch, inner_epoch)
        dataset.generate_synthetic_data(cur_task_dict)
        total_task_list = list(dataset.get_sampled_train_tasks().keys())
        # Note here we retrain the model from scratch for each outer epoch. Can we do better? (i.e. start from previous model)
        trainer.train(dataset, total_task_list , freeze_rep = False)
        inner_epoch += 1
    trainer.test(dataset, target_task_dict_test.keys())


#### Test the baseline strategy

# Generate the synthetic data for target tasks
dataset = SyntheticDataset(input_data, input_embed_model, task_embed_model)
dataset.generate_synthetic_data(target_task_dict)
dataset.generate_synthetic_data(target_task_dict_test)

# Initialize the model, trainer and the AL strategy
trainer_config = {"trainer_name":"pytorch_passive", "max_epoch": 100, "train_batch_size": 200, "lr": 0.01, "num_workers": 4,\
                  "optim_name": "SGD", "scheduler_name": "StepLR", "step_size": 100, "gamma": 0.9,
                  "test_batch_size": 500}
trainer_config = get_optimizer_fn(trainer_config)
trainer_config = get_scheduler_fn(trainer_config)
al_strategy = RandomSampling(target_task_dict, fixed_inner_epoch_num=2)
base_len = embed_dim * input_dim
model = ModifiedShallow(input_dim, task_dim, [input_dim, input_dim, embed_dim], ret_emb = False)
trainer = PyTorchPassiveTrainer(trainer_config, model)
outer_epoch_num = 3
# Right now it is still exponential increase, but we can change it to linear increase.
# Change to linear increase requires more careful design of the budget allocation. Pending.
# Or change it to a fixed budget for each outer epoch. Pending.
for outer_epoch in range(outer_epoch_num):
    budget = base_len * (1.1 ** outer_epoch)
    end_of_epoch = False
    inner_epoch = 0
    while not end_of_epoch:
        cur_task_dict, end_of_epoch = al_strategy.select(model, budget, outer_epoch, inner_epoch)
        dataset.generate_synthetic_data(cur_task_dict)
        total_task_list = list(dataset.get_sampled_train_tasks().keys())
        # Note here we retrain the model from scratch for each outer epoch. Can we do better? (i.e. start from previous model)
        trainer.train(dataset, total_task_list , freeze_rep = False)
        inner_epoch += 1
    trainer.test(dataset, target_task_dict_test.keys())






