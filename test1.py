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

hidden_layers = [32, 32, embed_dim]
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

# # Generate a set of target tasks that are overlapping with the source tasks space.
# # Source tasks are in the first task_dim-1 dimensions, and the target tasks are in the whole task_dim dimensions.
# # Here we assume size of target task is 3.
# target_task_dict = {}
# num_target_tasks = 3
# for i in range(num_target_tasks):
#     tmp = np.random.uniform(low=0, high=1, size=(task_dim, 1))
#     target_task_dict.update({f"overlapping_target_{i}": (tmp, num_target)})

# # Generate a set of target tasks that are totally inside the source tasks space.
# # Here we assume size of target task is task_dim-1, which is the same as the source tasks.
# target_task_dict = {}
# num_target_tasks = task_dim-1
# for i in range(num_target_tasks):
#     tmp = np.zeros((task_dim,1))
#     tmp[:-1,:] = np.random.uniform(low=0, high=1, size=(task_dim-1, 1))
#     target_task_dict.update({f"inside_target_{i}": (tmp, num_target)})

# 
source_task_dict = {}

# Test the random sampling strategy.
current_budget = 10000
strategy = RandomSampling()
tmp = strategy.select(task_embed_model, current_budget, seed=42)
source_task_dict.update(tmp)
print(source_task_dict)

# Test the dataset
dataset.generate_synthetic_data(source_task_dict)
dataset.generate_synthetic_data(target_task_dict)
# train_dataset = dataset.get_dataset(source_task_dict.keys(), mixed=True)
# print(train_dataset.meta_data) 

## Test the trainer (passive learning)

# Set test dataset
tmp = np.zeros((task_dim,1))
tmp[task_dim-1][0] = 1
target_task_dict_test = {"perp_single_target_test": (tmp, 1000)}
dataset.generate_synthetic_data(target_task_dict_test)

# with source
trainer_config = {"trainer_name":"pytorch_passive", "max_epoch": 100, "train_batch_size": 200, "lr": 0.01, "num_workers": 4,\
                  "optim_name": "SGD", "scheduler_name": "StepLR", "step_size": 100, "gamma": 0.9,
                  "test_batch_size": 500}
trainer_config = get_optimizer_fn(trainer_config)
trainer_config = get_scheduler_fn(trainer_config)
hidden_layers = [32, 32, embed_dim]
trainer_model = ModifiedShallow(input_dim, task_dim, hidden_layers, ret_emb = False)
trainer = PyTorchPassiveTrainer(trainer_config, trainer_model)
trainer.train(dataset, source_task_dict.keys())
trainer.train(dataset, target_task_dict.keys(), freeze_rep = True)
trainer.test(dataset, target_task_dict_test.keys())

# with source (actually I think it is better to do joint training)
trainer_config = {"trainer_name":"pytorch_passive", "max_epoch": 100, "train_batch_size": 200, "lr": 0.01, "num_workers": 4,\
                  "optim_name": "SGD", "scheduler_name": "StepLR", "step_size": 100, "gamma": 0.9,
                  "test_batch_size": 500}
trainer_config = get_optimizer_fn(trainer_config)
trainer_config = get_scheduler_fn(trainer_config)
hidden_layers = [32, 32, embed_dim]
trainer_model = ModifiedShallow(input_dim, task_dim, hidden_layers, ret_emb = False)
trainer = PyTorchPassiveTrainer(trainer_config, trainer_model)
trainer.train(dataset, list(source_task_dict.keys()) + list(target_task_dict.keys()))
trainer.test(dataset, target_task_dict_test.keys())


#without source
trainer_config = {"trainer_name":"pytorch_passive", "max_epoch": 10, "train_batch_size": 200, "lr": 0.01, "num_workers": 4,\
                  "optim_name": "SGD", "scheduler_name": "StepLR", "step_size": 100, "gamma": 0.9,
                  "test_batch_size": 500}
trainer_config = get_optimizer_fn(trainer_config)
trainer_config = get_scheduler_fn(trainer_config)
hidden_layers = [32, 32, embed_dim]
trainer_model = ModifiedShallow(input_dim, task_dim, hidden_layers, ret_emb = False)
trainer = PyTorchPassiveTrainer(trainer_config, trainer_model)
trainer.train(dataset, target_task_dict.keys(), freeze_rep = False)
trainer.test(dataset, target_task_dict_test.keys())

