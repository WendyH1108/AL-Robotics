import torch
from enum import Enum
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
import numpy as np
import torch
from tqdm import tqdm

class LabelType(Enum):
    """Formats of label."""
    MULTI_CLASS = 1
    MULTI_LABEL = 2


datasets = {}


def register_dataset(name: str, type: LabelType):
    """
    Register dataset with dataset name and label type.
    :param str name: dataset name.
    :param LabelType type: the type of label for the dataset.
    :return: function decorator that registers the dataset.
    """

    def dataset_decor(get_fn):
        datasets[name] = (type, get_fn)
        return get_fn

    return dataset_decor


class DatasetOnMemory(Dataset):
    """
    A PyTorch dataset where all data lives on CPU memory.
    """

    def __init__(self, X, y, meta_data=None):
        assert len(X) == len(y), "X and y must have the same length."
        assert meta_data is None or len(X) == len(meta_data), "X and meta_data must have the same length."
        self.X = X
        self.y = y
        self.meta_data = meta_data 

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x = self.X[item]
        y = self.y[item]
        if self.meta_data is not None:
            meta_data = self.meta_data[item]
            return x, y, meta_data
        else:
            return x, y, None

    def get_inputs(self):
        return self.X

    def get_labels(self):
        return self.y


#TODO: Here I consider all the inputs data are shared across all the tasks. Further modification is needed if we want to have different inputs domain for different tasks.
class SyntheticDataset:
    """
    Dataset for active learning. The dataset contains all of training, validation and testing data as well as their
    embeddings. The dataset also tracks the examples that have been labeled.
    """

    def __init__(self, input_dataset, input_embed_model, task_embed_model, noise_var = 0.0):

        #TODO: shared inputs
        self.input_dataset_pool = torch.Tensor(input_dataset)
        self.input_embed_model = input_embed_model
        self.task_embed_model = task_embed_model
        self.sampled_train_tasks = {} 
        self.sampled_test_tasks = {}
        self.input_ind_sets = {}
        self.label_sets = {}
        self.batch_size = 100
        self.num_workers = 4
        self.noise_var = noise_var # Here we consider homogeneous noise for all the tasks.

    def generate_random_inputs(self, n, seed=None):
        """
        Generate random inputs.
        :param n: number of inputs to generate.
        :return: generated inputs.
        """
        if seed is not None:
            np.random.seed(seed)
        selected_idx = np.random.choice(len(self.input_dataset_pool), n)
        return selected_idx, self.input_dataset_pool[selected_idx]

    def generate_synthetic_data(self, task_dict):
        """
        Generate synthetic data and stores into datasets.
        :param task_dict: dictionary of task name and the corresponding (w, n). 
            w is the weight vector for the task and n is the number of examples for the task. 
        :param float noise_var: variance of the noise added to the labels.
        """

        for task_name in task_dict:
            input_indices, inputs = self.generate_random_inputs(task_dict[task_name][1], seed=None)
            w = task_dict[task_name][0]
            self.input_embed_model.cuda()
            self.task_embed_model.cuda()
            self.input_embed_model.eval()
            self.task_embed_model.eval()
            labels = np.zeros((len(inputs), 1))
            counter = 0
            for i in tqdm(range(0, len(inputs), self.batch_size), desc='Generating data for task: '.format(task_name)):
                input = inputs[i: min(i+self.batch_size, len(inputs))].cuda()
                with torch.no_grad():
                    input_embed = self.input_embed_model(input)
                    label = self.task_embed_model(input_embed, w) \
                        + torch.randn(input_embed.size(0), 1).cuda() * self.noise_var
                    labels[counter: (counter + len(label))] = label.data.cpu().numpy()
                counter += len(label)

            # Store the generated data if the corresponding task does not exist.
            # Otherwise concatenate the new data to the existing data.
            # Note that the input indices are also stored for the task instead of the real input data.
            if task_name in self.input_ind_sets:
                self.input_ind_sets[task_name].extend(input_indices.tolist())
            else:
                self.input_ind_sets[task_name] = input_indices.tolist()

            if task_name in self.label_sets:
                self.label_sets[task_name].extend(labels.squeeze().tolist())
            else:
                self.label_sets[task_name] = labels.squeeze().tolist()

            # Also update the sampled tasks dictionary to store all the sampled tasks with name and parameter.
            if "test" in task_name:
                self.sampled_test_tasks.update({task_name: w})
            else:
                self.sampled_train_tasks.update({task_name: w})

    def get_dataset(self, task_name_list, mixed):
        """
        Get dataset for the task.
        :param str task_name: name of the task
        :param bool mixed: whether to mix the data from different tasks.
        :return: dataset for the tasks.
        """

        if mixed:
            total_input_indices = []
            total_labels = []
            for task_name in task_name_list:
                total_input_indices.extend(self.input_ind_sets[task_name])
                total_labels.extend(self.label_sets[task_name])
            total_ws = np.empty((len(total_labels), self.task_embed_model.get_output_dim()))
            counter = 0
            for task_name in task_name_list:
                try:
                    total_ws[counter: (counter + len(self.label_sets[task_name])),:] = self.sampled_train_tasks[task_name].T  
                except:
                    total_ws[counter: (counter + len(self.label_sets[task_name])),:] = self.sampled_test_tasks[task_name].T
                counter += len(self.label_sets[task_name])
            output = DatasetOnMemory(self.input_dataset_pool[total_input_indices], total_labels, total_ws)
        else:
            output = {}
            for task_name in task_name_list:
                assert (task_name in self.input_ind_sets) and (task_name in self.label_sets), \
                    "Dataset for task {} does not exist. Please generate first".format(task_name)
                output[task_name] = DatasetOnMemory(self.input_dataset_pool[self.input_ind_sets[task_name]], self.label_sets[task_name])
        return output
    
    def delete_dataset(self, task_name):
        """
        Delete dataset for the task.
        :param task_name: name of the task.
        """
        if task_name in self.input_ind_sets:
            del self.input_sets[task_name]
        if task_name in self.label_sets:
            del self.label_sets[task_name]
        if task_name in self.sampled_tasks:
            del self.sampled_tasks[task_name]

    def get_sampled_train_tasks(self):
        """
        Get the sampled train tasks.
        :return: tasks.
        """
        return self.sampled_train_tasks
    
    def get_sampled_test_tasks(self):
        """
        Get the sampled test tasks.
        :return: tasks.
        """
        return self.sampled_test_tasks
    

