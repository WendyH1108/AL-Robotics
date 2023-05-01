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

    def __init__(self, input_dataset, input_embed_model, task_embed_model, model=None, noise_var = 0.0):

        assert (input_embed_model is not None and task_embed_model is not None) or model is not None, \
            "Either input_embed_model and task_embed_model or the combined model must be provided."
        self.input_dataset_pool = torch.Tensor(self.__preprocess(input_dataset))
        self.input_embed_model = input_embed_model
        self.task_embed_model = task_embed_model
        self.model = model
        self.sampled_train_tasks = {} 
        self.sampled_val_tasks = {}
        self.sampled_test_tasks = {}
        self.input_ind_sets = {}
        self.label_sets = {}
        self.batch_size = 100
        self.num_workers = 4
        self.default_noise_var = noise_var # Here we consider homogeneous noise for all the tasks.

    def __preprocess(self, dataset):
        """
        Preprocess the dataset.
        :param dataset: dataset to be preprocessed.
        :return: preprocessed dataset.
        """
        dataset_mean = np.mean(dataset, axis=0)
        dataset_std = np.std(dataset, axis=0)
        return (dataset - dataset_mean)/dataset_std

    def __generate_random_inputs(self, n, seed=None):
        """
        Generate random inputs.
        :param n: number of inputs to generate.
        :return: generated inputs.
        """
        np.random.seed(seed)
        selected_idx = np.random.choice(len(self.input_dataset_pool), n)
        return selected_idx, self.input_dataset_pool[selected_idx]

    def generate_synthetic_data(self, task_dict, noise_var=None, seed = None,device = "cpu"):
        """
        Generate synthetic data and stores into datasets.
        :param task_dict: dictionary of task name and the corresponding (w, n). 
            w is the weight vector for the task and n is the number of examples for the task. 
        :param float noise_var: variance of the noise added to the labels.
        """

        seed = np.random.RandomState(seed).randint(1000000000) if seed is not None else None
        noise_var = self.default_noise_var if noise_var is None else noise_var
        for task_name in task_dict:
            input_indices, inputs = self.__generate_random_inputs(task_dict[task_name][1], seed=seed)
            w = task_dict[task_name][0]
            if self.model is None:
                if device == "cuda":
                    self.input_embed_model.cuda()
                    self.task_embed_model.cuda()
                self.input_embed_model.eval()
                self.task_embed_model.eval()
            else:
                if device == "cuda":
                    self.model.cuda()
                self.model.eval()
            labels = np.zeros((len(inputs), 1))
            counter = 0
            for i in range(0, len(inputs), self.batch_size):
                if device == "cuda":
                    input = inputs[i: min(i+self.batch_size, len(inputs))].cuda()
                else:
                    input = inputs[i: min(i+self.batch_size, len(inputs))]
                with torch.no_grad():
                    if self.model is None:
                        if device == "cuda":
                            input_embed = self.input_embed_model(input)
                            label = self.task_embed_model(input_embed, w) \
                                + torch.randn(input_embed.size(0), 1).cuda() * noise_var
                        else:
                            input_embed = self.input_embed_model(input)
                            label = self.task_embed_model(input_embed, w) \
                                + torch.randn(input_embed.size(0), 1) * noise_var
                    else:
                        if device == "cuda":
                            label = self.model(input, w) \
                                + torch.randn(input.size(0), 1).cuda() * noise_var
                        else:
                            label = self.model(input, w) \
                                + torch.randn(input.size(0), 1) * noise_var
                    labels[counter: (counter + len(label))] = label.data.cpu().numpy()
                counter += len(label)
                # print(label) #debug

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
            elif "val" in task_name:
                self.sampled_val_tasks.update({task_name: w})
            else:
                self.sampled_train_tasks.update({task_name: w})

    def generate_val_data(self, budget = 200):
        """
        Generate a few shot validation data.
        """
        task_name_list = self.sampled_train_tasks.keys()
        val_task_dict = {}
        for task_name in task_name_list:
            if task_name+"_val" not in self.sampled_val_tasks:
                val_task_dict[task_name+"_val"] = (self.sampled_train_tasks[task_name], budget)
        self.generate_synthetic_data(val_task_dict, noise_var=0.0)



    def get_dataset(self, task_name_list, mixed):
        """
        Get dataset for the task.
        :param str task_name: name of the task
        :param bool mixed: whether to mix the data from different tasks.
        :return: dataset for the tasks.
        """

        task_dim = self.task_embed_model.get_output_dim() if self.model is None else self.model.get_output_dim()
        if mixed:
            total_input_indices = []
            total_labels = []
            for task_name in task_name_list:
                total_input_indices.extend(self.input_ind_sets[task_name])
                total_labels.extend(self.label_sets[task_name])
            total_ws = np.empty((len(total_labels), task_dim))
            counter = 0
            for task_name in task_name_list:
                if "test" in task_name:
                    total_ws[counter: (counter + len(self.label_sets[task_name])),:] = self.sampled_test_tasks[task_name].T
                elif "val" in task_name:
                    total_ws[counter: (counter + len(self.label_sets[task_name])),:] = self.sampled_val_tasks[task_name].T
                else:
                    total_ws[counter: (counter + len(self.label_sets[task_name])),:] = self.sampled_train_tasks[task_name].T
                counter += len(self.label_sets[task_name])
            output = DatasetOnMemory(self.input_dataset_pool[total_input_indices], total_labels, total_ws)
        else:
            output = {}
            for task_name in task_name_list:
                assert (task_name in self.input_ind_sets) and (task_name in self.label_sets), \
                    "Dataset for task {} does not exist. Please generate first".format(task_name)
                total_ws = np.empty((len(self.label_sets[task_name]), task_dim))
                if "test" in task_name:
                    total_ws[:,:] = self.sampled_test_tasks[task_name].T
                elif "val" in task_name:
                    total_ws[:,:] = self.sampled_val_tasks[task_name].T
                else:
                    total_ws[:,:] = self.sampled_train_tasks[task_name].T 
                output[task_name] = DatasetOnMemory(self.input_dataset_pool[self.input_ind_sets[task_name]], self.label_sets[task_name], total_ws)
        return output
    
    def delete_dataset(self, task_name):
        """
        Delete dataset for the task.
        :param task_name: name of the task.
        """
        if task_name in self.input_ind_sets:
            del self.input_ind_sets[task_name]
        if task_name in self.label_sets:
            del self.label_sets[task_name]
        if task_name in self.sampled_test_tasks:
            del self.sampled_test_tasks[task_name]
        if task_name in self.sampled_train_tasks:
            del self.sampled_train_tasks[task_name]

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
    
class SimuDataset:
    """
    Dataset for active learning. The dataset contains all of training, validation and testing data as well as their
    embeddings. The dataset also tracks the examples that have been labeled.
    """

    def __init__(self, input_data, input_label, input_ws):

        self.input_data = self.__preprocess(input_data)
        self.input_label = input_label
        self.input_ws = input_ws
        self.sampled_train_tasks = {} 
        self.sampled_test_tasks = {}
        self.label_sets = {}
        self.batch_size = 100
        self.num_workers = 4

    def __preprocess(self, dataset):
        """
        Preprocess the dataset.
        :param dataset: dataset to be preprocessed.
        :return: preprocessed dataset.
        """
        for task in dataset.keys():
            dataset_mean = np.mean(dataset[task], axis=0)
            dataset_std = np.std(dataset[task], axis=0)
            dataset[task] = torch.Tensor(dataset[task] - dataset_mean)/dataset_std
        return dataset

    def generate_random_inputs(self, total_len, n, seed=None):
        """
        Generate random inputs.
        :param n: number of inputs to generate.
        :return: generated inputs.
        """
        if seed is not None:
            np.random.seed(seed)
        selected_idx = np.random.choice(total_len, n)
        return selected_idx


    def get_dataset(self, task_dict, mixed):
        """
        Get dataset for the task.
        :param str task_name: name of the task
        :param bool mixed: whether to mix the data from different tasks.
        :return: dataset for the tasks.
        """
        task_name_list = task_dict.keys()
        if mixed:
            input_data = []
            input_label = []
            input_ws = []
            for task_name in task_name_list:
                w = task_dict[task_name][0]
                if "test" in task_name:
                    self.sampled_test_tasks.update({task_name: w})
                else:
                    self.sampled_train_tasks.update({task_name: w})
                n = task_dict[task_name][1]
                idx = self.generate_random_inputs(self.input_data[task_name].shape[0], n)
                input_data += self.input_data[task_name][idx]
                input_label += [self.input_label[task_name][idx]]
                input_ws += torch.tensor(self.input_ws[task_name])[idx]
            output = DatasetOnMemory(input_data, np.array(input_label).flatten(), input_ws)
        else:
            output = {}
            for task_name in task_name_list:
                w = task_dict[task_name][0]
                if "test" in task_name:
                    self.sampled_test_tasks.update({task_name: w})
                else:
                    self.sampled_train_tasks.update({task_name: w})
                n = task_dict[task_name][1]
                idx = self.generate_random_inputs(self.input_data[task_name].shape[0], n)
                output[task_name] = DatasetOnMemory(self.input_data[task_name][idx], 
                                                    self.input_label[task_name][idx], 
                                                    self.input_ws[task_name][idx])

        return output
    
    def delete_dataset(self, task_name):
        """
        Delete dataset for the task.
        :param task_name: name of the task.
        """
        if task_name in self.input_ind_sets:
            del self.input_ind_sets[task_name]
        if task_name in self.label_sets:
            del self.label_sets[task_name]
        if task_name in self.sampled_test_tasks:
            del self.sampled_test_tasks[task_name]
        if task_name in self.sampled_train_tasks:
            del self.sampled_train_tasks[task_name]

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
