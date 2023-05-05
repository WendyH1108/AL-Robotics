import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
import numpy as np
import torch

from dataset.dataset_skeleton import MyDataset, DatasetOnMemory

class SyntheticDataset(MyDataset):

    dataset_name = "synthetic"

    def __init__(self, input_dataset, input_embed_model, task_embed_model, model=None, noise_var = 0.0, batch_size= 100, num_workers=4):
        super(SyntheticDataset, self).__init__(batch_size, num_workers)

        assert (input_embed_model is not None and task_embed_model is not None) or model is not None, \
            "Either input_embed_model and task_embed_model or the combined model must be provided."
        self.input_dataset_pool = torch.Tensor(self.__preprocess(input_dataset))
        self.input_embed_model = input_embed_model
        self.task_embed_model = task_embed_model
        self.input_ind_sets = {} # If we already have a pool of input.
        self.model = model
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

    def generate_synthetic_data(self, task_dict, noise_var=None, seed=None):
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
                self.input_embed_model.cuda()
                self.task_embed_model.cuda()
                self.input_embed_model.eval()
                self.task_embed_model.eval()
            else:
                self.model.cuda()
                self.model.eval()
            labels = np.zeros((len(inputs), 1))
            counter = 0
            for i in range(0, len(inputs), self.batch_size):
                input = inputs[i: min(i+self.batch_size, len(inputs))].cuda()
                with torch.no_grad():
                    if self.model is None:
                        input_embed = self.input_embed_model(input)
                        if seed is not None:
                            torch.manual_seed(seed + np.random.RandomState(int(i*100)).randint(1000000000))
                        label = self.task_embed_model(input_embed, w) \
                            + torch.randn(input_embed.size(0), 1).cuda() * noise_var
                        
                    else:
                        label = self.model(input, w) \
                            + torch.randn(input.size(0), 1).cuda() * noise_var
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



