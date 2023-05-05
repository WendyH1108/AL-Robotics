import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn


from model.shallow import ModifiedShallow
from trainer.trainer import *



class PyTorchPassiveTrainer():
    trainer_name = "pytorch_passive"

    def __init__(self, trainer_config, model, task_aug_kernel = None):
        super(PyTorchPassiveTrainer, self).__init__()
        self.trainer_config = trainer_config
        self.model = model
        assert isinstance(self.model, nn.Module), "Only support nn.Module for now."
        self.task_aug_kernel = task_aug_kernel

    def train(self, dataset, train_task_name_list, freeze_rep = False, shuffle=True, need_print = False, seed = None):
        """
        Train on the given source tasks in task_name_list
        :param shuffle: whether to shuffle the training data (if not, then solve task one by one)
        """

        # Get the training dataset based on the task_name_list.
        train_dataset = dataset.get_dataset(train_task_name_list, mixed=True)
        print(f"Training on {len(train_dataset)} samples.")
        # Set various parameters.
        max_epoch = self.trainer_config["max_epoch"]
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = self.trainer_config["optim_fn"](params)
        total_steps = self.trainer_config["max_epoch"] * len(train_dataset) // self.trainer_config["train_batch_size"]
        scheduler = self.trainer_config["scheduler_fn"](optimizer, total_steps) \
            if "scheduler_fn" in self.trainer_config else None
        
        self.model.cuda()
        counter = 0
        for epoch in tqdm(range(max_epoch), desc="Training: "):

            loader = DataLoader(train_dataset, batch_size=self.trainer_config["train_batch_size"], shuffle=shuffle,
                                    num_workers=self.trainer_config["num_workers"],
                                    drop_last=(len(train_dataset) >= self.trainer_config["train_batch_size"]))

            total_Loss = 0
            for input, label, w in loader:
                input, label = input.float().cuda(), label.float().cuda()
                # print(w[0]) #debug
                if self.task_aug_kernel is not None:
                    w = torch.from_numpy(self.task_aug_kernel(w.numpy()))
                w = w.float().cuda()
                # print(w[0]) #debugs

                pred = self.model(input, w.mT, freeze_rep = freeze_rep, ret_feat_and_label=False)
                if scheduler is not None:
                    scheduler(counter)
                    counter += 1
                optimizer.zero_grad()
                loss = F.mse_loss(pred, label)
                # print("label and loss",label, loss) #debug
                total_Loss += loss.item()*len(input)
                loss.backward()
                if "clip_grad" in self.trainer_config:
                    nn.utils.clip_grad_norm_(params, self.trainer_config["clip_grad"])
                optimizer.step()
            if need_print:
                print('Train Epoch: {} [total loss on on : {:.6f}] with lr {:.3f}'.format(epoch,  total_Loss/len(train_dataset), optimizer.param_groups[0]['lr'])) 
        print('Finish training after epoch: {} [total loss on : {:.6f}]'.format(epoch, total_Loss/len(train_dataset))) 

    def test(self, dataset, test_task_name_list, output_type = "max"):
        """
        Train on the given source tasks in task_name_list
        """
        self.model.cuda()
        self.model.eval()

        # Get the training dataset based on the task_name_list.

        if output_type == "max":        
            test_dataset = dataset.get_dataset(test_task_name_list, mixed=False)
            max_loss = 0
            max_loss_task = None
            for task_name in test_task_name_list:
                loader = DataLoader(test_dataset[task_name], batch_size=self.trainer_config["test_batch_size"], shuffle=False,
                                    num_workers=self.trainer_config["num_workers"])
                total_Loss = 0
                for input, label, w in loader:
                    input, label = input.float().cuda(), label.float().cuda()
                    if self.task_aug_kernel is not None:
                        w = torch.from_numpy(self.task_aug_kernel(w.numpy()))
                    w = w.float().cuda()
                    with torch.no_grad():
                        pred = self.model(input, w.mT, ret_feat_and_label=False)
                        loss = F.mse_loss(pred.float(), label.float(), reduction='sum')
                        total_Loss += loss.item()
                if max_loss < total_Loss/len(test_dataset[task_name]):
                    max_loss = total_Loss/len(test_dataset[task_name])
                    max_loss_task = task_name
                # print('[total test loss on {}: {:.6f}]'.format(task_name, total_Loss/len(test_dataset[task_name]))) 
            print('[Worst-case test loss on {}: {:.6f}]'.format(max_loss_task, total_Loss/len(test_dataset[max_loss_task]))) 
        else:
            test_dataset = dataset.get_dataset(test_task_name_list, mixed=True)
            loader = DataLoader(test_dataset, batch_size=self.trainer_config["test_batch_size"], shuffle=False,
                                            num_workers=self.trainer_config["num_workers"])
            total_Loss = 0
            print(f"Testing on {len(test_dataset)} samples.")
            print(test_dataset)
            for input, label, w in loader:
                input, label = input.float().cuda(), label.float().cuda()
                if self.task_aug_kernel is not None:
                    w = torch.from_numpy(self.task_aug_kernel(w.numpy()))
                w = w.float().cuda()
                with torch.no_grad():
                    pred = self.model(input, w.mT, ret_feat_and_label=False)
                    loss = F.mse_loss(pred.float(), label.float(), reduction='sum')
                    total_Loss += loss.item()
            print('[total test loss : {:.6f}]'.format(total_Loss/len(test_dataset))) 

        self.model.train()
        return total_Loss/len(test_dataset[max_loss_task]) if output_type == "max" else total_Loss/len(test_dataset)

    def update_config(self, trainer_config):
        self.trainer_config = trainer_config