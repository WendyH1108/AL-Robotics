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

    def __init__(self, trainer_config, model):
        super(PyTorchPassiveTrainer, self).__init__()
        self.trainer_config = trainer_config
        self.model = model
        assert isinstance(self.model, nn.Module), "Only support nn.Module for now."

    def train(self, dataset, train_task_name_list, freeze_rep = False, need_print = False):
        """
        Train on the given source tasks in task_name_list
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
            loader = DataLoader(train_dataset, batch_size=self.trainer_config["train_batch_size"], shuffle=True,
                                num_workers=self.trainer_config["num_workers"],
                                drop_last=(len(train_dataset) >= self.trainer_config["train_batch_size"]))

            total_Loss = 0
            for input, label, w in loader:
                input, label, w = input.float().cuda(), label.float().cuda(), w.float().cuda()
                pred = self.model(input, w.mT, freeze = freeze_rep, ret_feat_and_label=False)
                if scheduler is not None:
                    scheduler(counter)
                    counter += 1
                optimizer.zero_grad()
                loss = F.mse_loss(pred.float(), label.float())
                total_Loss += loss.item()*len(input)
                loss.backward()
                if "clip_grad" in self.trainer_config:
                    nn.utils.clip_grad_norm_(params, self.trainer_config["clip_grad"])
                optimizer.step()
            if need_print:
                print('Train Epoch: {} [total loss on on : {:.6f}] with lr {:.3f}'.format(epoch,  total_Loss/len(train_dataset), optimizer.param_groups[0]['lr'])) 
        print('Finish training after epoch: {} [total loss on : {:.6f}]'.format(epoch, total_Loss/len(train_dataset))) 

    def test(self, dataset, test_task_name_list):
        """
        Train on the given source tasks in task_name_list
        """
        self.model.cuda()
        # Get the training dataset based on the task_name_list.
        test_dataset = dataset.get_dataset(test_task_name_list, mixed=True)

        loader = DataLoader(test_dataset, batch_size=self.trainer_config["test_batch_size"], shuffle=False,
                            num_workers=self.trainer_config["num_workers"])
        self.model.eval()
        total_Loss = 0
        for input, label, w in loader:
            input, label, w = input.float().cuda(), label.float().cuda(), w.float().cuda()
            with torch.no_grad():
                pred = self.model(input, w.mT, ret_feat_and_label=False)
                loss = F.mse_loss(pred.float(), label.float())
                total_Loss += loss.item()*len(input)
        print('[total test loss on {}: {:.6f}]'.format(test_task_name_list, total_Loss/len(test_dataset))) 
        self.model.train()

    def update_config(self, trainer_config):
        self.trainer_config = trainer_config