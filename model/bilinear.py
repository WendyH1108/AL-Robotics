import numpy as np
import torch
import torch.nn as nn

class ModifiedBiLinear(nn.Module):

    def __init__(self, num_input, num_output, embed_dim, ret_emb, bias=False):
        super(ModifiedBiLinear, self).__init__()
        self.linear1 = nn.Linear(num_input, embed_dim, bias = bias)
        # self.linear1.weight.data /= 10
        self.linear2 = nn.Linear(embed_dim, num_output, bias = bias)
        # self.linear2.weight.data /= 10
        # print(self.linear1.weight.data) #debug
        # print(self.linear2.weight.data) #debug
        self.num_output = num_output
        self.num_input = num_input
        self.embed_dim = embed_dim
        self.ret_emb = ret_emb

    def forward(self, features, w=None, ret_feat_and_label=False, freeze_rep=False, freeze_head=False):
        if freeze_rep:
            with torch.no_grad():
                features = self.linear1(features)
        else:
            features = self.linear1(features)

        if self.ret_emb:
            return features
        
        assert w is not None and len(w.shape) == 2, "w should be a 2-d matrix."
        assert w.shape[0] == self.num_output, "w should be of shape (num_output, ...)."
        assert w.shape[1] == 1 or w.shape[1] == len(features), "w should be of shape (..., num_input) or (..., 1)"
        if freeze_head:
            # TODO:
            with torch.no_grad():
                if w.shape[1] == 1:
                    labels = self.linear2(features) @ torch.Tensor(w).cuda()
                else:
                    labels = torch.diag(self.linear2(features) @ torch.Tensor(w).cuda())
        else:
            if w.shape[1] == 1:
                labels = self.linear2(features) @ torch.Tensor(w).cuda()
            else:
                labels = torch.diag(self.linear2(features) @ torch.Tensor(w).cuda())
        if ret_feat_and_label:
            return labels, features.data
        else:
            return labels

    def get_full_task_embed_matrix(self):
        # Get the full embedding matrix, which is a k x d matrix.
        return self.linear2.weight.mT.clone().detach().cpu().numpy()
    
    def get_restricted_task_embed_matrix(self):
        # Get embedding matrix restricted to the subspace spanned by the columns of proj_matrix.
        # Here when using source samples, we can only get a accurate estimation on the subspace spanned by the source samples.
        # TODO: a more general way ?
        tmp = self.linear2.weight.mT.clone().detach().cpu().numpy()
        tmp[:, -1] = 0
        return tmp
    
    def get_input_embed_matrix(self):
        return self.linear1.weight.mT.clone().detach().cpu().numpy()

    def get_input_dim(self):
        return self.num_input
    
    def get_output_dim(self):
        return self.num_output
    
    def update_input_embedding(self, input_embed_matrix):
        self.linear1.weight = nn.Parameter(torch.Tensor(input_embed_matrix).mT)
        self.linear1.bias = nn.Parameter(torch.zeros(self.embed_dim))

    def update_task_embedding(self, task_embed_matrix):
        self.linear2.weight = nn.Parameter(torch.Tensor(task_embed_matrix).mT)
        self.linear2.bias = nn.Parameter(torch.zeros(self.num_output))

    def get_embed_dim(self):
        return self.embed_dim


class ModifiedBiLinear_augmented(nn.Module):

    def __init__(self, num_input, num_output, embed_dim, ret_emb, bias=False):
        super(ModifiedBiLinear_augmented, self).__init__()
        self.linear1 = nn.Linear(num_input+1, embed_dim+1, bias = bias)
        # self.linear1.weight.data /= 10
        self.linear2 = nn.Linear(embed_dim+1, num_output+1, bias = bias)
        # self.linear2.weight.data /= 10
        # print(self.linear1.weight.data) #debug
        # print(self.linear2.weight.data) #debug
        self.num_output = num_output
        self.num_input = num_input
        self.embed_dim = embed_dim
        self.ret_emb = ret_emb

    def forward(self, features, w=None, ret_feat_and_label=False, freeze_rep=False, freeze_head=False):
        features = torch.cat([features, torch.ones((features.shape[0],1)).cuda()], dim=1)
        if freeze_rep:
            with torch.no_grad():
                features = self.linear1(features)
        else:
            features = self.linear1(features)

        if self.ret_emb:
            return features
                
        assert w is not None and len(w.shape) == 2, "w should be a 2-d matrix."
        assert w.shape[0] == self.num_output, f"w should be of shape ({self.num_output}, ...)."
        assert w.shape[1] == 1 or w.shape[1] == len(features), "w should be of shape (..., num_input) or (..., 1)"
        # aug_w = np.concatenate([w, np.ones((1,w.shape[1]))], axis=0)
        aug_w = torch.cat([w, torch.zeros((1, w.shape[1])).cuda()], dim=0)
        if freeze_head:
            # TODO:
            with torch.no_grad():
                if w.shape[1] == 1:
                    labels = self.linear2(features) @ torch.Tensor(aug_w).cuda()
                else:
                    labels = torch.diag(self.linear2(features) @ torch.Tensor(aug_w).cuda())
        else:
            if w.shape[1] == 1:
                labels = self.linear2(features) @ torch.Tensor(aug_w).cuda()
            else:
                labels = torch.diag(self.linear2(features) @ torch.Tensor(aug_w).cuda())
        if ret_feat_and_label:
            return labels, features.data
        else:
            return labels

    def get_full_task_embed_matrix(self):
        # Get the full embedding matrix, which is a k x d matrix.
        return self.linear2.weight.mT.clone().detach().cpu().numpy()[:,:-1]
    
    def get_restricted_task_embed_matrix(self):
        # Get embedding matrix restricted to the subspace spanned by the columns of proj_matrix.
        # Here when using source samples, we can only get a accurate estimation on the subspace spanned by the source samples.
        # TODO: a more general way ?
        tmp = self.linear2.weight.mT.clone().detach().cpu().numpy()[:,:-1]
        tmp[:, -1] = 0
        return tmp
    
    def get_input_embed_matrix(self):
        return self.linear1.weight.mT.clone().detach().cpu().numpy()

    def get_input_dim(self):
        return self.num_input
    
    def get_output_dim(self):
        return self.num_output
    
    def update_input_embedding(self, input_embed_matrix):
        self.linear1.weight = nn.Parameter(torch.Tensor(input_embed_matrix).mT)
        self.linear1.bias = nn.Parameter(torch.zeros(self.embed_dim))

    def update_task_embedding(self, task_embed_matrix):
        self.linear2.weight = nn.Parameter(torch.Tensor(task_embed_matrix).mT)
        self.linear2.bias = nn.Parameter(torch.zeros(self.num_output))

    def get_embed_dim(self):
        return self.embed_dim
