import torch
import torch.nn as nn
from torchvision.ops import MLP


class ModifiedShallow(nn.Module):
    #TODO: check bias

    def __init__(self, num_input, num_output, hidden_layers, ret_emb, seed = 0 ,dropout=0.0):
        super(ModifiedShallow, self).__init__()
        self.shallow_model = MLP(in_channels=num_input, hidden_channels=hidden_layers, norm_layer=None, dropout=dropout, bias = False)
        self.linear = nn.Linear(hidden_layers[-1], num_output, bias = False)
        self.num_output = num_output
        self.num_input = num_input
        self.embed_dim = hidden_layers[-1]
        self.ret_emb = ret_emb

    def forward(self, features, w=None, ret_feat_and_label=False, freeze_rep=False, freeze_head=False, device = "cpu"):
        if freeze_rep:
            with torch.no_grad():
                features = self.shallow_model(features)
        else:
            features = self.shallow_model(features)

        if self.ret_emb:
            return features
        
        assert w is not None and len(w.shape) == 2, "w should be a 2-d matrix."
        assert w.shape[0] == self.num_output, "w should be of shape (num_output, ...)."
        assert w.shape[1] == 1 or w.shape[1] == len(features), "w should be of shape (..., num_input) or (..., 1)"
        if freeze_head:
            with torch.no_grad():
                if w.shape[1] == 1:
                    if device == "cuda":
                        labels = self.linear2(features) @ torch.Tensor(w).cuda()
                    else:
                        labels = self.linear2(features) @ torch.Tensor(w)
                else:
                    if device == "cuda":
                        labels = torch.diag(self.linear2(features) @ torch.Tensor(w).cuda())
                    else:
                        labels = torch.diag(self.linear2(features) @ torch.Tensor(w))
        else:
            if w.shape[1] == 1:
                if device == "cuda":
                    labels = self.linear(features) @ torch.Tensor(w).cuda()
                else:
                    labels = self.linear(features) @ torch.Tensor(w)
            else:
                if device == "cuda":
                    labels = torch.diag(self.linear(features) @ torch.Tensor(w).cuda())
                else:
                    labels = torch.diag(self.linear(features) @ torch.Tensor(w))
        if ret_feat_and_label:
            return labels, features.data
        else:
            return labels
        
    def get_full_task_embed_matrix(self):
        # Get the full embedding matrix, which is a d x d matrix.
        return self.linear.weight.mT.clone().detach().cpu().numpy()
    
    def get_restricted_task_embed_matrix(self):
        # Get embedding matrix restricted to the subspace spanned by the columns of proj_matrix.
        # Here when using source samples, we can only get a accurate estimation on the subspace spanned by the source samples.
        # TODO: a more general way ?
        tmp = self.linear.weight.mT.clone().detach().cpu().numpy()
        tmp[:, -1] = 0
        return tmp

    def get_input_dim(self):
        return self.num_input
    
    def get_embed_dim(self):
        return self.embed_dim
    
    def get_output_dim(self):
        return self.num_output
    
    def update_input_embedding(self, input_embed_matrix):
        raise NotImplementedError("Not implemented yet.")

    def update_task_embedding(self, task_embed_matrix):
        self.linear.weight = nn.Parameter(torch.Tensor(task_embed_matrix).mT)
        self.linear.bias = nn.Parameter(torch.zeros(self.num_output))

    

    