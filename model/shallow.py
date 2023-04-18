import torch
import torch.nn as nn
from torchvision.ops import MLP


class ModifiedShallow(nn.Module):

    def __init__(self, num_input, num_output, hidden_layers, ret_emb):
        super(ModifiedShallow, self).__init__()
        self.shallow_model = MLP(in_channels=num_input, hidden_channels=hidden_layers, norm_layer=None, dropout=0.0)
        self.linear = nn.Linear(hidden_layers[-1], num_output)
        self.num_output = num_output
        self.num_input = num_input
        self.ret_emb = ret_emb

    def forward(self, features, w=None, ret_feat_and_label=False, freeze=False):
        if freeze:
            with torch.no_grad():
                features = self.shallow_model(features)
        else:
            features = self.shallow_model(features)

        if self.ret_emb:
            return features
        
        assert w is not None and len(w.shape) == 2, "w should be a 2-d matrix."
        assert w.shape[0] == self.num_output, "w should be of shape (num_output, ...)."
        if ret_feat_and_label:
            return torch.diag(self.linear(features) @ torch.Tensor(w).cuda()), features.data
        else:
            return torch.diag(self.linear(features) @ torch.Tensor(w).cuda())
        
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
    
    def get_output_dim(self):
        return self.num_output
    

    