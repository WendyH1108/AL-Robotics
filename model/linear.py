import torch
import torch.nn as nn

class ModifiedLinear(nn.Module):

    def __init__(self, embed_matrix):
        super(ModifiedLinear, self).__init__()
        num_input = embed_matrix.shape[0]
        num_output = embed_matrix.shape[1]
        assert num_input < num_output, "Embed matrix should be k x d, where k < d. k is the low-dimensional embedding."
        model = nn.Linear(num_input, num_output)
        model.weight = nn.Parameter(torch.Tensor(embed_matrix).mT)
        model.bias = nn.Parameter(torch.zeros(num_output))
        self.model = model
        self.num_output = num_output
        self.num_input = num_input

    # TODO
    def forward(self, features, w):
        assert len(w.shape) == 2, "w should be a 2-d matrix."
        assert w.shape[0] == self.model.weight.shape[0], "w should have the same number of rows as the number of output features."
        return torch.diag(self.model(features) @ torch.Tensor(w).cuda())
    
    def get_full_task_embed_matrix(self):
        # Get the full embedding matrix, which is a d x d matrix.
        return self.model.weight.mT.clone().detach().cpu().numpy()
    
    def get_restricted_task_embed_matrix(self):
        # Get embedding matrix restricted to the subspace spanned by the columns of proj_matrix.
        # Here when using source samples, we can only get a accurate estimation on the subspace spanned by the source samples.
        # TODO: a more general way ?
        tmp = self.model.weight.mT.clone().detach().cpu().numpy()
        tmp[:, -1] = 0
        return tmp

    def get_input_dim(self):
        return self.num_input
    
    def get_output_dim(self):
        return self.num_output