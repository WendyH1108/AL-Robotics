import torch
import torch.nn as nn

class ModifiedLinear(nn.Module):

    def __init__(self, embed_matrix):
        super(ModifiedLinear, self).__init__()
        num_input = embed_matrix.shape[0]
        num_output = embed_matrix.shape[1]
        model = nn.Linear(num_input, num_output)
        model.weight = nn.Parameter(torch.Tensor(embed_matrix).mT)
        model.bias = nn.Parameter(torch.zeros(num_output))
        self.model = model
        self.num_output = num_output
        self.num_input = num_input

    # TODO
    def forward(self, features, w=None, freeze=False):
        assert not freeze, "Nothing to freeze."
        if w is None:
            assert self.num_input > self.num_output, "This is not a input embedding matrix."
            return self.model(features)
        else:
            assert w is not None and len(w.shape) == 2, "w should be a 2-d matrix."
            assert w.shape[0] == self.num_output, "w should be of shape (num_output, ...)."
            assert w.shape[1] == 1 or w.shape[1] == len(features), "w should be of shape (..., num_input) or (..., 1)"
            if w.shape[1] == 1:
                labels = self.model(features) @ torch.Tensor(w).cuda()
            else:
                labels = torch.diag(self.model(features) @ torch.Tensor(w).cuda())
        return labels
    
    def get_full_task_embed_matrix(self):
        # Get the full embedding matrix, which is a d x d matrix.
        assert self.num_input < self.num_output, "This is not a task embedding matrix."
        return self.model.weight.mT.clone().detach().cpu().numpy()
    
    def get_restricted_task_embed_matrix(self):
        # Get embedding matrix restricted to the subspace spanned by the columns of proj_matrix.
        # Here when using source samples, we can only get a accurate estimation on the subspace spanned by the source samples.
        # TODO: a more general way ?
        assert self.num_input < self.num_output, "This is not a task embedding matrix."
        tmp = self.model.weight.mT.clone().detach().cpu().numpy()
        tmp[:, -1] = 0
        return tmp

    def get_input_dim(self):
        return self.num_input
    
    def get_output_dim(self):
        return self.num_output
    
