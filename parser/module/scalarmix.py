import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, ParameterList


class ScalarMix(torch.nn.Module):

    def __init__(self, elmo_dim=1024, mixture_size=3, do_layer_norm=False):
        super(ScalarMix, self).__init__()
        self.elmo_dim = elmo_dim
        self.mixture_size = mixture_size
        self.do_layer_norm = do_layer_norm
        self.scalar_parameters = Parameter(torch.zeros(mixture_size))
        self.gamma = Parameter(torch.tensor(1.0))

    def _do_layer_norm(self, tensor, broadcast_mask, num_elements_not_masked):
        tensor_masked = tensor * broadcast_mask
        mean = torch.sum(tensor_masked) / num_elements_not_masked
        variance = torch.sum(
            ((tensor_masked - mean) * broadcast_mask)**2) / num_elements_not_masked
        return (tensor - mean) / torch.sqrt(variance + 1E-12)

    def forward(self, elmos, mask=None):
        normed_weights = F.softmax(self.scalar_parameters, dim=0)
        pieces = []
        if not self.do_layer_norm:
            return self.gamma * sum(weight * tensor.squeeze(2) for weight, tensor in zip(normed_weights, elmos))
        else:
            mask_float = mask.float()
            broadcast_mask = mask_float.unsqueeze(-1)
            input_dim = self.elmo_dim
            num_elements_not_masked = torch.sum(mask_float) * input_dim
            return self.gamma * sum(weight * self._do_layer_norm(tensor.squeeze(2), broadcast_mask, num_elements_not_masked) for weight, tensor in zip(normed_weights, elmos))
