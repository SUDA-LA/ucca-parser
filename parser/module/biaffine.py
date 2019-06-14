import torch
import torch.nn as nn


class Biaffine(nn.Module):
    """
    BiAffine Attention layer from https://arxiv.org/abs/1611.01734
    Expects inputs as batch-first sequences [batch_size, seq_length, dim].

    Returns score matrices as [batch_size, dim, dim] for arc attention
    (out_channels=1), and score as [batch_size, out_channels, dim, dim]
    for label attention (where out_channels=#labels).
    """

    def __init__(self, in_dim, out_channels, bias_head=True, bias_dep=True):
        super(Biaffine, self).__init__()
        self.bias_head = bias_head
        self.bias_dep = bias_dep
        self.U = nn.Parameter(torch.Tensor(out_channels,
                                           in_dim + int(bias_head),
                                           in_dim + int(bias_dep)))
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / self.U.size(1)**0.5
        # self.U.data.uniform_(-stdv, stdv)
        self.U.data.zero_()

    def forward(self, Rh, Rd):
        """
        Returns S = (Rh @ U @ Rd.T) with dims [batchsize, n_channels, t, t]
        S[b, c, i, j] = Score sample b Label c Head i Dep j
        """

        if self.bias_head:
            Rh = self.add_ones_col(Rh)
        if self.bias_dep:
            Rd = self.add_ones_col(Rd)

        # Add dimension to Rh and Rd for batch matrix products,
        # shape [batch, t, d] -> [batch, 1, t, d]
        Rh = Rh.unsqueeze(1)
        Rd = Rd.unsqueeze(1)

        S = Rh @ self.U @ torch.transpose(Rd, -1, -2)
        # If out_channels == 1, squeeze [batch, 1, t, t] -> [batch, t, t]
        return S.squeeze(1)

    @staticmethod
    def add_ones_col(X):
        """
        Add column of ones to each matrix in batch.
        """
        batch_size, len, dim = X.size()
        b = X.new_ones((batch_size, len, 1), requires_grad=True)
        return torch.cat([X, b], -1)

    def __repr__(self):
        tmpstr = self.__class__.__name__
        tmpstr += '(\n  (U): {}\n)'.format(self.U.size())
        return tmpstr
