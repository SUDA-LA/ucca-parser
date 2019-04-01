import torch.nn as nn
import torch.nn.init as init


class Feedforward(nn.Module):
    def __init__(self, input_size, hidden_dim, out_dim, drop=0, norm=False):
        super(Feedforward, self).__init__()
        self.norm = norm
        self.linear1 = nn.Linear(input_size, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)
        if norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)

        self.reset_parameters()

    def forward(self, x):
        x = self.linear1(x)
        if self.norm:
            x = self.layer_norm(x)
        x = self.activation(x)
        return self.linear2(self.drop(x))

    def reset_parameters(self):
        init.orthogonal_(self.linear1.weight)
        init.orthogonal_(self.linear2.weight)
