import torch.nn as nn
import torch.nn.init as init


class MLP(nn.Module):
    def __init__(self, input_size, layer_size, activation, drop=0):
        super(MLP, self).__init__()
        self.linear = nn.Linear(input_size, layer_size)
        self.activation = activation
        self.reset_parameters()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.activation:
            return self.drop(self.activation(self.linear(x)))
        else:
            return self.drop((self.linear(x)))

    def reset_parameters(self):
        init.orthogonal_(self.linear.weight)
