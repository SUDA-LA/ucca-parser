import torch.nn as nn
import torch.nn.init as init


class Feedforward(nn.Module):
    def __init__(self, input_size, hidden_dim, out_dim, drop):
        super(Feedforward, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)
        self.reset_parameters()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        return self.linear2(self.drop(x))

    def reset_parameters(self):
        init.orthogonal_(self.linear1.weight)
        init.orthogonal_(self.linear1.weight)
