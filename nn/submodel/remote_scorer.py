import torch
import torch.nn as nn

from nn.module import MLP, Biaffine


class Remote_Scorer(nn.Module):
    def __init__(self, lstm_dim, mlp_edge_dim, mlp_label_dim, n_label):
        super(Remote_Scorer, self).__init__()
        self.edge_head_mlp = MLP(lstm_dim*2, mlp_edge_dim, nn.ReLU())
        self.edge_dep_mlp = MLP(lstm_dim*2, mlp_edge_dim, nn.ReLU())
        self.label_head_mlp = MLP(lstm_dim*2, mlp_label_dim, nn.ReLU())
        self.label_dep_mlp = MLP(lstm_dim*2, mlp_label_dim, nn.ReLU())

        self.edge_biaffine = Biaffine(mlp_edge_dim, 1, bias_dep=True, bias_head=False)
        self.label_biaffine = Biaffine(mlp_label_dim, n_label, bias_dep=True, bias_head=True)

    def forward(self, span_vectors):
        edge_head_mlp_out = self.edge_head_mlp(span_vectors)
        edge_dep_mlp_out = self.edge_dep_mlp(span_vectors)
        label_head_mlp_out = self.label_head_mlp(span_vectors)
        label_dep_mlp_out = self.label_dep_mlp(span_vectors)

        edge_scores = self.edge_biaffine(edge_head_mlp_out, edge_dep_mlp_out)
        label_scores = self.label_biaffine(label_head_mlp_out, label_dep_mlp_out)
        return edge_scores, label_scores
