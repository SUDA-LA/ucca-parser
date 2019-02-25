import torch
import torch.nn as nn

from nn.module import Feedforward


class Chart_Span_Scorer(nn.Module):
    def __init__(self, lstm_dim, label_hidden_dim, n_label, drop):
        super(Chart_Span_Scorer, self).__init__()
        self.label_ffn = Feedforward(lstm_dim*2, label_hidden_dim, n_label-1, drop)

    def forward(self, span):
        label_score = self.label_ffn(span)
        return label_score


class Topdown_Span_Scorer(nn.Module):
    def __init__(self, lstm_dim, label_hidden_dim, split_hidden_dim, n_label, drop):
        super(Topdown_Span_Scorer, self).__init__()
        self.label_ffn = Feedforward(lstm_dim*2, label_hidden_dim, n_label, drop)
        self.split_ffn = Feedforward(lstm_dim*2, split_hidden_dim, 1, drop)

    def forward(self, span):
        label_score = self.label_ffn(span)
        split_score = self.split_ffn(span)
        return label_score, split_score
