import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from pytorch_pretrained_bert.modeling import BertModel

from .scalarmix import ScalarMix


class Bert_Embedding(nn.Module):
    def __init__(self, bert_path, bert_layer, bert_dim, freeze=True):
        super(Bert_Embedding, self).__init__()
        self.bert_layer = bert_layer
        self.bert = BertModel.from_pretrained(bert_path)
        self.scalar_mix = ScalarMix(bert_dim, bert_layer[-1] - bert_layer[0])

        if freeze:
            self.freeze()

    def forward(self, subword_idxs, subword_masks, token_starts_masks):
        sen_lens = token_starts_masks.sum(dim=1)
        bert_outs, _ = self.bert(
            subword_idxs,
            token_type_ids=None,
            attention_mask=subword_masks,
            output_all_encoded_layers=True,
        )
        bert_outs = bert_outs[self.bert_layer[0] : self.bert_layer[-1]]
        bert_outs = self.scalar_mix(bert_outs)
        bert_outs = torch.split(bert_outs[token_starts_masks], sen_lens.tolist())
        bert_outs = pad_sequence(bert_outs, batch_first=True)
        return bert_outs

    def freeze(self):
        for para in self.bert.parameters():
            para.requires_grad = False


class Bert_Encoder(nn.Module):
    def __init__(self, bert_path, bert_dim, freeze=False):
        super(Bert_Encoder, self).__init__()
        self.bert_dim = bert_dim
        self.bert = BertModel.from_pretrained(bert_path)

        if freeze:
            self.freeze()

    def forward(self, subword_idxs, subword_masks, token_starts_masks):
        sen_lens = token_starts_masks.sum(dim=1)
        bert_outs, _ = self.bert(
            subword_idxs,
            token_type_ids=None,
            attention_mask=subword_masks,
            output_all_encoded_layers=False,
        )
        bert_outs = torch.split(bert_outs[token_starts_masks], sen_lens.tolist())
        bert_outs = pad_sequence(bert_outs, batch_first=True)
        return bert_outs

    def freeze(self):
        for para in self.bert.parameters():
            para.requires_grad = False