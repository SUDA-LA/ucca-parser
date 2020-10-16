import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import (pack_padded_sequence, pack_sequence,
                                pad_packed_sequence, pad_sequence)

from ..module import CharLSTM, EncoderLayer, PositionEncoding


class LSTM_Encoder(nn.Module):
    def __init__(
        self,
        vocab,
        ext_emb,
        word_dim,
        char_dim,
        charlstm_dim,
        lstm_dim,
        lstm_layer,
        emb_drop=0.5,
        lstm_drop=0.4,
        char_drop=0,
    ):
        super(LSTM_Encoder, self).__init__()
        self.vocab = vocab
        self.ext_word_embedding = nn.Embedding.from_pretrained(ext_emb)
        self.word_embedding = nn.Embedding(
            vocab.num_train_word, word_dim, padding_idx=0)

        self.charlstm = CharLSTM(
            vocab.num_char, char_dim, charlstm_dim, char_drop)

        self.lstm = nn.LSTM(
            input_size=word_dim + charlstm_dim,
            hidden_size=lstm_dim // 2,
            bidirectional=True,
            num_layers=lstm_layer,
            dropout=lstm_drop,
        )
        self.emb_drop = nn.Dropout(emb_drop)
        self.lstm_dim = lstm_dim
        self.reset_parameters()

    def reset_parameters(self):
        self.word_embedding.weight.data.zero_()

    def forward(self, word_idxs, char_idxs):
        mask = word_idxs.ne(self.vocab.PAD_index)
        sen_lens = mask.sum(1)
        sorted_lens, sorted_idx = torch.sort(sen_lens, dim=0, descending=True)
        reverse_idx = torch.sort(sorted_idx, dim=0)[1]
        max_len = sorted_lens[0]

        word_idxs = word_idxs[:, :max_len]
        char_idxs, mask = char_idxs[:, :max_len], mask[:, :max_len]

        word_emb = self.ext_word_embedding(word_idxs)
        word_emb += self.word_embedding(word_idxs.masked_fill_(word_idxs.ge(self.word_embedding.num_embeddings),
                                                               self.vocab.UNK_index))
        char_vec = self.charlstm(char_idxs[mask])
        char_vec = pad_sequence(torch.split(char_vec, sen_lens.tolist()), True)

        emb = torch.cat((word_emb, char_vec), -1)
        emb = self.emb_drop(emb)

        emb = emb[sorted_idx]
        lstm_input = pack_padded_sequence(emb, sorted_lens, batch_first=True)

        r_out, _ = self.lstm(lstm_input, None)
        lstm_out, _ = pad_packed_sequence(r_out, batch_first=True)

        # get all span vectors
        x = lstm_out[reverse_idx].transpose(0, 1)
        x = x.unsqueeze(1) - x
        x_forward, x_backward = x.chunk(2, dim=-1)

        mask = (mask & word_idxs.ne(self.vocab.STOP_index))[:, :-1]
        mask = mask.unsqueeze(1) & mask.new_ones(
            max_len - 1, max_len - 1).triu(1)
        lens = mask.sum((1, 2))
        x_forward = x_forward[:-1, :-1].permute(2, 1, 0, 3)
        x_backward = x_backward[1:, 1:].permute(2, 0, 1, 3)
        x_span = torch.cat([x_forward[mask], x_backward[mask]], -1)
        x_span = pad_sequence(torch.split(x_span, lens.tolist()), True)
        return x_span, (sen_lens - 2).tolist()


class Attention_Encoder(nn.Module):
    def __init__(
        self,
        vocab,
        ext_emb,
        max_seq_len,
        word_dim,
        position_dim,
        char_dim,
        charlstm_dim,
        n_layers,
        n_head,
        d_k,
        d_v,
        d_model,
        d_inner,
        emb_drop=0.4,
        char_drop=0.2,
        relu_drop=0.1,
        attention_drop=0.2,
        residual_drop=0.2,
        partition=True,
    ):

        super(Attention_Encoder, self).__init__()
        self.vocab = vocab
        self.ext_word_embedding = nn.Embedding.from_pretrained(ext_emb)
        self.word_embedding = nn.Embedding(
            vocab.num_train_word, word_dim, padding_idx=0)
        self.charlstm = CharLSTM(
            vocab.num_char, char_dim, charlstm_dim, char_drop)
        self.position_encoding = PositionEncoding(
            position_dim, max_seq_len, padding_idx=0, freeze=False
        )
        # self.project = nn.Linear(word_dim, d_model)
        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(d_model, d_inner, n_head, d_k, d_v, attention_drop=attention_drop,
                             residual_drop=residual_drop, partition=partition)
                for _ in range(n_layers)
            ]
        )
        self.emb_drop = nn.Dropout(emb_drop)
        self.char_drop = nn.Dropout(char_drop)
        self.layer_norm = nn.LayerNorm(d_model)
        self.reset_parameters()

    def reset_parameters(self):
        self.word_embedding.weight.data.zero_()

    def forward(self, word_idxs, char_idxs):
        # -- Prepare masks
        mask = word_idxs.ne(self.vocab.PAD_index)
        sen_lens = mask.sum(1)
        max_len = max(sen_lens)

        word_idxs = word_idxs[:, :max_len]
        char_idxs, mask = char_idxs[:, :max_len], mask[:, :max_len]

        # -- Forward
        pos_idxs = torch.arange(
            start=1, end=max_len + 1, device=word_idxs.device
        ) * mask.type(torch.long)
        char_vec = self.charlstm(char_idxs[mask])
        char_vec = pad_sequence(torch.split(char_vec, sen_lens.tolist()), True)
        char_vec = self.char_drop(char_vec)

        word_emb = self.ext_word_embedding(word_idxs)
        word_emb += self.word_embedding(word_idxs.masked_fill_(word_idxs.ge(self.word_embedding.num_embeddings),
                                                               self.vocab.UNK_index))
        word_vec = self.emb_drop(word_emb)
        content = word_vec + char_vec

        position = self.position_encoding(pos_idxs)
        enc_output = torch.cat((content, position), dim=-1)
        enc_output = self.layer_norm(enc_output)

        # enc_output = self.project(enc_output)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, mask=mask)

        x = enc_output.transpose(0, 1)
        x = x.unsqueeze(1) - x
        # x_forward, x_backward = x.chunk(2, dim=-1)
        x_forward, x_backward = x[:, :, :, 0::2], x[:, :, :, 1::2]

        mask = (mask & word_idxs.ne(self.vocab.STOP_index))[:, :-1]
        mask = mask.unsqueeze(1) & mask.new_ones(
            max_len - 1, max_len - 1).triu(1)
        lens = mask.sum((1, 2))
        x_forward = x_forward[:-1, :-1].permute(2, 1, 0, 3)
        x_backward = x_backward[1:, 1:].permute(2, 0, 1, 3)
        x_span = torch.cat([x_forward[mask], x_backward[mask]], -1)
        x_span = pad_sequence(torch.split(x_span, lens.tolist()), True)

        return x_span, (sen_lens - 2).tolist()
