import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pack_sequence,
    pad_sequence,
    pad_packed_sequence,
)

from parser.module import CharLSTM, EncoderLayer, PositionEncoding


class LSTM_Encoder(nn.Module):
    def __init__(
        self,
        vocab,
        lang_dim,
        word_dim,
        pos_dim,
        dep_dim,
        ent_dim,
        ent_iob_dim,
        lstm_dim,
        lstm_layer,
        emb_drop=0.5,
        lstm_drop=0.4,
        char_drop=0,
    ):
        super(LSTM_Encoder, self).__init__()
        self.vocab = vocab
        self.lang_embedding = nn.Embedding(vocab.num_lang, lang_dim)
        self.word_embedding = nn.Embedding(vocab.num_word, word_dim, padding_idx=0)

        self.pos_embedding = nn.Embedding(vocab.num_pos, pos_dim, padding_idx=0)
        self.dep_embedding = nn.Embedding(vocab.num_dep, dep_dim, padding_idx=0)
        self.ent_embedding = nn.Embedding(vocab.num_ent, ent_dim, padding_idx=0)
        self.ent_iob_embedding = nn.Embedding(vocab.num_ent_iob, ent_iob_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=word_dim + pos_dim + dep_dim + ent_dim + ent_iob_dim + lang_dim,
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

    def forward(self, lang_idxs, word_idxs, pos_idxs, dep_idxs, ent_idxs, ent_iob_idxs):
        mask = word_idxs.ne(self.vocab.PAD_index)
        sen_lens = mask.sum(1)
        sorted_lens, sorted_idx = torch.sort(sen_lens, dim=0, descending=True)
        reverse_idx = torch.sort(sorted_idx, dim=0)[1]
        max_len = sorted_lens[0]

        lang_idxs, word_idxs = lang_idxs[:, :max_len], word_idxs[:, :max_len]
        pos_idxs = pos_idxs[:, :max_len]
        dep_idxs = dep_idxs[:, :max_len]
        ent_idxs = ent_idxs[:, :max_len]
        ent_iob_idxs = ent_iob_idxs[:, :max_len]
        mask = mask[:, :max_len]

        lang_emb = self.lang_embedding(lang_idxs)
        word_emb = self.word_embedding(word_idxs)
        pos_emb = self.pos_embedding(pos_idxs)
        dep_emb = self.dep_embedding(dep_idxs)
        ent_emb = self.ent_embedding(ent_idxs)
        ent_iob_emb = self.ent_iob_embedding(ent_iob_idxs)


        emb = torch.cat((lang_emb, word_emb, pos_emb, dep_emb, ent_emb, ent_iob_emb), -1)
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
        mask = mask.unsqueeze(1) & mask.new_ones(max_len - 1, max_len - 1).triu(1)
        lens = mask.sum((1, 2))
        x_forward = x_forward[:-1, :-1].permute(2, 1, 0, 3)
        x_backward = x_backward[1:, 1:].permute(2, 0, 1, 3)
        x_span = torch.cat([x_forward[mask], x_backward[mask]], -1)
        x_span = pad_sequence(torch.split(x_span, lens.tolist()), True)
        return x_span, (sen_lens - 2).tolist()
