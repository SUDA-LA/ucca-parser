import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pack_sequence,
    pad_sequence,
    pad_packed_sequence,
)

from parser.module import CharLSTM


class Shared_Encoder(nn.Module):
    def __init__(
        self,
        vocab,
        ext_emb,
        n_word,
        word_dim,
        n_char,
        char_dim,
        charlstm_dim,
        lstm_dim,
        lstm_layer,
        emb_drop,
        lstm_drop,
    ):
        super(Shared_Encoder, self).__init__()
        self.vocab = vocab
        self.word_embedding = nn.Embedding(n_word, word_dim, padding_idx=0)
        self.ext_word_embedding = nn.Embedding.from_pretrained(ext_emb)
        self.charlstm = CharLSTM(n_char, char_dim, charlstm_dim)

        self.lstm = nn.LSTM(
            input_size=word_dim + charlstm_dim * 2,
            hidden_size=lstm_dim,
            bidirectional=True,
            num_layers=lstm_layer,
            dropout=lstm_drop,
        )
        self.emb_drop = nn.Dropout(emb_drop)
        self.lstm_dim = lstm_dim
        self.reset_parameters()

    def reset_parameters(self):
        self.word_embedding.weight.data.zero_()

    def forward(self, word_idxs, ext_word_idxs, char_idxs):
        mask = word_idxs.ne(self.vocab.PAD_index)
        sen_lens = mask.sum(1)
        sorted_lens, sorted_idx = torch.sort(sen_lens, dim=0, descending=True)
        reverse_idx = torch.sort(sorted_idx, dim=0)[1]

        word_emb = self.word_embedding(word_idxs)
        ext_word_emb = self.ext_word_embedding(ext_word_idxs)
        char_vec = self.charlstm(char_idxs[mask])
        char_vec = pad_sequence(torch.split(char_vec, sen_lens.tolist()), True)

        word_emb += ext_word_emb
        emb = torch.cat((word_emb, char_vec), -1)
        emb = self.emb_drop(emb)

        emb = emb[sorted_idx]
        lstm_input = pack_padded_sequence(emb, sorted_lens, batch_first=True)

        r_out, _ = self.lstm(lstm_input, None)
        lstm_out, _ = pad_packed_sequence(r_out, batch_first=True)
        
        # get all span vectors
        max_len = sorted_lens[0]
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

