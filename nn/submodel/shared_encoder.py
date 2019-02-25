import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence


class Shared_Encoder(nn.Module):
    def __init__(
        self,
        ext_emb,
        n_word,
        word_dim,
        n_pos,
        pos_dim,
        n_dep,
        dep_dim,
        n_entity,
        entity_dim,
        n_ent_iob,
        ent_iob_dim,
        lstm_dim,
        lstm_layer,
        emb_drop,
        lstm_drop,
    ):
        super(Shared_Encoder, self).__init__()
        self.word_embedding = nn.Embedding(n_word, word_dim, padding_idx=0)
        self.ext_word_embedding = nn.Embedding.from_pretrained(ext_emb)

        self.pos_embedding = nn.Embedding(n_pos, pos_dim, padding_idx=0)
        self.dep_embedding = nn.Embedding(n_dep, dep_dim, padding_idx=0)
        self.entity_embedding = nn.Embedding(n_entity, entity_dim, padding_idx=0)
        self.ent_iob_embedding = nn.Embedding(n_ent_iob, ent_iob_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=word_dim + pos_dim + dep_dim + entity_dim + ent_iob_dim,
            hidden_size=lstm_dim,
            bidirectional=True,
            num_layers=lstm_layer,
            dropout=lstm_drop,
        )
        self.emb_drop = nn.Dropout(emb_drop)
        self.lstm_dim = lstm_dim
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.pos_embedding.weight)
        init.normal_(self.dep_embedding.weight)
        init.normal_(self.entity_embedding.weight)
        init.normal_(self.ent_iob_embedding.weight)
        self.word_embedding.weight.data.zero_()

    def forward(self, word_idx, ext_word_idxs, pos_idx, dep_idx, entity_idx, ent_iob_idx, mask):
        word_emb = self.word_embedding(word_idx)
        ext_word_emb = self.ext_word_embedding(ext_word_idxs)

        pos_emb = self.pos_embedding(pos_idx)
        dep_emb = self.dep_embedding(dep_idx)
        entity_emb = self.entity_embedding(entity_idx)
        ent_iob_emb = self.ent_iob_embedding(ent_iob_idx)
        
        word_emb += ext_word_emb
        emb = torch.cat((word_emb, pos_emb, dep_emb, entity_emb, ent_iob_emb), -1)
        emb = self.emb_drop(emb)

        emb = emb[mask]
        lens = mask.sum(1)
        lstm_input = torch.split(emb, lens.tolist())
        lstm_input = pack_sequence(lstm_input)

        r_out, _ = self.lstm(lstm_input, None)
        lstm_out, _ = pad_packed_sequence(r_out, batch_first=True)
        return lstm_out
