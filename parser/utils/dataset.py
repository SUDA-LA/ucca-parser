import torch
import torch.utils.data as Data
from torch.nn.utils.rnn import pad_sequence


class TensorDataSet(Data.Dataset):
    def __init__(self, *data):
        super(TensorDataSet, self).__init__()
        self.items = list(zip(*data))

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)


def collate_fn(data):
    subword_idxs, subword_masks, token_starts_masks, lang_idx, word_idx, pos_idx, dep_idx, ent_idx, ent_iob_idx, passages, trees, all_nodes, all_remote = zip(*data)
    return (
        pad_sequence(subword_idxs, True),
        pad_sequence(subword_masks, True),
        pad_sequence(token_starts_masks, True),
        pad_sequence(lang_idx, True),
        pad_sequence(word_idx, True),
        pad_sequence(pos_idx, True),
        pad_sequence(dep_idx, True),
        pad_sequence(ent_idx, True),
        pad_sequence(ent_iob_idx, True),
        passages,
        trees,
        all_nodes,
        all_remote,
    )