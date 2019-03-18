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
    word_idx, ext_word_idx, char_idx, passages, trees, all_nodes, all_remote = zip(*data)
    return (
        pad_sequence(word_idx, True),
        pad_sequence(ext_word_idx, True),
        pad_sequence(char_idx, True),
        passages,
        trees,
        all_nodes,
        all_remote,
    )


def collate_fn_cuda(data):
    word_idx, ext_word_idx, char_idx, passages, trees, all_nodes, all_remote = zip(*data)
    return (
        pad_sequence(word_idx, True).cuda(),
        pad_sequence(ext_word_idx, True).cuda(),
        pad_sequence(char_idx, True).cuda(),
        passages,
        trees,
        all_nodes,
        all_remote,
    )
