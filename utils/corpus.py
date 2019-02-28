import os

import torch

from ucca.convert import to_text, xml2passage

from .instance import Instance
from .dataset import TensorDataSet


class Corpus(object):
    def __init__(self, dic_name=None):
        self.dic_name = dic_name
        self.passages = self.read_passages(dic_name)
        self.instances = [Instance(passage) for passage in self.passages]

    @property
    def num_sentences(self):
        return len(self.passages)

    def __repr__(self):
        return "%s : %d sentences" % (self.dic_name, self.num_sentences)

    def __getitem(self, index):
        return self.passages[index]

    @staticmethod
    def read_passages(path):
        passages = []
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                print(file_path)
            passages.append(xml2passage(file_path))
        return passages

    def generate_inputs(self, vocab, is_training=False):
        word_idxs, ext_word_idxs, pos_idxs, dep_idxs, entity_idxs, ent_iob_idxs, masks = [], [], [], [], [], [], []
        trees, all_nodes, all_remote, passages = [], [], [], []
        for instance in self.instances:
            _word_idxs, _ext_word_idxs = vocab.word2id(
                [vocab.START] + instance.words + [vocab.STOP]
            )
            _pos_idxs = vocab.pos2id([vocab.START] + instance.pos + [vocab.STOP])
            _dep_idxs = vocab.dep2id([vocab.START] + instance.dep + [vocab.STOP])
            _entity_idxs = vocab.entity2id([vocab.START] + instance.entity + [vocab.STOP])
            _iob_idxs = vocab.ent_iob2id([vocab.START] + instance.ent_iob + [vocab.STOP])

            # _sentence = list(zip(instance.words, instance.pos))
            _masks = torch.ones(instance.size + 2, dtype=torch.uint8)

            nodes, (heads, deps, labels) = instance.gerenate_remote()
            if len(heads) == 0:
                _remotes = ()
            else:
                heads, deps = torch.tensor(heads), torch.tensor(deps)
                labels = [[vocab.edge_label2id(l) for l in label] for label in labels]
                labels = torch.tensor(labels)
                _remotes = (heads, deps, labels)

            word_idxs.append(torch.tensor(_word_idxs))
            ext_word_idxs.append(torch.tensor(_ext_word_idxs))
            ent_iob_idxs.append(torch.tensor(_iob_idxs))
            pos_idxs.append(torch.tensor(_pos_idxs))
            dep_idxs.append(torch.tensor(_dep_idxs))
            entity_idxs.append(torch.tensor(_entity_idxs))

            masks.append(_masks)
            # sentences.append(_sentence)
            passages.append(instance.passage)
            if is_training:
                trees.append(instance.tree)
                all_nodes.append(nodes)
                all_remote.append(_remotes)
            else:
                trees.append([])
                all_nodes.append([])
                all_remote.append([])

        return TensorDataSet(
            word_idxs,
            ext_word_idxs,
            pos_idxs,
            dep_idxs,
            entity_idxs,
            ent_iob_idxs,
            masks,
            passages,
            trees,
            all_nodes,
            all_remote,
        )
