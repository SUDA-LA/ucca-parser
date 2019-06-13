import os

import torch

from ucca.convert import to_text, xml2passage

from .instance import Instance
from .dataset import TensorDataSet


class Corpus(object):
    def __init__(self, dic_name=None, lang=None):
        self.dic_name = dic_name
        self.passages = self.read_passages(dic_name)
        self.instances = [Instance(passage) for passage in self.passages]
        self.language = lang

    @property
    def num_sentences(self):
        return len(self.passages)
    
    @property
    def lang(self):
        return self.language

    def __repr__(self):
        return "%s : %d sentences, %s language" % (self.dic_name, self.num_sentences, self.lang)

    def __getitem(self, index):
        return self.passages[index]

    @staticmethod
    def read_passages(path):
        passages = []
        for file in sorted(os.listdir(path)):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                print(file_path)
            passages.append(xml2passage(file_path))
        return passages

    def generate_inputs(self, vocab, is_training=False):
        lang_idxs, word_idxs, char_idxs = [], [], []
        trees, all_nodes, all_remote = [], [], []
        for instance in self.instances:
            _word_idxs = vocab.word2id([vocab.START] + instance.words + [vocab.STOP])
            _char_idxs = vocab.char2id([vocab.START] + instance.words + [vocab.STOP])
            _lang_idxs = [vocab.lang2id(self.lang)] * len(_word_idxs)

            nodes, (heads, deps, labels) = instance.gerenate_remote()
            if len(heads) == 0:
                _remotes = ()
            else:
                heads, deps = torch.tensor(heads), torch.tensor(deps)
                labels = [[vocab.edge_label2id(l) for l in label] for label in labels]
                labels = torch.tensor(labels)
                _remotes = (heads, deps, labels)

            lang_idxs.append(torch.tensor(_lang_idxs))
            word_idxs.append(torch.tensor(_word_idxs))
            char_idxs.append(torch.tensor(_char_idxs))

            if is_training:
                trees.append(instance.tree)
                all_nodes.append(nodes)
                all_remote.append(_remotes)
            else:
                trees.append([])
                all_nodes.append([])
                all_remote.append([])

        return TensorDataSet(
            lang_idxs,
            word_idxs,
            char_idxs,
            self.passages,
            trees,
            all_nodes,
            all_remote,
        )


class Embedding(object):
    def __init__(self, words, vectors):
        super(Embedding, self).__init__()

        self.words = words
        self.vectors = vectors
        self.pretrained = {w: v for w, v in zip(words, vectors)}

    def __len__(self):
        return len(self.words)

    def __contains__(self, word):
        return word in self.pretrained

    def __getitem__(self, word):
        return self.pretrained[word]

    @property
    def dim(self):
        return len(self.vectors[0])

    @classmethod
    def load(cls, fname, smooth=True):
        with open(fname, 'r') as f:
            lines = [line for line in f]
        splits = [line.split() for line in lines[1:]]
        reprs = [(s[0], list(map(float, s[1:]))) for s in splits]
        words, vectors = map(list, zip(*reprs))
        vectors = torch.tensor(vectors)
        if smooth:
            vectors /= torch.std(vectors)
        embedding = cls(words, vectors)

        return embedding