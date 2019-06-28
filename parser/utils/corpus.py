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
        subword_idxs, subword_masks, token_starts_masks = [], [], []
        lang_idxs, word_idxs = [], []
        pos_idxs, dep_idxs, ent_idxs, ent_iob_idxs = [], [], [], []
        trees, all_nodes, all_remote = [], [], []
        for instance in self.instances:
            subword_ids, mask, token_starts = vocab.subword_tokenize_to_ids(instance.words)
            subword_idxs.append(subword_ids)
            subword_masks.append(mask)
            token_starts_masks.append(token_starts)

            _word_idxs = vocab.word2id([vocab.START] + instance.words + [vocab.STOP])
            _pos_idxs = vocab.pos2id([vocab.START] + instance.pos + [vocab.STOP])
            _dep_idxs = vocab.dep2id([vocab.START] + instance.dep + [vocab.STOP])
            _entity_idxs = vocab.entity2id([vocab.START] + instance.ent + [vocab.STOP])
            _iob_idxs = vocab.ent_iob2id([vocab.START] + instance.ent_iob + [vocab.STOP])
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
            ent_iob_idxs.append(torch.tensor(_iob_idxs))
            pos_idxs.append(torch.tensor(_pos_idxs))
            dep_idxs.append(torch.tensor(_dep_idxs))
            ent_idxs.append(torch.tensor(_entity_idxs))

            if is_training:
                trees.append(instance.tree)
                all_nodes.append(nodes)
                all_remote.append(_remotes)
            else:
                trees.append([])
                all_nodes.append([])
                all_remote.append([])

        return TensorDataSet(
            subword_idxs,
            subword_masks,
            token_starts_masks,
            lang_idxs,
            word_idxs,
            pos_idxs,
            dep_idxs,
            ent_idxs,
            ent_iob_idxs,
            self.passages,
            trees,
            all_nodes,
            all_remote,
        )
        
    def filter(self, max_len, vocab):
        passages, instances = [], []
        for p, i in zip(self.passages, self.instances):
            if len(vocab.tokenize(i.words)) <= max_len:
                passages.append(p)
                instances.append(i)
            else:
                print("filter one sentence larger than 512!")
        self.passages = passages
        self.instances = instances


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