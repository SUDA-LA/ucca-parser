import io
import pickle
from collections import Counter
from itertools import chain
from parser import InternalParseNode

import torch


class Vocab(object):
    def __init__(self, corpus):
        word, word_count, pos, dep, entity, ent_iob, edge_label, parse_label = self.collect(corpus)
        self.word_count = word_count

        self.UNK = "<UNK>"
        self.START = "<START>"
        self.STOP = "<STOP>"
        self.PAD = "<PAD>"
        self.NULL = "<NULL>"

        self._word = [self.PAD] + word + [self.START, self.STOP, self.UNK]
        self._pos = [self.PAD] + pos + [self.START, self.STOP]
        self._dep = [self.PAD] + dep + [self.START, self.STOP]
        self._entity = [self.PAD] + entity + [self.START, self.STOP]
        self._ent_iob = [self.PAD] + ent_iob + [self.START, self.STOP]

        self._edge_label = [self.NULL] + edge_label
        self._parse_label = [()] + parse_label

        self._word2id = {w: i for i, w in enumerate(self._word)}
        self._pos2id = {p: i for i, p in enumerate(self._pos)}
        self._dep2id = {p: i for i, p in enumerate(self._dep)}
        self._entity2id = {p: i for i, p in enumerate(self._entity)}
        self._ent_iob2id = {p: i for i, p in enumerate(self._ent_iob)}

        self._edge_label2id = {e: i for i, e in enumerate(self._edge_label)}
        self._parse_label2id = {p: i for i, p in enumerate(self._parse_label)}

    def read_embedding(self, fname):
        print('reading pretrained embedding...')   
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        splits = [line.split() for line in fin]
        # read pretrained embedding file
        ext_words, vectors = zip(*[
            (split[0], list(map(float, split[1:]))) for split in splits
        ])

        embdim = len(vectors[0])
        ext_words = list(ext_words)
        ext_words.insert(0, self.PAD)
        vectors = list(vectors)
        vectors.insert(0, [0]*embdim)
        
        ext_words.extend([self.START, self.STOP, self.UNK])
        vectors.extend([[0]*embdim for _ in range(3)])
        self._ext_word2id = {w: i for i, w in enumerate(ext_words)}

        vectors = torch.tensor(vectors) / torch.std(torch.tensor(vectors))
        return vectors
 
    @staticmethod
    def collect(corpus):
        token, pos, edge, dep, entity, ent_iob = [], [], [], [], [], []
        for passage in corpus.passages:
            for node in passage.layer("0").all:
                token.append(node.text)
                pos.append(node.extra["pos"])
                dep.append(node.extra["dep"])
                entity.append(node.extra["ent_type"])
                ent_iob.append(node.extra["ent_iob"])
            for node in passage.layer("1").all:
                for e in node._incoming:
                    edge.append(e.tag)
        word_count = Counter(token)
        words, pos, edge_label = sorted(set(token)), sorted(set(pos)), sorted(set(edge))
        dep, entity, ent_iob = sorted(set(dep)), sorted(set(entity)), sorted(set(ent_iob))

        parse_label = []
        for instance in corpus.instances:
            instance.tree = instance.tree.convert()
            nodes = [instance.tree]
            while nodes:
                node = nodes.pop()
                if isinstance(node, InternalParseNode):
                    parse_label.append(node.label)
                    nodes.extend(reversed(node.children))
        parse_label = sorted(set(parse_label))
        return words, word_count, pos, dep, entity, ent_iob, edge_label, parse_label

    @property
    def num_word(self):
        return len(self._word)

    @property
    def num_pos(self):
        return len(self._pos)

    @property
    def num_dep(self):
        return len(self._dep)

    @property
    def num_entity(self):
        return len(self._entity)

    @property
    def num_ent_iob(self):
        return len(self._ent_iob)

    @property
    def num_edge_label(self):
        return len(self._edge_label)

    @property
    def num_parse_label(self):
        return len(self._parse_label)


    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        return obj

    def __repr__(self):
        return "word:%d, pos:%d, dep:%d, entity:%d, iob:%d, edge_label:%d, parse_label:%d" % (
            self.num_word,
            self.num_pos,
            self.num_dep,
            self.num_entity,
            self.num_ent_iob,
            self.num_edge_label,
            self.num_parse_label,
        )

    def word2id(self, word):
        assert (isinstance(word, str) or isinstance(word, list))
        if isinstance(word, str):
            word_idx = self._word2id.get(word, self._word2id[self.UNK])
            ext_word_idx = self._ext_word2id.get(word, self._ext_word2id[self.UNK])
            return word_idx, ext_word_idx
        elif isinstance(word, list):
            word_idxs = [self._word2id.get(w, self._word2id[self.UNK]) for w in word]
            ext_word_idxs = [self._ext_word2id.get(w, self._ext_word2id[self.UNK]) for w in word]
            return word_idxs, ext_word_idxs

    def pos2id(self, pos):
        assert isinstance(pos, str) or isinstance(pos, list)
        if isinstance(pos, str):
            return self._pos2id.get(pos, 0)  # if pos not in training data, index to 0 ?
        elif isinstance(pos, list):
            return [self._pos2id.get(l, 0) for l in pos]

    def dep2id(self, dep):
        assert isinstance(dep, str) or isinstance(dep, list)
        if isinstance(dep, str):
            return self._dep2id.get(dep, 0)
        elif isinstance(dep, list):
            return [self._dep2id.get(l, 0) for l in dep]

    def entity2id(self, entity):
        assert isinstance(entity, str) or isinstance(entity, list)
        if isinstance(entity, str):
            return self._entity2id.get(entity, 0)
        elif isinstance(entity, list):
            return [self._entity2id.get(l, 0) for l in entity]

    def ent_iob2id(self, iob):
        assert isinstance(iob, str) or isinstance(iob, list)
        if isinstance(iob, str):
            return self._ent_iob2id.get(iob, 0)
        elif isinstance(iob, list):
            return [self._ent_iob2id.get(l, 0) for l in iob]

    def edge_label2id(self, label):
        if isinstance(label, str):
            return self._edge_label2id[label]
        else:
            return [self._edge_label2id[l] for l in label]
    
    def id2parse_label(self, id):
        return self._parse_label[id]

    def id2edge_label(self, id):
        return self._edge_label[id]
        
    def id2word(self, id):
        return self._word[id]

    def id2pos(self, id):
        return self._pos[id]

    def parse_label2id(self, label):
        return self._parse_label2id[label]
