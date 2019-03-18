import io
import pickle
from collections import Counter
from itertools import chain
from parser.convert import InternalParseNode

import torch


class Vocab(object):
    def __init__(self, corpus):
        char, word, edge_label, parse_label = self.collect(corpus)

        self.UNK = "<UNK>"
        self.START = "<START>"
        self.STOP = "<STOP>"
        self.PAD = "<PAD>"
        self.NULL = "<NULL>"

        self._char = [self.PAD, self.UNK] + char
        self._word = [self.PAD, self.START, self.STOP, self.UNK] + word

        self._edge_label = [self.NULL] + edge_label
        self._parse_label = [()] + parse_label

        self._char2id = {c: i for i, c in enumerate(self._char)}
        self._word2id = {w: i for i, w in enumerate(self._word)}

        self._edge_label2id = {e: i for i, e in enumerate(self._edge_label)}
        self._parse_label2id = {p: i for i, p in enumerate(self._parse_label)}

    @property
    def PAD_index(self):
        return self._word2id[self.PAD]

    @property
    def STOP_index(self):
        return self._word2id[self.STOP]

    @property
    def NULL_index(self):
        return self._parse_label2id[()]


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
        ext_words = [self.PAD, self.START, self.STOP, self.UNK] + ext_words
        vectors = list(vectors)
        vectors = [[0]*embdim for _ in range(4)] + vectors
        
        self._ext_word2id = {w: i for i, w in enumerate(ext_words)}

        vectors = torch.tensor(vectors) / torch.std(torch.tensor(vectors))
        return vectors
 
    @staticmethod
    def collect(corpus):
        token, edge = [], []
        for passage in corpus.passages:
            for node in passage.layer("0").all:
                token.append(node.text)
            for node in passage.layer("1").all:
                for e in node._incoming:
                    if e.attrib.get("remote"):
                        edge.append(e.tag)
        # word_count = Counter(token)
        words, edge_label = sorted(set(token)), sorted(set(edge))

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

        chars = sorted(set(''.join(words)))
        return chars, words, edge_label, parse_label

    @property
    def num_word(self):
        return len(self._word)

    @property
    def num_char(self):
        return len(self._char)

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
        return "word:%d, char:%d, edge_label:%d, parse_label:%d" % (
            self.num_word,
            self.num_char,
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

    def char2id(self, char, max_len=20):
        assert (isinstance(char, str) or isinstance(char, list))
        if isinstance(char, str):
            return self._char2id.get(char, self._char2id[self.UNK])
        elif isinstance(char, list):
            return [[self._char2id.get(c, self._char2id[self.UNK]) for c in w[:max_len]] + 
                    [0] * (max_len - len(w)) for w in char]

    def edge_label2id(self, label):
        if isinstance(label, str):
            return self._edge_label2id[label]
        else:
            return [self._edge_label2id[l] for l in label]
    
    def id2parse_label(self, id):
        return self._parse_label[id]

    def id2edge_label(self, id):
        return self._edge_label[id]

    def parse_label2id(self, label):
        return self._parse_label2id[label]
