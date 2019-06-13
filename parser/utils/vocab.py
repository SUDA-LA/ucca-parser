import io
import pickle
from collections import Counter
from itertools import chain
from parser.convert import InternalParseNode

import torch
import torch.nn.init as init


class Vocab(object):
    def __init__(self, corpus):
        char, word, edge_label, parse_label, language = self.collect(corpus)

        self.UNK = "<UNK>"
        self.START = "<START>"
        self.STOP = "<STOP>"
        self.PAD = "<PAD>"
        self.NULL = "<NULL>"

        self._char = [self.PAD, self.UNK] + char
        self._lang = language
        self._word = [self.PAD, self.START, self.STOP, self.UNK] + word
        self.num_train_word = len(self._word)

        self._edge_label = [self.NULL] + edge_label
        self._parse_label = [()] + parse_label

        self._char2id = {c: i for i, c in enumerate(self._char)}
        self._word2id = {w: i for i, w in enumerate(self._word)}

        self._edge_label2id = {e: i for i, e in enumerate(self._edge_label)}
        self._parse_label2id = {p: i for i, p in enumerate(self._parse_label)}
        self._lang2id = {l: i for i, l in enumerate(self._lang)}

    def read_embedding(self, dim, pre_emb=None):
        if pre_emb:
            assert dim == pre_emb.dim
            self.extend(pre_emb.words)
            embeddings = torch.zeros(self.num_word, pre_emb.dim)
            init.normal_(embeddings, 0, 1 / pre_emb.dim ** 0.5)
            for i, word in enumerate(self._word):
                if word in pre_emb:
                    embeddings[i] = pre_emb[word]
            return embeddings
        else:
            embeddings = torch.zeros(self.num_word, dim)
            init.normal_(embeddings, 0, 1 / dim ** 0.5)
            return embeddings

    def extend(self, words):
        self._word.extend(sorted(set(words).difference(self._word2id)))
        self._word2id = {word: i for i, word in enumerate(self._word)}
 
    @staticmethod
    def collect(corpus):
        token, edge = [], []
        language = []
        for c in corpus:
            language.append(c.lang)
            for passage in c.passages:
                for node in passage.layer("0").all:
                    token.append(node.text)
                for node in passage.layer("1").all:
                    for e in node._incoming:
                        if e.attrib.get("remote"):
                            edge.append(e.tag)
        # word_count = Counter(token)
        words, edge_label = sorted(set(token)), sorted(set(edge))

        parse_label = []
        for c in corpus:
            for instance in c.instances:
                instance.tree = instance.tree.convert()
                nodes = [instance.tree]
                while nodes:
                    node = nodes.pop()
                    if isinstance(node, InternalParseNode):
                        parse_label.append(node.label)
                        nodes.extend(reversed(node.children))
        parse_label = sorted(set(parse_label))

        chars = sorted(set(''.join(words)))
        return chars, words, edge_label, parse_label, language

    @property
    def PAD_index(self):
        return self._word2id[self.PAD]

    @property
    def STOP_index(self):
        return self._word2id[self.STOP]

    @property
    def UNK_index(self):
        return self._word2id[self.UNK]

    @property
    def NULL_index(self):
        return self._parse_label2id[()]

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

    @property
    def num_lang(self):
        return len(self._lang)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        return obj

    def __repr__(self):
        return "word:%d, char:%d, edge_label:%d, parse_label:%d, language:%d" % (
            self.num_word,
            self.num_char,
            self.num_edge_label,
            self.num_parse_label,
            self.num_lang,
        )

    def word2id(self, word):
        assert (isinstance(word, str) or isinstance(word, list))
        if isinstance(word, str):
            word_idx = self._word2id.get(word, self.UNK_index)
            return word_idx
        elif isinstance(word, list):
            word_idxs = [self._word2id.get(w, self.UNK_index) for w in word]
            return word_idxs

    def char2id(self, char, max_len=20):
        assert (isinstance(char, str) or isinstance(char, list))
        if isinstance(char, str):
            return self._char2id.get(char, self._char2id[self.UNK])
        elif isinstance(char, list):
            return [[self._char2id.get(c, self._char2id[self.UNK]) for c in w[:max_len]] + 
                    [0] * (max_len - len(w)) for w in char]

    def edge_label2id(self, label):
        if isinstance(label, str):
            return self._edge_label2id.get(label, 0)
        else:
            return [self._edge_label2id.get(l, 0) for l in label]
    
    def id2parse_label(self, id):
        return self._parse_label[id]

    def id2edge_label(self, id):
        return self._edge_label[id]

    def parse_label2id(self, label):
        return self._parse_label2id[label]

    def lang2id(self, lang):
        return self._lang2id[lang]