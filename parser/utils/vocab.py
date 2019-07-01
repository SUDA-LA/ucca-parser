import io
import pickle
from collections import Counter
from itertools import chain
from parser.convert import InternalParseNode

import torch
import torch.nn.init as init
from pytorch_pretrained_bert.tokenization import BertTokenizer
import unicodedata


def flatten(list_of_lists):
    for list in list_of_lists:
        for item in list:
            yield item


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (
        (cp >= 33 and cp <= 47)
        or (cp >= 58 and cp <= 64)
        or (cp >= 91 and cp <= 96)
        or (cp >= 123 and cp <= 126)
    ):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _clean_text(text):
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xFFFD or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def judge_ignore(word):
    if len(_clean_text(word)) == 0:
        return True
    if all(ord(char) == 0 or ord(char) == 0xFFFD or _is_control(char) for char in word):
        return True
    return False

class Vocab(object):
    def __init__(self, bert_vocab, corpus):
        self.bert_vocab = bert_vocab
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_vocab, do_lower_case=False
        )
        language, word, pos, dep, ent, ent_iob, edge_label, parse_label = self.collect(corpus)

        self.UNK = "<UNK>"
        self.START = "<START>"
        self.STOP = "<STOP>"
        self.PAD = "<PAD>"
        self.NULL = "<NULL>"

        self._lang = language
        self._word = [self.PAD, self.START, self.STOP, self.UNK] + word
        self._pos = [self.PAD] + pos + [self.START, self.STOP]
        self._dep = [self.PAD] + dep + [self.START, self.STOP]
        self._ent = [self.PAD] + ent + [self.START, self.STOP]
        self._ent_iob = [self.PAD] + ent_iob + [self.START, self.STOP]

        self.num_train_word = len(self._word)

        self._edge_label = [self.NULL] + edge_label
        self._parse_label = [()] + parse_label

        self._pos2id = {p: i for i, p in enumerate(self._pos)}
        self._dep2id = {p: i for i, p in enumerate(self._dep)}
        self._ent2id = {p: i for i, p in enumerate(self._ent)}
        self._ent_iob2id = {p: i for i, p in enumerate(self._ent_iob)}
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
        pos, dep, ent, ent_iob = [], [], [], []
        language = []
        for c in corpus:
            language.append(c.lang)
            for passage in c.passages:
                for node in passage.layer("0").all:
                    token.append(node.text)
                    pos.append(node.extra["pos"])
                    dep.append(node.extra["dep"])
                    ent.append(node.extra["ent_type"])
                    ent_iob.append(node.extra["ent_iob"])
                for node in passage.layer("1").all:
                    for e in node._incoming:
                        if e.attrib.get("remote"):
                            edge.append(e.tag)
        # word_count = Counter(token)
        words, edge_label = sorted(set(token)), sorted(set(edge))
        pos, dep, ent, ent_iob = sorted(set(pos)), sorted(set(dep)), sorted(set(ent)), sorted(set(ent_iob))

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

        return language, words, pos, dep, ent, ent_iob, edge_label, parse_label

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
    def num_pos(self):
        return len(self._pos)

    @property
    def num_dep(self):
        return len(self._dep)

    @property
    def num_ent(self):
        return len(self._ent)

    @property
    def num_ent_iob(self):
        return len(self._ent_iob)

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
        return "word:%d, pos:%d, dep:%d, ent:%d, ent_iob:%d, edge_label:%d, parse_label:%d, language:%d" % (
            self.num_word,
            self.num_pos,
            self.num_dep,
            self.num_ent,
            self.num_ent_iob,
            self.num_edge_label,
            self.num_parse_label,
            self.num_lang,
        )

    def convert_tokens_to_ids(self, tokens):
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = torch.tensor(token_ids, dtype=torch.long)
        mask = torch.ones(len(ids), dtype=torch.long)
        return ids, mask

    def subword_tokenize(self, tokens):
        subwords = list(map(self.tokenizer.tokenize, tokens))
        subword_lengths = [1] + list(map(len, subwords)) + [1]
        subwords = ["[CLS]"] + list(flatten(subwords)) + ["[SEP]"]
        token_start_idxs = torch.cumsum(torch.tensor([0] + subword_lengths[:-1]), dim=0)
        return subwords, token_start_idxs

    def subword_tokenize_to_ids(self, tokens):
        tokens = ["[PAD]" if judge_ignore(t) else t for t in tokens]
        subwords, token_start_idxs = self.subword_tokenize(tokens)
        subword_ids, mask = self.convert_tokens_to_ids(subwords)
        token_starts = torch.zeros(len(subword_ids), dtype=torch.uint8)
        token_starts[token_start_idxs] = 1
        return subword_ids, mask, token_starts

    def tokenize(self, tokens):
        subwords = list(map(self.tokenizer.tokenize, tokens))
        subword_lengths = [1] + list(map(len, subwords)) + [1]
        subwords = ["[CLS]"] + list(flatten(subwords)) + ["[SEP]"]
        return subwords

    def word2id(self, word):
        assert (isinstance(word, str) or isinstance(word, list))
        if isinstance(word, str):
            word_idx = self._word2id.get(word, self.UNK_index)
            return word_idx
        elif isinstance(word, list):
            word_idxs = [self._word2id.get(w, self.UNK_index) for w in word]
            return word_idxs

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
            return self._ent2id.get(entity, 0)
        elif isinstance(entity, list):
            return [self._ent2id.get(l, 0) for l in entity]

    def ent_iob2id(self, iob):
        assert isinstance(iob, str) or isinstance(iob, list)
        if isinstance(iob, str):
            return self._ent_iob2id.get(iob, 0)
        elif isinstance(iob, list):
            return [self._ent_iob2id.get(l, 0) for l in iob]

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