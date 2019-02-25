import torch
from ucca.layer0 import Terminal

from .convert import to_UCCA
from .trees import InternalParseNode, LeafParseNode


def augment(scores, oracle_index):
    increment = torch.ones_like(scores)
    increment[oracle_index] = 0
    return scores + increment


def get_span_encoding(lstm_out, i, j):
    lstm_dim = lstm_out.size(1) // 2
    forward = lstm_out[j][:lstm_dim] - lstm_out[i][:lstm_dim]
    backward = lstm_out[i + 1][lstm_dim:] - lstm_out[j + 1][lstm_dim:]
    return torch.cat((forward, backward))


class Decoder(object):
    def get_loss(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()


class Remote_Decoder(Decoder):
    def __init__(self, vocab, remote_scorer):
        super(Remote_Decoder, self).__init__()
        self.remote_scorer = remote_scorer
        self.vocab = vocab

    def score(self, lstm_out, all_span):
        span_vectors = [get_span_encoding(lstm_out, i, j) for i, j in all_span]
        span_vectors = torch.stack(span_vectors)
        edge_scores, label_scores = self.remote_scorer(span_vectors.unsqueeze(0))
        return edge_scores.squeeze(0), label_scores.squeeze(0).permute(1, 2, 0)

    def get_loss(self, lstm_out, all_span, remote):
        loss_func = torch.nn.CrossEntropyLoss(reduction="sum")
        edge_scores, label_scores = self.score(lstm_out, all_span)
        head, dep, label = remote
        edge_loss = loss_func(edge_scores[head], dep)

        label_loss = loss_func(label_scores[head, dep], label)
        return edge_loss + label_loss

    def predict(self, lstm_out, all_span, remote_head):
        edge_scores, label_scores = self.score(lstm_out, all_span)
        result = []
        for head in remote_head:
            dep = edge_scores[head].argmax()
            label = label_scores[head, dep].argmax()
            result.append((dep, label))
        return result

    def to_UCCA(self, passage, tree, lstm_out):
        passage = to_UCCA(passage, tree)
        self.restore_remote(passage, lstm_out)
        return passage

    def restore_remote(self, passage, lstm_out):
        def get_span(node):
            terminals = node.get_terminals()
            return (terminals[0].position - 1, terminals[-1].position)

        all_span, ids = [], []
        new_id = {}
        remote_head = []
        assert "1" in passage._layers
        for node in passage.layer("1").all:
            if len(node._incoming) > 0 and "-remote" in node._incoming[0].tag:
                assert len(node._incoming) == 1
                i = node._incoming[0]
                remote_head.append(i.child.ID)
                i._tag = i._tag.strip("-remote")
                all_span.append(get_span(node))
                ids.append(node.ID)
                new_id[node.ID] = len(new_id)
            else:
                if (
                    node.ID not in new_id
                    and len(node.get_terminals()) > 0
                    and not all(isinstance(n, Terminal) for n in node.children)
                ):
                    all_span.append(get_span(node))
                    ids.append(node.ID)
                    new_id[node.ID] = len(new_id)

        remote_head = [new_id[r] for r in remote_head]
        if len(remote_head) == 0:
            return
        else:
            remote_edge = self.predict(lstm_out, all_span, remote_head)
        for head, (dep, label) in zip(remote_head, remote_edge):
            head = passage._nodes[ids[head]]
            dep = passage._nodes[ids[dep]]
            label = self.vocab.id2edge_label(label)
            passage.layer("1").add_remote(dep, label, head)


class Chart(Decoder):
    def __init__(self, vocab, span_scorer):
        super(Chart, self).__init__()
        self.span_scorer = span_scorer
        self.vocab = vocab

    def get_label_scores(self, lstm_out, left, right):
        non_empty_label_scores = self.span_scorer(
            get_span_encoding(lstm_out, left, right)
        )
        return torch.cat(
            [torch.zeros(1, device=lstm_out.device), non_empty_label_scores]
        )

    def helper(self, force_gold, training, lstm_out, gold=None):
        if force_gold:
            assert training

        sen_len = lstm_out.size(0) - 2
        chart = {}

        for length in range(1, sen_len + 1):
            for left in range(0, sen_len + 1 - length):
                right = left + length

                label_scores = self.get_label_scores(lstm_out, left, right)

                if training:
                    oracle_label = gold.oracle_label(left, right)
                    oracle_label_index = self.vocab.parse_label2id(oracle_label)

                if force_gold:
                    label = oracle_label
                    label_score = label_scores[oracle_label_index]
                else:
                    if training:
                        label_scores = augment(label_scores, oracle_label_index)

                    argmax_label_index = (
                        label_scores.argmax()
                        if length < sen_len
                        else label_scores[1:].argmax() + 1
                    )
                    argmax_label = self.vocab.id2parse_label(int(argmax_label_index))
                    label = argmax_label
                    label_score = label_scores[argmax_label_index]

                if length == 1:
                    tree = LeafParseNode(left, 'pos', 'word')
                    if label:
                        tree = InternalParseNode(label, [tree])
                    chart[left, right] = [tree], label_score
                    continue

                if force_gold:
                    oracle_splits = gold.oracle_splits(left, right)
                    oracle_split = min(oracle_splits)
                    best_split = oracle_split
                else:
                    best_split = max(
                        range(left + 1, right),
                        key=lambda split: chart[left, split][1]
                        + chart[split, right][1],
                    )

                left_trees, left_score = chart[left, best_split]
                right_trees, right_score = chart[best_split, right]

                children = left_trees + right_trees
                if label:
                    children = [InternalParseNode(label, children)]

                chart[left, right] = (children, label_score + left_score + right_score)
        children, score = chart[0, sen_len]
        assert len(children) == 1
        return children[0], score

    def get_loss(self, lstm_out, gold):
        tree, score = self.helper(False, True, lstm_out, gold)
        oracle_tree, oracle_score = self.helper(True, True, lstm_out, gold)
        correct = tree.convert().linearize() == gold.convert().linearize()
        loss = score - oracle_score
        return tree, loss

    def predict(self, lstm_out):
        tree, score = self.helper(False, False, lstm_out, None)
        return tree, score


class Top_down(Decoder):
    def __init__(self, vocab, span_scorer):
        super(Top_down, self).__init__()
        self.span_scorer = span_scorer
        self.vocab = vocab
        
    def get_label_scores(self, lstm_out, left, right):
        label_scores, _ = self.span_scorer(get_span_encoding(lstm_out, left, right))
        return label_scores

    def get_split_scores(self, input):
        _, split_scores = self.span_scorer(input)
        return split_scores

    def helper(self, is_train, lstm_out, left, right, gold=None):
        sen_len = lstm_out.size(0) - 2
        label_scores = self.get_label_scores(lstm_out, left, right)

        if is_train:
            oracle_label = gold.oracle_label(left, right)
            oracle_label_index = self.vocab.parse_label2id(oracle_label)
            label_scores = augment(label_scores, oracle_label_index)

        argmax_label_index = (
            label_scores.argmax()
            if right - left < sen_len
            else label_scores[1:].argmax() + 1
        )
        argmax_label = self.vocab.id2parse_label(int(argmax_label_index))

        if is_train:
            label = oracle_label
            label_loss = (
                label_scores[argmax_label_index] - label_scores[oracle_label_index]
            )
        else:
            label = argmax_label
            label_loss = label_scores[argmax_label_index]

        if right - left == 1:
            tree = LeafParseNode(left, 'pos', 'word')
            if label:
                tree = InternalParseNode(label, [tree])
            return [tree], label_loss

        left_encodings = []
        right_encodings = []
        for split in range(left + 1, right):
            left_encodings.append(get_span_encoding(lstm_out, left, split))
            right_encodings.append(get_span_encoding(lstm_out, split, right))
        left_scores = self.get_split_scores(torch.stack(left_encodings))
        right_scores = self.get_split_scores(torch.stack(right_encodings))
        split_scores = left_scores + right_scores

        if is_train:
            oracle_splits = gold.oracle_splits(left, right)
            oracle_split = min(oracle_splits)
            oracle_split_index = oracle_split - (left + 1)
            split_scores = augment(split_scores, oracle_split_index)

        argmax_split_index = split_scores.argmax()
        argmax_split = argmax_split_index + (left + 1)

        if is_train:
            split = oracle_split
            split_loss = (
                split_scores[argmax_split_index] - split_scores[oracle_split_index]
            )
        else:
            split = argmax_split
            split_loss = split_scores[argmax_split_index]

        left_trees, left_loss = self.helper(
            is_train, lstm_out, left, int(split), gold
        )
        right_trees, right_loss = self.helper(
            is_train, lstm_out, int(split), right, gold
        )

        children = left_trees + right_trees
        if label:
            children = [InternalParseNode(label, children)]

        return children, label_loss + split_loss + left_loss + right_loss

    def get_loss(self, lstm_out, gold):
        sen_len = lstm_out.size(0) - 2
        children, loss = self.helper(True, lstm_out, 0, sen_len, gold)
        return children[0], loss

    def predict(self, lstm_out):
        sen_len = lstm_out.size(0) - 2
        children, loss = self.helper(False, lstm_out, 0, sen_len, None)
        return children[0], loss
