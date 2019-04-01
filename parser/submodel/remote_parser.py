import torch
import torch.nn as nn

from parser.module import MLP, Biaffine
from parser.convert import get_position


class Remote_Parser(nn.Module):
    def __init__(self, vocab, lstm_dim, mlp_label_dim):
        super(Remote_Parser, self).__init__()
        self.vocab = vocab
        self.label_head_mlp = MLP(lstm_dim, mlp_label_dim, nn.ReLU())
        self.label_dep_mlp = MLP(lstm_dim, mlp_label_dim, nn.ReLU())

        self.label_biaffine = Biaffine(
            mlp_label_dim, vocab.num_edge_label, bias_dep=True, bias_head=True
        )

    def forward(self, span_vectors):
        label_head_mlp_out = self.label_head_mlp(span_vectors)
        label_dep_mlp_out = self.label_dep_mlp(span_vectors)

        label_scores = self.label_biaffine(label_head_mlp_out, label_dep_mlp_out)
        return label_scores

    def score(self, span_vectors, sen_len, all_span):
        span_vectors = [span_vectors[get_position(sen_len, i, j)] for i, j in all_span]
        span_vectors = torch.stack(span_vectors)
        label_scores = self.forward(span_vectors.unsqueeze(0))
        return label_scores.squeeze(0).permute(1, 2, 0)

    def get_loss(self, spans, sen_lens, all_nodes, all_remote):
        loss_func = torch.nn.CrossEntropyLoss()
        batch_loss = []
        for i, length in enumerate(sen_lens):
            if len(all_remote[i]) == 0:
                batch_loss.append(0)
                continue
            span_num = (1 + length) * length // 2
            label_scores = self.score(spans[i][:span_num], length, all_nodes[i])
            head, dep, label = all_remote[i]
            batch_loss.append(
                loss_func(
                    label_scores[head.view(-1), dep.view(-1)],
                    label.view(-1).to(spans[i].device),
                )
            )
        return batch_loss

    def predict(self, span, sen_len, all_nodes, remote_head):
        label_scores = self.score(span, sen_len, all_nodes)
        labels = label_scores[remote_head].argmax(dim=-1)
        return labels

    def restore_remote(self, passages, spans, sen_lens):
        def get_span_index(node):
            terminals = node.get_terminals()
            return (terminals[0].position - 1, terminals[-1].position)

        for passage, span, length in zip(passages, spans, sen_lens):
            heads = []
            nodes = passage.layer("1").all
            ndict = {node: i for i, node in enumerate(nodes)}
            span_index = [get_span_index(i) for i in nodes]
            for node in nodes:
                for edge in node._incoming:
                    if "-remote" in edge.tag:
                        heads.append(node)
                        if hasattr(edge, "categories"):
                            edge.categories[0]._tag = edge.categories[0]._tag.strip(
                                "-remote"
                            )
                        else:
                            edge._tag = edge._tag.strip("-remote")
            heads = [ndict[node] for node in heads]

            if len(heads) == 0:
                continue
            else:
                label_scores = self.predict(span, length, span_index, heads)

            for head, label_score in zip(heads, label_scores):
                for i, score in enumerate(label_score):
                    label = self.vocab.id2edge_label(score)
                    if label is not self.vocab.NULL:
                        passage.layer("1").add_remote(nodes[i], label, nodes[head])
        return passages
