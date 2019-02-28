import torch

from nn.submodel import (Chart_Span_Scorer, Remote_Scorer, Shared_Encoder,
                         Topdown_Span_Scorer)

from .convert import to_UCCA
from .decoder import Chart, Remote_Decoder, Top_down


class UCCA_Parser(torch.nn.Module):
    def __init__(self, vocab, args):
        super(UCCA_Parser, self).__init__()
        self.vocab = vocab
        pre_emb = vocab.read_embedding(args.emb_path)
        self.shared_encoder = Shared_Encoder(
            pre_emb,
            vocab.num_word,
            args.word_dim,
            vocab.num_pos,
            args.pos_dim,
            vocab.num_dep,
            args.dep_dim,
            vocab.num_entity,
            args.entity_dim,
            vocab.num_ent_iob,
            args.ent_iob_dim,
            args.lstm_dim,
            args.lstm_layer,
            args.emb_drop,
            args.lstm_drop,
        )

        if args.type == "chart":
            self.span_scorer = Chart_Span_Scorer(
                args.lstm_dim, args.label_hidden, vocab.num_parse_label, args.ffn_drop
            )
            self.span_decoder = Chart(vocab, self.span_scorer)
        elif args.type == "topdown":
            self.span_scorer = Topdown_Span_Scorer(
                args.lstm_dim,
                args.label_hidden,
                args.split_hidden,
                vocab.num_parse_label,
                args.ffn_drop,
            )
            self.span_decoder = Top_down(vocab, self.span_scorer)

        self.remote_scorer = Remote_Scorer(
            args.lstm_dim, args.mlp_label_dim, vocab.num_edge_label
        )
        self.remote_decoder = Remote_Decoder(vocab, self.remote_scorer)

        self.cuda = 0 <= args.gpu <= 7
        if self.cuda:
            self.shared_encoder.cuda()
            self.span_scorer.cuda()

    def parse(self, batch):
        if self.training:
            word_idxs, ext_word_idxs, pos_idxs, dep_idxs, entity_idxs, ent_iob_idxs, masks, passages, trees, all_nodes, all_remote = (
                batch
            )
        else:
            word_idxs, ext_word_idxs, pos_idxs, dep_idxs, entity_idxs, ent_iob_idxs, masks, passages = (
                batch
            )  # when predicting, gold layer1 will be removed if the passage have.

        lstm_outs = self.shared_encoder(
            word_idxs,
            ext_word_idxs,
            pos_idxs,
            dep_idxs,
            entity_idxs,
            ent_iob_idxs,
            masks,
        )
        batch_size = int(masks.size(0))
        lens = masks.sum(1).tolist()

        lstm_outs = torch.split(lstm_outs[masks], lens)

        if self.training:
            span_losses = 0.0
            remote_losses = 0.0
            for lstm_out, tree, nodes, remote in zip(
                lstm_outs, trees, all_nodes, all_remote
            ):
                _, span_loss = self.span_decoder.get_loss(lstm_out, tree)
                span_losses += span_loss
                if len(remote) == 0:
                    remote_loss = 0.0
                else:
                    remote_loss = self.remote_decoder.get_loss(lstm_out, nodes, remote)
                remote_losses += remote_loss
            return span_losses / batch_size + remote_loss
        else:
            predict_passages = []
            predict_trees = []
            for lstm_out, p in zip(lstm_outs, passages):
                pred_tree, _ = self.span_decoder.predict(lstm_out)
                pred_tree = pred_tree.convert()
                predict_trees.append(pred_tree)
                predict_passages.append(
                    self.remote_decoder.to_UCCA(p, pred_tree, lstm_out)
                )
            return predict_passages, predict_trees
