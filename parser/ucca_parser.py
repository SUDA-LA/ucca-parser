import torch

from parser.submodel import (
    Chart_Span_Parser,
    Remote_Parser,
    Shared_Encoder,
    Topdown_Span_Parser,
)

from parser.convert import to_UCCA


class UCCA_Parser(torch.nn.Module):
    def __init__(self, vocab, args):
        super(UCCA_Parser, self).__init__()
        self.vocab = vocab
        pre_emb = vocab.read_embedding(args.emb_path)
        self.shared_encoder = Shared_Encoder(
            vocab,
            pre_emb,
            vocab.num_word,
            args.word_dim,
            vocab.num_char,
            args.char_dim,
            args.charlstm_dim,
            args.lstm_dim,
            args.lstm_layer,
            args.emb_drop,
            args.lstm_drop,
        )
        if args.type == "chart":
            self.span_parser = Chart_Span_Parser(
                vocab,
                args.lstm_dim,
                args.label_hidden,
                vocab.num_parse_label,
                args.ffn_drop,
            )
        elif args.type == "topdown":
            self.span_parser = Topdown_Span_Parser(
                vocab,
                args.lstm_dim,
                args.label_hidden,
                args.split_hidden,
                vocab.num_parse_label,
                args.ffn_drop,
            )

        self.remote_parser = Remote_Parser(
            vocab, args.lstm_dim, args.mlp_label_dim, vocab.num_edge_label
        )

    def parse(self, word_idxs, ext_word_idxs, char_idxs, passages, trees=None, all_nodes=None, all_remote=None):
        spans, sen_lens = self.shared_encoder(word_idxs, ext_word_idxs, char_idxs)

        if self.training:
            span_loss = self.span_parser.get_loss(spans, sen_lens, trees)
            remote_loss = self.remote_parser.get_loss(spans, sen_lens, all_nodes, all_remote)
            return span_loss + remote_loss
        else:
            predict_trees = self.span_parser.predict(spans, sen_lens)
            predict_passages = [to_UCCA(passage, pred_tree) for passage, pred_tree in zip(passages, predict_trees)]
            predict_passages = self.remote_parser.restore_remote(predict_passages, spans, sen_lens)
            return predict_passages
