import torch

from .submodel import (
    Chart_Span_Parser,
    Remote_Parser,
    LSTM_Encoder,
    Attention_Encoder,
    Topdown_Span_Parser,
    Global_Chart_Span_Parser,
)
from .convert import to_UCCA
from .utils import get_config


class UCCA_Parser(torch.nn.Module):
    def __init__(self, vocab, args, pre_emb=None):
        super(UCCA_Parser, self).__init__()
        self.vocab = vocab
        self.encoder = args.encoder
        self.type = args.type
        if args.encoder == "lstm":
            self.shared_encoder = LSTM_Encoder(
                vocab=vocab,
                ext_emb=pre_emb,
                word_dim=args.word_dim,
                char_dim=args.char_dim,
                charlstm_dim=args.charlstm_dim,
                lstm_dim=args.lstm_dim,
                lstm_layer=args.lstm_layer,
                emb_drop=args.emb_drop,
                lstm_drop=args.lstm_drop,
                char_drop=args.char_drop,
            )
        elif args.encoder == "attention":
            self.shared_encoder = Attention_Encoder(
                vocab=vocab,
                ext_emb=pre_emb,
                max_seq_len=args.max_seq_len,
                word_dim=args.word_dim,
                position_dim=args.position_dim,
                char_dim=args.char_dim,
                charlstm_dim=args.charlstm_dim,
                n_layers=args.attn_layer,
                n_head=args.n_head,
                d_k=args.d_k,
                d_v=args.d_v,
                d_model=args.d_model,
                d_inner=args.d_inner,
                emb_drop=args.emb_drop,
                char_drop=args.char_drop,
                relu_drop=0.1,
                attention_drop=0.2,
                residual_drop=0.2,
                partition=args.partition,
            )
        if args.type == "chart":
            self.span_parser = Chart_Span_Parser(
                vocab=vocab,
                lstm_dim=args.lstm_dim if args.encoder=='lstm' else args.d_model,
                label_hidden_dim=args.label_hidden,
                drop=args.ffn_drop,
                norm=False if args.encoder=='lstm' else True,
            )
        elif args.type == "top-down":
            self.span_parser = Topdown_Span_Parser(
                vocab=vocab,
                lstm_dim=args.lstm_dim if args.encoder=='lstm' else args.d_model,
                label_hidden_dim=args.label_hidden,
                split_hidden_dim=args.split_hidden,
                drop=args.ffn_drop,
                norm=False if args.encoder=='lstm' else True,
            )
        elif args.type == "global-chart":
            self.span_parser = Global_Chart_Span_Parser(
                vocab=vocab,
                lstm_dim=args.lstm_dim if args.encoder=='lstm' else args.d_model,
                label_hidden_dim=args.label_hidden,
                drop=args.ffn_drop,
                norm=False if args.encoder=='lstm' else True,
            )

        self.remote_parser = Remote_Parser(
            vocab=vocab,
            lstm_dim=args.lstm_dim if args.encoder=='lstm' else args.d_model,
            mlp_label_dim=args.mlp_label_dim,
        )

    def parse(self, word_idxs, char_idxs, passages, trees=None, all_nodes=None, all_remote=None):
        spans, sen_lens = self.shared_encoder(word_idxs, char_idxs)

        if self.training:
            span_loss = self.span_parser.get_loss(spans, sen_lens, trees)
            remote_loss = self.remote_parser.get_loss(spans, sen_lens, all_nodes, all_remote)
            return span_loss, remote_loss
        else:
            predict_trees = self.span_parser.predict(spans, sen_lens)
            predict_passages = [to_UCCA(passage, pred_tree) for passage, pred_tree in zip(passages, predict_trees)]
            predict_passages = self.remote_parser.restore_remote(predict_passages, spans, sen_lens)
            return predict_passages

    @classmethod
    def load(cls, vocab_path, config_path, state_path):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        state = torch.load(state_path, map_location=device)
        vocab = torch.load(vocab_path)
        config = get_config(config_path)

        network = cls(vocab, config.ucca, state['embeddings'])
        network.load_state_dict(state['state_dict'])
        network.to(device)

        return network

    def save(self, fname):
        state = {
            'embeddings': self.shared_encoder.ext_word_embedding.weight,
            'state_dict': self.state_dict(),
        }
        torch.save(state, fname)