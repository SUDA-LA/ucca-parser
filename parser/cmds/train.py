import argparse
import json
import os
from parser import UCCA_Parser
import torch.optim as optim
import torch
import torch.utils.data as Data

from parser.utils import (
    Corpus,
    Trainer,
    Vocab,
    collate_fn,
    get_config,
    Embedding,
    UCCA_Evaluator,
    MyScheduledOptim,
)


class Train(object):
    def add_subparser(self, name, parser):
        subparser = parser.add_parser(name, help="Train a model.")
        subparser.add_argument("--train_path", required=True, help="train data dir")
        subparser.add_argument("--dev_path", required=True, help="dev data dir")
        subparser.add_argument("--emb_path", help="pretrained embedding path", default="")
        subparser.add_argument("--save_path", required=True, help="dic to save all file")
        subparser.add_argument("--config_path", required=True, help="init config file")
        subparser.add_argument("--test_wiki_path", help="wiki test data dir", default="")
        subparser.add_argument("--test_20k_path", help="20k data dir", default="")
        subparser.set_defaults(func=self)

        return subparser

    def __call__(self, args):
        config = get_config(args.config_path)
        assert config.ucca.type in ["chart", "top-down", "global-chart"]

        with open(os.path.join(args.save_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, default=lambda o: o.__dict__, indent=4)

        print("save all files to %s" % (args.save_path))
        # read training , dev file
        print("loading datasets and transforming to trees...")
        train = Corpus(args.train_path)
        dev = Corpus(args.dev_path)
        print(train, "\n", dev)

        # init vocab
        print("collecting words and labels in training dataset...")
        vocab = Vocab(config.ucca.bert_vocab, train)
        print(vocab)

        # filter sentences larger than 512
        train.filter(512, vocab)
        
        # prepare pre-trained embedding
        if args.emb_path:
            print("reading pre-trained embedding...")
            pre_emb = Embedding.load(args.emb_path)
            print(
                "pre-trained words:%d, dim=%d in %s"
                % (len(pre_emb), pre_emb.dim, args.emb_path)
            )
        else:
            pre_emb = None
        embedding = vocab.read_embedding(config.ucca.word_dim, pre_emb)
        vocab_path = os.path.join(args.save_path, "vocab.pt")
        torch.save(vocab, vocab_path)

        # init parser
        print("initializing model...")
        ucca_parser = UCCA_Parser(vocab, config.ucca, pre_emb=embedding)
        if torch.cuda.is_available():
            ucca_parser = ucca_parser.cuda()

        # prepare data
        print("preparing input data...")
        train_loader = Data.DataLoader(
            dataset=train.generate_inputs(vocab, True),
            batch_size=config.ucca.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        dev_loader = Data.DataLoader(
            dataset=dev.generate_inputs(vocab, False),
            batch_size=10,
            shuffle=False,
            collate_fn=collate_fn,
        )

        optimizer = optim.Adam(ucca_parser.parameters(), lr=config.ucca.lr)
        ucca_evaluator = UCCA_Evaluator(
            parser=ucca_parser,
            gold_dic=args.dev_path,
        )

        trainer = Trainer(
            parser=ucca_parser,
            optimizer=optimizer,
            evaluator=ucca_evaluator,
            batch_size=config.ucca.batch_size,
            epoch=config.ucca.epoch,
            patience=config.ucca.patience,
            path=args.save_path,
        )
        trainer.train(train_loader, dev_loader)

        # reload parser
        del ucca_parser
        torch.cuda.empty_cache()
        print("reloading the best parser for testing...")
        vocab_path = os.path.join(args.save_path, "vocab.pt")
        state_path = os.path.join(args.save_path, "parser.pt")
        config_path = os.path.join(args.save_path, "config.json")
        ucca_parser = UCCA_Parser.load(vocab_path, config_path, state_path)

        if args.test_wiki_path:
            print("evaluating test data : %s" % (args.test_wiki_path))
            test = Corpus(args.test_wiki_path)
            print(test)
            test_loader = Data.DataLoader(
                dataset=test.generate_inputs(vocab, False),
                batch_size=10,
                shuffle=False,
                collate_fn=collate_fn,
            )
            ucca_evaluator = UCCA_Evaluator(
                parser=ucca_parser,
                gold_dic=args.test_wiki_path,
            )
            ucca_evaluator.compute_accuracy(test_loader)
            ucca_evaluator.remove_temp()

        if args.test_20k_path:
            print("evaluating test data : %s" % (args.test_20k_path))
            test = Corpus(args.test_20k_path)
            print(test)
            test_loader = Data.DataLoader(
                dataset=test.generate_inputs(vocab, False),
                batch_size=10,
                shuffle=False,
                collate_fn=collate_fn,
            )
            ucca_evaluator = UCCA_Evaluator(
                parser=ucca_parser,
                gold_dic=args.test_20k_path,
            )
            ucca_evaluator.compute_accuracy(test_loader)
            ucca_evaluator.remove_temp()