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
        subparser.add_argument("--en_train_path", required=True, help="en train data dir")
        subparser.add_argument("--fr_train_path", required=True, help="fr train data dir")
        subparser.add_argument("--de_train_path", required=True, help="de train data dir")

        subparser.add_argument("--en_dev_path", required=True, help="en dev data dir")
        subparser.add_argument("--fr_dev_path", required=True, help="fr dev data dir")
        subparser.add_argument("--de_dev_path", required=True, help="de dev data dir")

        subparser.add_argument("--save_path", required=True, help="dic to save all file")
        subparser.add_argument("--config_path", required=True, help="init config file")

        subparser.add_argument("--en_test_wiki_path", help="en wiki test data dir", default="")
        subparser.add_argument("--en_test_20k_path", help="en 20k data dir", default="")
        subparser.add_argument("--fr_test_20k_path", help="fr 20k data dir", default="")
        subparser.add_argument("--de_test_20k_path", help="de 20k data dir", default="")
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
        en_train = Corpus(args.en_train_path, "en")
        fr_train = Corpus(args.fr_train_path, "fr")
        de_train = Corpus(args.de_train_path, "de")
        print(en_train, "\n", fr_train, "\n", de_train)

        en_dev = Corpus(args.en_dev_path, "en")
        fr_dev = Corpus(args.fr_dev_path, "fr")
        de_dev = Corpus(args.de_dev_path, "de")
        print(en_dev, "\n", fr_dev, "\n", de_dev)

        # init vocab
        print("collecting words and labels in training dataset...")
        vocab = Vocab((en_train, fr_train, de_train))
        print(vocab)

        vocab_path = os.path.join(args.save_path, "vocab.pt")
        torch.save(vocab, vocab_path)

        # init parser
        print("initializing model...")
        ucca_parser = UCCA_Parser(vocab, config.ucca)
        if torch.cuda.is_available():
            ucca_parser = ucca_parser.cuda()

        # prepare data
        train_dataset = Data.ConcatDataset([en_train.generate_inputs(vocab, True), fr_train.generate_inputs(vocab, True), de_train.generate_inputs(vocab, True)])
        dev_dataset = Data.ConcatDataset([en_dev.generate_inputs(vocab, False), fr_dev.generate_inputs(vocab, False), de_dev.generate_inputs(vocab, False)])
        print("preparing input data...")
        train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=config.ucca.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        dev_loader = Data.DataLoader(
            dataset=dev_dataset,
            batch_size=10,
            shuffle=False,
            collate_fn=collate_fn,
        )

        optimizer = optim.Adam(ucca_parser.parameters(), lr=config.ucca.lr)
        ucca_evaluator = UCCA_Evaluator(
            parser=ucca_parser,
            gold_dic=[args.en_dev_path, args.fr_dev_path, args.de_dev_path]
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

        if args.en_test_wiki_path:
            print("evaluating en wiki test data : %s" % (args.en_test_wiki_path))
            test = Corpus(args.en_test_wiki_path, "en")
            print(test)
            test_loader = Data.DataLoader(
                dataset=test.generate_inputs(vocab, False),
                batch_size=10,
                shuffle=False,
                collate_fn=collate_fn,
            )
            ucca_evaluator = UCCA_Evaluator(
                parser=ucca_parser,
                gold_dic=[args.en_test_wiki_path],
            )
            ucca_evaluator.compute_accuracy(test_loader)
            ucca_evaluator.remove_temp()

        if args.en_test_20k_path:
            print("evaluating en 20K test data : %s" % (args.en_test_20k_path))
            test = Corpus(args.en_test_20k_path, "en")
            print(test)
            test_loader = Data.DataLoader(
                dataset=test.generate_inputs(vocab, False),
                batch_size=10,
                shuffle=False,
                collate_fn=collate_fn,
            )
            ucca_evaluator = UCCA_Evaluator(
                parser=ucca_parser,
                gold_dic=[args.en_test_20k_path],
            )
            ucca_evaluator.compute_accuracy(test_loader)
            ucca_evaluator.remove_temp()
        
        if args.fr_test_20k_path:
            print("evaluating fr 20K test data : %s" % (args.fr_test_20k_path))
            test = Corpus(args.fr_test_20k_path, "fr")
            print(test)
            test_loader = Data.DataLoader(
                dataset=test.generate_inputs(vocab, False),
                batch_size=10,
                shuffle=False,
                collate_fn=collate_fn,
            )
            ucca_evaluator = UCCA_Evaluator(
                parser=ucca_parser,
                gold_dic=[args.fr_test_20k_path],
            )
            ucca_evaluator.compute_accuracy(test_loader)
            ucca_evaluator.remove_temp()

        if args.de_test_20k_path:
            print("evaluating de 20K test data : %s" % (args.de_test_20k_path))
            test = Corpus(args.de_test_20k_path, "de")
            print(test)
            test_loader = Data.DataLoader(
                dataset=test.generate_inputs(vocab, False),
                batch_size=10,
                shuffle=False,
                collate_fn=collate_fn,
            )
            ucca_evaluator = UCCA_Evaluator(
                parser=ucca_parser,
                gold_dic=[args.de_test_20k_path],
            )
            ucca_evaluator.compute_accuracy(test_loader)
            ucca_evaluator.remove_temp()