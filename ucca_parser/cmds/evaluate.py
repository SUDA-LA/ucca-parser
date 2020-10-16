import argparse
import os
from ucca_parser import UCCA_Parser

import torch
import torch.utils.data as Data
from ucca.convert import passage2file

from ucca_parser.utils import Corpus, collate_fn, UCCA_Evaluator


class Evaluate(object):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Evaluate the specified model and dataset.'
        )
        subparser.add_argument("--gold_path", required=True, help="gold test data dir")
        subparser.add_argument("--save_path", required=True, help="path to save the model")
        subparser.add_argument("--batch_size", type=int, default=10, help="batch size")
        subparser.set_defaults(func=self)

        return subparser

    def __call__(self, args):
        # read test
        print("loading datasets...")
        test = Corpus(args.gold_path)
        print(test)

        # reload parser
        print("reloading parser...")
        vocab_path = os.path.join(args.save_path, "vocab.pt")
        state_path = os.path.join(args.save_path, "parser.pt")
        config_path = os.path.join(args.save_path, "config.json")
        ucca_parser = UCCA_Parser.load(vocab_path, config_path, state_path)

        test_loader = Data.DataLoader(
            dataset=test.generate_inputs(ucca_parser.vocab, False),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        print("evaluating test data : %s" % (args.gold_path))
        ucca_evaluator = UCCA_Evaluator(
            parser=ucca_parser,
            gold_dic=args.gold_path,
        )
        ucca_evaluator.compute_accuracy(test_loader)
        ucca_evaluator.remove_temp()
