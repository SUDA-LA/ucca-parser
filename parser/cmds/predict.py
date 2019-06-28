import argparse
import os
import datetime
from parser import UCCA_Parser

import torch
import torch.utils.data as Data
from ucca.convert import passage2file

from parser.utils import Corpus, collate_fn


@torch.no_grad()
def write_test(parser, test, path):
    parser.eval()

    test_predicted = []
    for batch in test:
        subword_idxs, subword_masks, token_starts_masks, lang_idxs, word_idxs, pos_idxs, dep_idxs, ent_idxs, ent_iob_idxs, passages, trees, all_nodes, all_remote = (
            batch
        )
        if torch.cuda.is_available():
            subword_idxs = subword_idxs.cuda()
            subword_masks = subword_masks.cuda()
            token_starts_masks = token_starts_masks.cuda()
            lang_idxs = lang_idxs.cuda()
            word_idxs = word_idxs.cuda()
            pos_idxs = pos_idxs.cuda()
            dep_idxs = dep_idxs.cuda()
            ent_idxs = ent_idxs.cuda()
            ent_iob_idxs = ent_iob_idxs.cuda()
        pred_passages = parser.parse(subword_idxs, subword_masks, token_starts_masks, lang_idxs, word_idxs, pos_idxs, dep_idxs, ent_idxs, ent_iob_idxs, passages)
        test_predicted.extend(pred_passages)

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print(e)

    for passage in test_predicted:
        passage2file(passage, os.path.join(path, passage.ID + ".xml"))


class Predict(object):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Use a trained model to make predictions.'
        )
        subparser.add_argument("--test_path", required=True, help="test data dir")
        subparser.add_argument("--save_path", required=True, help="path to save the model")
        subparser.add_argument("--pred_path", required=True, help="save predict passages")
        subparser.add_argument("--batch_size", type=int, default=10, help="batch size")
        subparser.add_argument("--language", default="fr", choices=["en", "fr", "de"], help="language")
        subparser.set_defaults(func=self)

        return subparser

    def __call__(self, args):
        # read test
        print("loading datasets...")
        test = Corpus(args.test_path, args.language)
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

        print("predicting test files...")
        start_time = datetime.datetime.now()
        write_test(ucca_parser, test_loader, args.pred_path)
        end_time = datetime.datetime.now()
        print("parsing time is " + str(end_time - start_time) + "\n")
