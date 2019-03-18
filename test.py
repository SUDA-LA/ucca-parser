import argparse
import os
import datetime
from parser import UCCA_Parser

import torch
import torch.utils.data as Data
from ucca.convert import passage2file

from utils import Corpus, Trainer, Vocab, collate_fn, collate_fn_cuda


@torch.no_grad()
def write_test(parser, test, path):
    parser.eval()

    test_predicted = []
    for batch in test:
        word_idxs, ext_word_idxs, char_idxs, passages, trees, all_nodes, all_remote = (
            batch
        )
        pred_passages = parser.parse(word_idxs, ext_word_idxs, char_idxs, passages)
        test_predicted.extend(pred_passages)

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print(e)

    for passage in test_predicted:
        passage2file(passage, os.path.join(path, passage.ID + ".xml"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument("--test_path", required=True, help="test data dir")
    parser.add_argument("--model_path", required=True, help="path to save the model")
    parser.add_argument(
        "--pred_path", required=True, help="dic to save the dev predict passages"
    )

    parser.add_argument("--gpu", type=int, default=-1, help="gpu id")
    parser.add_argument("--thread", type=int, default=4, help="thread num")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    args = parser.parse_args()

    # choose GPU and init seed
    if args.gpu >= 0:
        use_cuda = True
        torch.cuda.set_device(args.gpu)
        torch.set_num_threads(args.thread)
        print("using GPU device : %d" % args.gpu)
    else:
        use_cuda = False
        torch.set_num_threads(args.thread)

    # read test
    print("loading datasets...")
    test = Corpus(args.test_path)
    print(test)

    device = "cuda:" + str(args.gpu) if use_cuda else "cpu"
    # reload parser
    ucca_parser = torch.load(args.model_path, map_location=device)

    test_loader = Data.DataLoader(
        dataset=test.generate_inputs(ucca_parser.vocab, False),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn if not use_cuda else collate_fn_cuda,
    )

    print("predicting test files...")
    start_time = datetime.datetime.now()
    write_test(ucca_parser, test_loader, args.pred_path)
    end_time = datetime.datetime.now()
    print("parsing time is " + str(end_time - start_time) + "\n")
