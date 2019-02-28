import argparse
import json
import os
from parser import UCCA_Parser

import torch
import torch.utils.data as Data

from utils import Corpus, Trainer, Vocab, collate_fn, collate_fn_cuda


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--train_path", required=True, help="train data dir")
    parser.add_argument("--dev_path", required=True, help="dev data dir")
    parser.add_argument("--emb_path", required=True, help="pretrained embedding path")
    parser.add_argument("--save_path", required=True, help="dic to save the model")
    parser.add_argument("--pred_path", required=True, help="dic to save the dev predict passages")

    parser.add_argument("--gpu", type=int, default=-1, help="gpu id")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--thread", type=int, default=4, help="thread num")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--epoch", type=int, default=100, help="max epoch")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--type", choices=["chart", "topdown"], default="topdown")
    parser.add_argument("--word_dim", type=int, default=300)
    parser.add_argument("--pos_dim", type=int, default=50)
    parser.add_argument("--dep_dim", type=int, default=50)
    parser.add_argument("--entity_dim", type=int, default=25)
    parser.add_argument("--ent_iob_dim", type=int, default=25)

    parser.add_argument("--lstm_layer", type=int, default=2)
    parser.add_argument("--lstm_dim", type=int, default=200)
    parser.add_argument("--lstm_drop", type=float, default=0.4)
    parser.add_argument("--emb_drop", type=float, default=0.5)
    parser.add_argument("--label_hidden", type=int, default=200)
    parser.add_argument("--split_hidden", type=int, default=200)
    parser.add_argument("--ffn_drop", type=float, default=0.2)
    parser.add_argument("--mlp_label_dim", type=int, default=100)
    args = parser.parse_args()
    
    with open(os.path.join(args.save_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False)

    # choose GPU and init seed
    assert args.gpu in range(-1, 8)
    if args.gpu >= 0:
        use_cuda = True
        torch.cuda.set_device(args.gpu)
        torch.set_num_threads(args.thread)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        print("using GPU device : %d" % args.gpu)
        print("GPU seed = %d" % torch.cuda.initial_seed())
        print("CPU seed = %d" % torch.initial_seed())
    else:
        use_cuda = False
        torch.set_num_threads(args.thread)
        torch.manual_seed(args.seed)
        print("CPU seed = %d" % torch.initial_seed())

    # read training , dev file
    print("loading datasets and transforming to trees...")
    train = Corpus(args.train_path)
    dev = Corpus(args.dev_path)
    print(train, "\n", dev, "\n")

    # init vocab
    print("collecting words and labels in training dataset...")
    vocab = Vocab(train)
    print(vocab)

    # init parser
    ucca_parser = UCCA_Parser(vocab, args)

    # prepare data
    train_loader = Data.DataLoader(
        dataset=train.generate_inputs(vocab, True),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn if not use_cuda else collate_fn_cuda,
    )
    dev_loader = Data.DataLoader(
        dataset=dev.generate_inputs(vocab, False),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn if not use_cuda else collate_fn_cuda,
    )

    trainer = Trainer(ucca_parser, args)
    trainer.train(train_loader, dev_loader)
