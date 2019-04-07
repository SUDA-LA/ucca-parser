import argparse
import json
import os
from parser import UCCA_Parser
import torch.optim as optim
import torch
import torch.utils.data as Data

from parser.utils import Corpus, Trainer, Vocab, collate_fn, get_config, Embedding, UCCA_Evaluator, MyScheduledOptim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--train_path", required=True, help="train data dir")
    parser.add_argument("--dev_path", required=True, help="dev data dir")
    parser.add_argument("--emb_path", required=True, help="pretrained embedding path")
    parser.add_argument("--save_path", required=True, help="dic to save all file")

    parser.add_argument("--gpu", default=-1, help="gpu id")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--thread", type=int, default=1, help="thread num")

    args = parser.parse_args()

    config = get_config("./config.json")
    assert config.ucca.type in ["chart", "top-down", "global-chart"]
    assert config.ucca.encoder in ["lstm", "attention"]
    assert config.ucca.partition in [True, False]

    with open(os.path.join(args.save_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, default=lambda o: o.__dict__, indent=4)

    # choose GPU and init seed
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_num_threads(args.thread)
    torch.manual_seed(args.seed)
    print("using GPU device : %s" % args.gpu)
    print("seed = %d" % args.seed)

    # read training , dev file
    print("loading datasets and transforming to trees...")
    train = Corpus(args.train_path)
    dev = Corpus(args.dev_path)
    print(train, "\n", dev)

    # init vocab
    print("collecting words and labels in training dataset...")
    vocab = Vocab(train)
    print(vocab)
    
    # prepare pre-trained embedding
    if args.emb_path:
        print("reading pre-trained embedding...")
        pre_emb = Embedding.load(args.emb_path)
        print("pre-trained words:%d, dim=%d in %s" % (len(pre_emb), pre_emb.dim, args.emb_path))
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

    if ucca_parser.encoder == "lstm":
        optimizer = optim.Adam(ucca_parser.parameters(), lr=config.ucca.lr)
    elif ucca_parser.encoder == "attention":
        optimizer = optim.Adam(ucca_parser.parameters(), lr=config.ucca.lr)
        # optimizer = MyScheduledOptim(optimizer)

    ucca_evaluator = UCCA_Evaluator(
        parser=ucca_parser,
        gold_dev_dic=args.dev_path, 
        pred_dev_dic=os.path.join(args.save_path, "predict-dev"),
        temp_pred_dic=os.path.join(args.save_path, "temp-dev"),
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
