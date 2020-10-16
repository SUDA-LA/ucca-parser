import argparse
import os
from ucca_parser.cmds import Train, Predict, Evaluate

import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='UCCA Parser.'
    )
    subparsers = parser.add_subparsers(title='Commands')
    subcommands = {
        'train': Train(),
        'predict': Predict(),
        'evaluate': Evaluate(),
    }
    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        subparser.add_argument("--gpu", default=-1, help="gpu id")
        subparser.add_argument("--seed", type=int, default=1, help="random seed")
        subparser.add_argument("--threads", type=int, default=1, help="thread num")
    args = parser.parse_args()

    print("Set the max num of threads to %d" % (args.threads))
    print("Set the seed for generating random numbers to %d" % (args.seed))
    print("Set the device with ID %s visible" % (args.gpu))
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.func(args)
