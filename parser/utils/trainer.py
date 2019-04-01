import datetime
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from ucca import evaluation
from ucca.convert import passage2file


@torch.no_grad()
def check_dev(parser, dev):
    parser.eval()

    dev_gold = []
    dev_predicted = []
    for batch in dev:
        word_idxs, char_idxs, passages, trees, all_nodes, all_remote = (
            batch
        )
        word_idxs = word_idxs.cuda() if torch.cuda.is_available() else word_idxs
        char_idxs = char_idxs.cuda() if torch.cuda.is_available() else char_idxs

        dev_gold.extend(passages)
        pred_passages = parser.parse(word_idxs, char_idxs, passages)
        dev_predicted.extend(pred_passages)
    results = []
    for pred, gold in zip(dev_predicted, dev_gold):
        results.append(evaluation.evaluate(pred, gold))
    summary = evaluation.Scores.aggregate(results)
    return summary, dev_predicted


def write_dev(dev_predicted, path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print(e)

    for passage in dev_predicted:
        passage2file(passage, os.path.join(path, passage.ID + ".xml"))


def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string


class Trainer(object):
    def __init__(self, parser, optimizer, batch_size, epoch, patience, path):
        self.parser = parser
        self.optimizer = optimizer
        self.save_path = os.path.join(path, "parser.pt")
        self.pred_path = os.path.join(path, "predict-dev")
        self.batch_size = batch_size
        self.epoch = epoch
        self.patience = patience


    def train(self, train, dev):
        # record some parameters
        best_f, best_epoch, patience = 0, 0, 0

        # begin to train
        print("start to train the model ")
        for e in range(1, self.epoch + 1):
            epoch_start_time = time.time()

            self.parser.train()
            time_start = datetime.datetime.now()

            for step, batch in enumerate(train):
                self.optimizer.zero_grad()
                self.parser.zero_grad()
                span_losses, remote_losses = 0, 0
                word_idxs, char_idxs, passages, trees, all_nodes, all_remote = (
                    batch
                )
                for i in range(0, len(word_idxs), 5):
                    span_loss, remote_loss = self.parser.parse(
                        word_idxs[i : i + 5].cuda()
                        if torch.cuda.is_available()
                        else word_idxs[i : i + 5],
                        char_idxs[i : i + 5].cuda()
                        if torch.cuda.is_available()
                        else char_idxs[i : i + 5],
                        passages[i : i + 5],
                        trees[i : i + 5],
                        all_nodes[i : i + 5],
                        all_remote[i : i + 5],
                    )
                    span_losses += sum(span_loss)
                    remote_losses += sum(remote_loss)
                loss = span_losses / len(word_idxs) + remote_losses
                loss.backward()
                # nn.utils.clip_grad_norm_(self.parser.parameters(), 5.0)
                self.optimizer.step()

                print(
                    "epoch %d batch %d/%d batch-loss %f epoch-elapsed %s "
                    % (
                        e,
                        step + 1,
                        int(math.ceil(len(train.dataset) / self.batch_size)),
                        loss,
                        format_elapsed(epoch_start_time),
                    )
                )

            # save the model when dev precision get better
            summary, dev_predicted = check_dev(self.parser, dev)
            if summary.average_f1() > best_f:
                print("save the model...")
                print("the best f is %f" % (summary.average_f1()))
                summary.print()
                best_f = summary.average_f1()
                patience = 0
                best_epoch = e
                self.parser.save(self.save_path)
                write_dev(dev_predicted, self.pred_path)
            else:
                patience += 1
            # if self.encoder == 'attention':
            #     if self.optimizer.n_current_steps + 1 >self.n_warmup_steps:
            #         self.optimizer.schedule(best_f)
            time_end = datetime.datetime.now()
            print("epoch executing time is " + str(time_end - time_start) + "\n")
            if patience > self.patience:
                break

        print("train finished with epoch: %d / %d" % (e, self.epoch))
        print("the best epoch is %d , the best F1 on dev is %f" % (best_epoch, best_f))
        print(str(datetime.datetime.now()))
