import os
import subprocess
import torch
from ucca.convert import passage2file
import tempfile


def write_passages(dev_predicted, path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print(e)

    for passage in dev_predicted:
        passage2file(passage, os.path.join(path, passage.ID + ".xml"))


class UCCA_Evaluator(object):
    def __init__(
        self, parser, gold_dic=None, pred_dic=None,
    ):
        self.parser = parser
        self.gold_dic = gold_dic
        self.pred_dic = pred_dic
        self.temp_pred_dic = tempfile.TemporaryDirectory(prefix="ucca-eval-")
        self.best_F = 0

    @torch.no_grad()
    def predict(self, loader):
        self.parser.eval()
        predicted = []
        for batch in loader:
            word_idxs, pos_idxs, dep_idxs, ent_idxs, ent_iob_idxs, passages, trees, all_nodes, all_remote = batch
            word_idxs = word_idxs.cuda() if torch.cuda.is_available() else word_idxs
            pos_idxs = pos_idxs.cuda() if torch.cuda.is_available() else pos_idxs
            dep_idxs = dep_idxs.cuda() if torch.cuda.is_available() else dep_idxs
            ent_idxs = ent_idxs.cuda() if torch.cuda.is_available() else ent_idxs
            ent_iob_idxs = ent_iob_idxs.cuda() if torch.cuda.is_available() else ent_iob_idxs

            pred_passages = self.parser.parse(word_idxs, pos_idxs, dep_idxs, ent_idxs, ent_iob_idxs, passages)
            predicted.extend(pred_passages)
        return predicted
        
    def remove_temp(self):
        self.temp_pred_dic.cleanup()

    def compute_accuracy(self, loader):
        passage_predicted = self.predict(loader)
        write_passages(passage_predicted, self.temp_pred_dic.name)

        child = subprocess.Popen(
            "python -m scripts.evaluate_standard {} {} -f".format(
                self.gold_dic, self.temp_pred_dic.name
            ),
            shell=True,
            stdout=subprocess.PIPE,
        )
        eval_info = str(child.communicate()[0], encoding="utf-8")
        try:
            Fscore = eval_info.strip().split("\n")[-1]
            Fscore = Fscore.strip().split()[-1]
            Fscore = float(Fscore)
            print("Fscore={}".format(Fscore))
        except IndexError:
            print("Unable to get FScore. Skipping.")
            Fscore = 0

        if Fscore > self.best_F:
            print('\n'.join(eval_info.split('\n')[1:]))
            self.best_F = Fscore
            if self.pred_dic:
                write_passages(passage_predicted, self.pred_dic)
        return Fscore