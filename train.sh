#!/bin/bash

train_path=/data/wjiang/UCCA/train-dev-data/train-xml/UCCA_English-Wiki
dev_path=/data/wjiang/UCCA/train-dev-data/dev-xml/UCCA_English-Wiki
emb_path=/data/wjiang/data/embedding/cc.en.300.vec

dic=/data/wjiang/UCCA/UCCA-Parser-pre/experiment/english-pre
log_file=$dic/log.train
pred_path=$dic/predict-dev
save_file=$dic/
type=topdown

nohup python -u train.py --train_path=$train_path --dev_path=$dev_path --emb_path=$emb_path --type=$type --save_path=$save_file --pred_path=$pred_path > $log_file  2>&1 &
