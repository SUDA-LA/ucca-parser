#!/bin/bash

# train_path=./debug/train
# dev_path=./debug/dev
# emb_path=/data/wjiang/data/embedding/debug
# dic=./debug/result
train_path=/data/wjiang/UCCA/train-dev-data/train-xml/UCCA_English-Wiki
dev_path=/data/wjiang/UCCA/train-dev-data/dev-xml/UCCA_English-Wiki
emb_path=/data/wjiang/data/embedding/cc.en.300.vec
dic=./experiment/localloss

log_file=$dic/log.train
pred_path=$dic/predict-dev
save_file=$dic/
gpu=3
type=chart

nohup python -u train.py  --gpu=$gpu --save_path=$save_file --train_path=$train_path --dev_path=$dev_path --emb_path=$emb_path --type=$type  --pred_path=$pred_path > $log_file  2>&1 &
