#!/bin/bash

# train_path=./train
# dev_path=./dev
# emb_path=/data/zhangyu/ucca-parser/data/debug
# dic=./debug
train_path=./train-dev-data/train-xml/UCCA_English-Wiki
dev_path=./train-dev-data/dev-xml/UCCA_English-Wiki
emb_path=../ucca-parser/data/cc.en.300.vec

dic=./experiment/english

log_file=$dic/log.train
pred_path=$dic/predict-dev
save_file=$dic/
type=topdown

nohup python -u train.py --save_path=$save_file --train_path=$train_path --dev_path=$dev_path --emb_path=$emb_path --type=$type  --pred_path=$pred_path > $log_file  2>&1 &
