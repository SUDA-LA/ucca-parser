#!/bin/bash

# train_path=./debug/train
# dev_path=./debug/dev
# emb_path=/data/wjiang/data/embedding/debug
# dic=./debug/result
train_path=/data/wjiang/UCCA/train-dev-data/train-xml/UCCA_English-Wiki
dev_path=/data/wjiang/UCCA/train-dev-data/dev-xml/UCCA_English-Wiki
emb_path=/data/wjiang/data/embedding/cc.en.300.vec
dic=./experiment/english-topdown-lstm

save_file=$dic/
gpu=7

if [ ! -d "$dic" ]; then
    mkdir "$dic"
fi

python -u train.py \
    --gpu=$gpu \
    --save_path=$save_file \
    --train_path=$train_path \
    --dev_path=$dev_path \
    --emb_path=$emb_path