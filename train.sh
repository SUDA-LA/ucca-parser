#!/bin/bash

train_path=/data/wjiang/UCCA/train-dev-data/train-xml/UCCA_English-Wiki
dev_path=/data/wjiang/UCCA/train-dev-data/dev-xml/UCCA_English-Wiki
emb_path=/data/wjiang/data/embedding/cc.en.300.vec
save_path=./exp/lexical-bert/english
config_path=./config.json
test_wiki_path=/data/wjiang/UCCA/test-data/test-xml-gold/UCCA_English-Wiki
test_20k_path=/data/wjiang/UCCA/test-data/test-xml-gold/UCCA_English-20K

gpu=1

if [ ! -d "$save_path" ]; then
    mkdir "$save_path"
fi

python -u run.py train\
    --gpu=$gpu \
    --save_path=$save_path \
    --train_path=$train_path \
    --test_wiki_path=$test_wiki_path \
    --test_20k_path=$test_20k_path \
    --dev_path=$dev_path \
    --emb_path=$emb_path \
    --config_path=$config_path