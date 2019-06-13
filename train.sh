#!/bin/bash

en_train_path=/data/wjiang/UCCA/train-dev-data/train-xml/UCCA_English-Wiki
fr_train_path=/data/wjiang/UCCA/train-dev-data/train-xml/UCCA_French-20K
de_train_path=/data/wjiang/UCCA/train-dev-data/train-xml/UCCA_German-20K

en_dev_path=/data/wjiang/UCCA/train-dev-data/dev-xml/UCCA_English-Wiki
fr_dev_path=/data/wjiang/UCCA/train-dev-data/dev-xml/UCCA_French-20K
de_dev_path=/data/wjiang/UCCA/train-dev-data/dev-xml/UCCA_German-20K

save_path=./exp/multilingual-lexical/debug
config_path=./config.json

en_test_wiki_path=/data/wjiang/UCCA/test-data/test-xml-gold/UCCA_English-Wiki
en_test_20k_path=/data/wjiang/UCCA/test-data/test-xml-gold/UCCA_English-20K
fr_test_20k_path=/data/wjiang/UCCA/test-data/test-xml-gold/UCCA_French-20K
de_test_20k_path=/data/wjiang/UCCA/test-data/test-xml-gold/UCCA_German-20K


gpu=-1

if [ ! -d "$save_path" ]; then
    mkdir "$save_path"
fi

python -u run.py train\
    --gpu=$gpu \
    --save_path=$save_path \
    --en_train_path=$en_train_path \
    --fr_train_path=$fr_train_path \
    --de_train_path=$de_train_path \
    --en_dev_path=$en_dev_path \
    --fr_dev_path=$fr_dev_path \
    --de_dev_path=$de_dev_path \
    --en_test_wiki_path=$en_test_wiki_path \
    --en_test_20k_path=$en_test_20k_path \
    --fr_test_20k_path=$fr_test_20k_path \
    --de_test_20k_path=$de_test_20k_path \
    --config_path=$config_path