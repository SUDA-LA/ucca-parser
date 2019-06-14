#!/bin/bash

gold_path=/data/wjiang/UCCA/train-dev-data/dev-xml/UCCA_French-20K
save_path=./exp/multilingual-lexical/french/
language=fr

gpu=5
python -u run.py evaluate\
    --gpu=$gpu \
    --gold_path=$gold_path \
    --save_path=$save_path \
    --language=$language
