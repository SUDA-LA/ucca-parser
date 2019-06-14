#!/bin/bash

gold_path=/data/wjiang/UCCA/test-data/test-xml-gold/UCCA_French-20K
save_path=./exp/multilingual-lexical/debug/
language=fr

gpu=1
python -u run.py evaluate\
    --gpu=$gpu \
    --gold_path=$gold_path \
    --save_path=$save_path \
    --language=$language
