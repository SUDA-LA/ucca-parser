#!/bin/bash


test_path=/data/wjiang/UCCA/test-data/test-xml/UCCA_French-20K
save_path=./exp/multilingual-lexical-bert/french
pred_path=$save_path/UCCA_French-20K
language=fr

gpu=1
python -u run.py predict\
    --gpu=$gpu \
    --test_path=$test_path \
    --pred_path=$pred_path \
    --save_path=$save_path \
    --language=$language
