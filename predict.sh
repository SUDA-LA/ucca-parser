#!/bin/bash


test_path=/data/wjiang/UCCA/test-data/test-xml/UCCA_English-Wiki
save_path=./exp/lexical/english
pred_path=$save_path/UCCA_English-Wiki

gpu=2
python -u run.py predict\
    --gpu=$gpu \
    --test_path=$test_path \
    --pred_path=$pred_path \
    --save_path=$save_path
