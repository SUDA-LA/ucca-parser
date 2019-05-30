#!/bin/bash

gold_path=/data/wjiang/UCCA/test-data/test-xml-gold/UCCA_English-20K
save_path=./exp/baseline/english-topdown-lstm/

gpu=0
python -u run.py evaluate\
    --gpu=$gpu \
    --gold_path=$gold_path \
    --save_path=$save_path
