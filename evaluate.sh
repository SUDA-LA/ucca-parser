#!/bin/bash

gold_path=/data/wjiang/UCCA/test-data/test-xml-gold/UCCA_English-Wiki
save_path=./exp/lexical-bert/english/

gpu=2
python -u run.py evaluate\
    --gpu=$gpu \
    --gold_path=$gold_path \
    --save_path=$save_path
