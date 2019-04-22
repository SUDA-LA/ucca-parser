#!/bin/bash


# dic=/data/wjiang/UCCA/UCCA-Parser-pre/experiment/english-topdown
dic=./experiment/english-topdown-lstm/
test_path=/data/wjiang/UCCA/test-data/test-xml/UCCA_English-Wiki
pred_path=$dic/UCCA_English-Wiki
model_path=$dic

gpu=6
python -u test.py \
    --gpu=$gpu \
    --test_path=$test_path \
    --pred_path=$pred_path \
    --model_path=$model_path
