#!/bin/bash


dic=./debug/result
# dic=/data/wjiang/UCCA/UCCA-Parser-pre/experiment/english-topdown
test_path=/data/wjiang/UCCA/test-data/test-xml/UCCA_English-Wiki
pred_path=$dic/UCCA_English-Wiki
model_path=$dic/parser-chart.pt
log_file=$dic/log.test
gpu=5

nohup python -u test.py --gpu=$gpu --test_path=$test_path --pred_path=$pred_path --model_path=$model_path > $log_file  2>&1 &
