#!/bin/bash

dic=/data/wjiang/UCCA/UCCA-Parser-pre/experiment/english-topdown
test_path=/data/wjiang/UCCA/test-data/test-xml/UCCA_English-Wiki
pred_path=$dic/UCCA_English-Wiki
model_path=$dic/parser-topdown.pt
log_file=$dic/log.test

nohup python -u test.py --test_path=$test_path --pred_path=$pred_path --model_path=$model_path > $log_file  2>&1 &
