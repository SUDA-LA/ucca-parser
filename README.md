# UCCA Parser

An implementation of "[HLT@SUDA at SemEval 2019 Task 1: UCCA Graph Parsing as Constituent Tree Parsing](https://arxiv.org/abs/1903.04153)".

This version of the implementation uses lexical features in the corpus and BERT embeddings.

## Requirements

```txt
python >= 3.6.0
pytorch == 1.0.0
ucca == 1.0.127
pytorch-pretrained-bert == 0.6.2
```

Note that the code has not been tested on the newest version of ucca module.

## Datasets

The datasets are all provided by SemEval-2019 Task 1: Cross-lingual Semantic Parsing with UCCA. The official website is https://competitions.codalab.org/competitions/19160. 

Pre-trained embeddings: [http://fasttext.cc](http://fasttext.cc/)

Pre-trained BERT : This version uses cased-multilingual-base BERT from [https://github.com/huggingface/pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT). To run the code, please download parameter and vocab files first and change paths in the config.json.

## Performance
Here are the results I re-ran on July 1, 2019, which are almost the same with the results in the paper.

| description              | dev primary | dev remote | dev average | test wiki primary | test wiki remote | test wiki average | test 20K    primary | test 20K remote | test 20K average |
| ------------------------ | ----------- | ---------- | ----------- | ----------------- | ---------------- | ----------------- | ------------------- | --------------- | ---------------- |
| English-Topdown-Lexical-BERT | 82.7        | 55.6       | 82.2        | 81.2              | 57.9             | 80.8              | 77.6                | 31.0            | 76.6             |
| German-Topdown-Lexical-BERT  | 84.7        | 60.5       | 84.2        | /                 | /                | /                 | 85.2                | 63.3            | 84.7             |

## Usage

You can start the training, evaluation and prediction process by using subcommands registered in `parser.cmds` or just use the shell scripts included in.

```sh
$ python run.py -h
usage: run.py [-h] {train,predict,evaluate} ...

UCCA Parser.

optional arguments:
  -h, --help            show this help message and exit

Commands:
  {train,predict,evaluate}
    train               Train a model.
    predict             Use a trained model to make predictions.
    evaluate            Evaluate the specified model and dataset.
```

Optional arguments of the subparsers are as follows:

Note that the path to save the model is a directory. After training, there are three files in the directory which are named "config.json", "vocab.pt" and "parser.pt".

```sh
$ python run.py train -h
usage: run.py train [-h] --train_path TRAIN_PATH --dev_path DEV_PATH
                    [--emb_path EMB_PATH] --save_path SAVE_PATH --config_path
                    CONFIG_PATH [--test_wiki_path TEST_WIKI_PATH]
                    [--test_20k_path TEST_20K_PATH] [--gpu GPU] [--seed SEED]
                    [--threads THREADS]

optional arguments:
  -h, --help            show this help message and exit
  --train_path TRAIN_PATH
                        train data dir
  --dev_path DEV_PATH   dev data dir
  --emb_path EMB_PATH   pretrained embedding path
  --save_path SAVE_PATH
                        dic to save all file
  --config_path CONFIG_PATH
                        dic to save all file
  --test_wiki_path TEST_WIKI_PATH
                        wiki test data dir
  --test_20k_path TEST_20K_PATH
                        20k data dir
  --gpu GPU             gpu id
  --seed SEED           random seed
  --threads THREADS     thread num


$ python run.py evaluate -h
usage: run.py evaluate [-h] --gold_path GOLD_PATH --save_path SAVE_PATH
                       [--batch_size BATCH_SIZE] [--gpu GPU] [--seed SEED]
                       [--threads THREADS]

optional arguments:
  -h, --help            show this help message and exit
  --gold_path GOLD_PATH
                        gold test data dir
  --save_path SAVE_PATH
                        path to save the model
  --batch_size BATCH_SIZE
                        batch size
  --gpu GPU             gpu id
  --seed SEED           random seed
  --threads THREADS     thread num


$ python run.py predict -h
usage: run.py predict [-h] --test_path TEST_PATH --save_path SAVE_PATH
                      --pred_path PRED_PATH [--batch_size BATCH_SIZE]
                      [--gpu GPU] [--seed SEED] [--threads THREADS]

optional arguments:
  -h, --help            show this help message and exit
  --test_path TEST_PATH
                        test data dir
  --save_path SAVE_PATH
                        path to save the model
  --pred_path PRED_PATH
                        save predict passages
  --batch_size BATCH_SIZE
                        batch size
  --gpu GPU             gpu id
  --seed SEED           random seed
  --threads THREADS     thread num

```

## Conversion

Conversion code is included in `parser.convert`.  The function `UCCA2tree` is used to convert a UCCA passage to a tree. The function `to_UCCA` is used to convert a tree to a UCCA passage. Remote edge recovery code is included in `parser.submodel.remote_parser.py` independently.
