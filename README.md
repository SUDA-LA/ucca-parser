# UCCA Parser

An implementation of "[HLT@SUDA at SemEval 2019 Task 1: UCCA Graph Parsing as Constituent Tree Parsing](https://arxiv.org/abs/1903.04153)".

For simplicity, this version of the implementation does not use any extra feature in the corpus, except pre-trained embeddings. We use Char-LSTM in place of embeddings of POS, dependency labels and entity labels, resulting in about 0.4% drop on the results of test data. 

Other model choices such as using chart decoding or using self-attentive encoder are also included in this version. You can choose in config.json.

## Requirements

```txt
python >= 3.6.0
pytorch == 1.0.0
ucca == 1.0.127
```

Node that the code has not been tested on the newest version of ucca module.

## Datasets

The datasets are all provided by SemEval-2019 Task 1: Cross-lingual Semantic Parsing with UCCA. The official website is https://competitions.codalab.org/competitions/19160. 

Pre-trained embeddings: [http://fasttext.cc](http://fasttext.cc/)

## Performance

| description              | dev primary | dev remote | dev average | test wiki primary | test wiki remote | test wiki average | test 20K    primary | test 20K remote | test 20K average |
| ------------------------ | ----------- | ---------- | ----------- | ----------------- | ---------------- | ----------------- | ------------------- | --------------- | ---------------- |
| English-Topdown-CharLSTM | 79.5        | 43.8       | **79.0**    | 77.6              | 48.0             | **77.2**          | 73.3                | 19.0            | **72.3**         |
| German-Topdown-CharLSTM  | 82.9        | 51.4       | 82.3        | 83.5              | 57.6             | 83.0              | /                   | /               | /                |

## Usage

You can start the training, evaluation and prediction process by using subcommands registered in `parser.cmds`.

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

Note that the path to save the model is a directory. After training, there are three files in the directory which are named "config.json"、"vocab.pt" and "parser.pt".

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



