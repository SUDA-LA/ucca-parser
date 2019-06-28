# UCCA Parser

An implementation of "[HLT@SUDA at SemEval 2019 Task 1: UCCA Graph Parsing as Constituent Tree Parsing](https://arxiv.org/abs/1903.04153)".

This version of the implementation use lexical features in the corpus and bert features.

## Requirements

```txt
python >= 3.6.0
pytorch == 1.0.0
ucca == 1.0.127
pytorch_pretrained_bert == 0.6.2
```

Note that the code has not been tested on the newest version of ucca module.

## Datasets

The datasets are all provided by SemEval-2019 Task 1: Cross-lingual Semantic Parsing with UCCA. The official website is https://competitions.codalab.org/competitions/19160. 

Pre-trained embeddings: [http://fasttext.cc](http://fasttext.cc/)

Pre-trained bert model: This version uses base-multilingual-cased bert from pytorch_pretrained_bert. To run the code, please download parameter and vocab files from https://github.com/huggingface/pytorch-pretrained-BERT.

## Performance

Here are the results on French dataset I re-run on June 28, 2019.

| description          | dev primary | dev remote | dev average | test 20K    primary | test 20K remote | test 20K average |
| -------------------- | ----------- | ---------- | ----------- | ------------------- | --------------- | ---------------- |
| multilingual-lexical-bert |         |        |         |                 |             |              |




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

Note that the path to save the model is a directory. After training, there are three files in the directory which are named "config.json"、"vocab.pt" and "parser.pt".

```sh
$ python run.py train -h
usage: run.py train [-h] --en_train_path EN_TRAIN_PATH --fr_train_path
                    FR_TRAIN_PATH --de_train_path DE_TRAIN_PATH --en_dev_path
                    EN_DEV_PATH --fr_dev_path FR_DEV_PATH --de_dev_path
                    DE_DEV_PATH --save_path SAVE_PATH --config_path
                    CONFIG_PATH [--en_test_wiki_path EN_TEST_WIKI_PATH]
                    [--en_test_20k_path EN_TEST_20K_PATH]
                    [--fr_test_20k_path FR_TEST_20K_PATH]
                    [--de_test_20k_path DE_TEST_20K_PATH] [--gpu GPU]
                    [--seed SEED] [--threads THREADS]

optional arguments:
  -h, --help            show this help message and exit
  --en_train_path EN_TRAIN_PATH
                        en train data dir
  --fr_train_path FR_TRAIN_PATH
                        fr train data dir
  --de_train_path DE_TRAIN_PATH
                        de train data dir
  --en_dev_path EN_DEV_PATH
                        en dev data dir
  --fr_dev_path FR_DEV_PATH
                        fr dev data dir
  --de_dev_path DE_DEV_PATH
                        de dev data dir
  --save_path SAVE_PATH
                        dic to save all file
  --config_path CONFIG_PATH
                        init config file
  --en_test_wiki_path EN_TEST_WIKI_PATH
                        en wiki test data dir
  --en_test_20k_path EN_TEST_20K_PATH
                        en 20k data dir
  --fr_test_20k_path FR_TEST_20K_PATH
                        fr 20k data dir
  --de_test_20k_path DE_TEST_20K_PATH
                        de 20k data dir
  --gpu GPU             gpu id
  --seed SEED           random seed
  --threads THREADS     thread num


$ python run.py evaluate -h
usage: run.py evaluate [-h] --gold_path GOLD_PATH --save_path SAVE_PATH
                       [--batch_size BATCH_SIZE] [--language {en,fr,de}]
                       [--gpu GPU] [--seed SEED] [--threads THREADS]

optional arguments:
  -h, --help            show this help message and exit
  --gold_path GOLD_PATH
                        gold test data dir
  --save_path SAVE_PATH
                        path to save the model
  --batch_size BATCH_SIZE
                        batch size
  --language {en,fr,de}
                        language
  --gpu GPU             gpu id
  --seed SEED           random seed
  --threads THREADS     thread num


$ python run.py predict -h
usage: run.py predict [-h] --test_path TEST_PATH --save_path SAVE_PATH
                      --pred_path PRED_PATH [--batch_size BATCH_SIZE]
                      [--language {en,fr,de}] [--gpu GPU] [--seed SEED]
                      [--threads THREADS]

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
  --language {en,fr,de}
                        language
  --gpu GPU             gpu id
  --seed SEED           random seed
  --threads THREADS     thread num

```

## Convertion

Convertion codes are included in `parser.convert`.  The function `UCCA2tree` is used to convert a ucca passage to a tree. The function `to_UCCA` is used to convert a tree to a UCCA passage. Remote edges recovery codes are included in `parser.submodel.remote_parser.py` independently.
