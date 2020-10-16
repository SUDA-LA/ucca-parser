import json
import unicodedata
from argparse import Namespace


def get_config(config_filepath):
    with open(config_filepath, "r") as config_file:
        conf = json.load(config_file, object_hook=lambda d: Namespace(**d))
    return conf


def is_punct(word):
    return all(unicodedata.category(char).startswith('P') for char in word)
