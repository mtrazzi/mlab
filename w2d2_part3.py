#!/usr/bin/env python3
# %%

import hashlib
import os
import sys
import zipfile
import torch as t
import transformers
from itertools import chain
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm
import w2d2_test
from w2d2_part1_data_prep_solution import maybe_download

# %load_ext autoreload
# %autoreload 2

MAIN = __name__ == "__main__"
device = t.device("cuda" if t.cuda.is_available() else "cpu")
DATA_FOLDER = "./data/w2d2"
DATASET = "2"
BASE_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/"
DATASETS = {"103": "wikitext-103-raw-v1.zip", "2": "wikitext-2-raw-v1.zip"}
TOKENS_FILENAME = os.path.join(DATA_FOLDER, f"wikitext_tokens_{DATASET}.pt")
IS_CI = os.getenv("IS_CI")

# %%

if MAIN and (not IS_CI):
    path = os.path.join(DATA_FOLDER, DATASETS[DATASET])
    maybe_download(BASE_URL + DATASETS[DATASET], path)
    expected_hexdigest = {"103": "0ca3512bd7a238be4a63ce7b434f8935", "2": "f407a2d53283fc4a49bcff21bc5f3770"}
    with open(path, "rb") as f:
        actual_hexdigest = hashlib.md5(f.read()).hexdigest()
        assert actual_hexdigest == expected_hexdigest[DATASET]
if MAIN:
    print(f"Using dataset WikiText-{DATASET} - options are 2 and 103")
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
if MAIN and (not IS_CI):
    z = zipfile.ZipFile(path)

    def decompress(split: str) -> str:
        return z.read(f"wikitext-{DATASET}-raw/wiki.{split}.raw").decode("utf-8")

    train_text = decompress("train").splitlines()
    val_text = decompress("valid").splitlines()
    test_text = decompress("test").splitlines()
# %%


def tokenize_1d(tokenizer, lines: list[str], max_seq: int) -> t.Tensor:
    """Tokenize text and rearrange into chunks of the maximum length.

    Return (batch, seq) and an integer dtype
    """
    tokens = tokenizer(lines, truncation=False)
    flattened_list = list(chain.from_iterable(tokens["input_ids"]))
    truncated_idx = len(flattened_list) - (len(flattened_list) % max_seq)
    truncated_list = flattened_list[:truncated_idx]
    return t.tensor(truncated_list).reshape(-1, max_seq)


if MAIN and (not IS_CI):
    max_seq = 128
    print("Tokenizing training text...")
    train_data = tokenize_1d(tokenizer, train_text, max_seq)
    print("Training data shape is: ", train_data.shape)
    print("Tokenizing validation text...")
    val_data = tokenize_1d(tokenizer, val_text, max_seq)
    print("Tokenizing test text...")
    test_data = tokenize_1d(tokenizer, test_text, max_seq)
    print("Saving tokens to: ", TOKENS_FILENAME)
    t.save((train_data, val_data, test_data), TOKENS_FILENAME)
