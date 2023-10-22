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
from ipdb import set_trace as p

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


def flat(x: t.Tensor) -> t.Tensor:
    """Helper function for combining batch and sequence dimensions."""
    return rearrange(x, "b s ... -> (b s) ...")


def unflat(x: t.Tensor, max_seq: int) -> t.Tensor:
    """Helper function for separating batch and sequence dimensions."""
    return rearrange(x, "(b s) ... -> b s ...", s=max_seq)


def random_mask(
    input_ids: t.Tensor, mask_token_id: int, vocab_size: int, select_frac=0.15, mask_frac=0.8, random_frac=0.1
) -> tuple[t.Tensor, t.Tensor]:
    """Given a batch of tokens, return a copy with tokens replaced according to Section 3.1 of the paper.

    input_ids:  (batch, seq)

    Return: (model_input, was_selected) where:

    model_input: (batch, seq) - a new Tensor with the replacements made, suitable for passing to the BertLanguageModel. Don't modify the original tensor!

    was_selected: (batch, seq) - 1 if the token at this index will contribute to the MLM loss, 0 otherwise
    """
    # .to(input_ids.device) at some point
    assert random_frac < (1 - mask_frac)
    try:
        max_seq = input_ids.shape[1]
        flat_input = flat(input_ids)
        rand_ten = t.rand(flat_input.shape[0])
        selected = rand_ten < select_frac
        to_mask = (rand_ten < mask_frac * select_frac)
        to_rand = (rand_ten > mask_frac * select_frac) & (rand_ten < (mask_frac + random_frac) * select_frac)
        model_input = flat_input.clone().detach()
        model_input[t.nonzero(to_mask).flatten()] = mask_token_id
        model_input[t.nonzero(to_rand).flatten()] = t.randint(0, vocab_size, (t.nonzero(to_rand).flatten()).shape)
    except Exception as e:
        print(e)
        import ipdb; ipdb.set_trace()
    return unflat(model_input, max_seq=max_seq), unflat(selected, max_seq=max_seq)


# if MAIN:
#     w2d2_test.test_random_mask(random_mask, input_size=1000000, max_seq=max_seq)

if MAIN:
    # Cross Entropy Loss Of Unigram Predictions
    all_tokens = t.concat([train_data, val_data, test_data]).flatten()
    token_counts = t.bincount(all_tokens)
    unigram = token_counts / sum(token_counts)
    print(f"Loss of unigram is {t.distributions.Categorical(unigram).entropy()}")
# %%

def cross_entropy_selected(pred: t.Tensor, target: t.Tensor, was_selected: t.Tensor) -> t.Tensor:
    """
    pred: (batch, seq, vocab_size) - predictions from the model
    target: (batch, seq, ) - the original (not masked) input ids
    was_selected: (batch, seq) - 1 if the token at this index will contribute to the MLM loss, 0 otherwise

    Out: the mean loss per predicted token
    """
    if t.all(was_selected == 0):
        return t.tensor([t.nan])
    idxs = t.nonzero(flat(was_selected)).flatten()
    # here by using the cross_entropy on the problematic indexes we're dividing by the number of predicted
    # since the cross_entropy function divides by the number of "batches"
    loss = t.nn.functional.cross_entropy(flat(pred)[idxs], flat(target)[idxs])
    return loss


if MAIN:
    w2d2_test.test_cross_entropy_selected(cross_entropy_selected)
if MAIN and (not IS_CI):
    batch_size = 8
    seq_length = 512
    batch = t.randint(0, tokenizer.vocab_size, (batch_size, seq_length))
    pred = t.rand((batch_size, seq_length, tokenizer.vocab_size))
    (masked, was_selected) = random_mask(batch, tokenizer.mask_token_id, tokenizer.vocab_size)
    loss = cross_entropy_selected(pred, batch, was_selected).item()
    print(f"Random MLM loss on random tokens - does this make sense? {loss:.2f}")