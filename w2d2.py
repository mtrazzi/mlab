# %%%

import hashlib
import os
import re
import sys
import tarfile
from dataclasses import dataclass
import requests
import torch as t
import transformers
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

MAIN = __name__ == "__main__"
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_FOLDER = "./data/w2d2/"
IMDB_PATH = os.path.join(DATA_FOLDER, "acllmdb_v1.tar.gz")
SAVED_TOKENS_PATH = os.path.join(DATA_FOLDER, "tokens.pt")
device = t.device("cuda" if t.cuda.is_available() else "cpu")
IS_CI = os.getenv("IS_CI")
if IS_CI:
    sys.exit(0)

# %%%


def maybe_download(url: str, path: str) -> None:
    """Download the file from url and save it to path. If path already exists, do nothing."""
    if os.path.exists(path):
        return
    print(f"Downloading {url} to {path}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                f.write(chunk)
    print("Done!")


if MAIN:
    os.makedirs(DATA_FOLDER, exist_ok=True)
    expected_hexdigest = "7c2ac02c03563afcf9b574c7e56c153a"
    maybe_download(IMDB_URL, IMDB_PATH)
    with open(IMDB_PATH, "rb") as f:
        actual_hexdigest = hashlib.md5(f.read()).hexdigest()
        assert actual_hexdigest == expected_hexdigest

# %%


@dataclass(frozen=True)
class Review:
    split: str
    is_positive: bool
    stars: int
    text: str


def load_reviews(path: str) -> list[Review]:
    "SOLUTION"
    reviews = []
    tar = tarfile.open(path, "r:gz")
    for info in tqdm(tar.getmembers()):
        m = re.match(r"aclImdb/(train|test)/(pos|neg)/\d+_(\d+)\.txt", info.name)
        if m is not None:
            split, posneg, stars = m.groups()
            buf = tar.extractfile(info)
            assert buf is not None
            text = buf.read().decode("utf-8")
            reviews.append(Review(split, posneg == "pos", int(stars), text))
    return reviews


reviews = []
if MAIN:
    reviews = load_reviews(IMDB_PATH)
    assert sum(r.split == "train" for r in reviews) == 25000
    assert sum(r.split == "test" for r in reviews) == 25000

# %%


def to_dataset(tokenizer, reviews: list[Review]) -> TensorDataset:
    """Tokenize the reviews (which should all belong to the same split) and bundle into a TensorDataset.

    The tensors in the TensorDataset should be (in this exact order):

    input_ids: shape (batch, sequence length), dtype int64
    attention_mask: shape (batch, sequence_length), dtype int
    sentiment_labels: shape (batch, ), dtype int
    star_labels: shape (batch, ), dtype int
    """
    text = [r.text for r in reviews]
    sentiment_labels = t.tensor([r.is_positive for r in reviews])
    star_labels = t.tensor([r.stars for r in reviews])
    d = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    return TensorDataset(d["input_ids"], d["attention_mask"], sentiment_labels, star_labels)


if MAIN:
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    train_data = to_dataset(tokenizer, [r for r in reviews if r.split == "train"])
    test_data = to_dataset(tokenizer, [r for r in reviews if r.split == "test"])
    t.save((train_data, test_data), SAVED_TOKENS_PATH)

# %%
