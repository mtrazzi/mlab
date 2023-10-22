#!/usr/bin/env python3

# %%

import os
import sys
import torch as t
import transformers
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm.auto import tqdm
import wandb
from w2d1 import BertConfig, BertLanguageModel, predict
from w2d2_part3 import cross_entropy_selected, random_mask

MAIN = __name__ == "__main__"
device = t.device("cuda" if t.cuda.is_available() else "cpu")
IS_CI = os.getenv("IS_CI")
if IS_CI:
    sys.exit(0)

# %%

if MAIN:
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    hidden_size = 512
    assert hidden_size % 64 == 0
    bert_config_tiny = BertConfig(
        max_position_embeddings=128,
        hidden_size=hidden_size,
        intermediate_size=4 * hidden_size,
        num_layers=8,
        num_heads=hidden_size // 64,
    )
    config_dict = dict(
        filename="./data/w2d2/bert_lm.pt",
        lr=0.0002,
        epochs=40,
        batch_size=128,
        weight_decay=0.01,
        mask_token_id=tokenizer.mask_token_id,
        warmup_step_frac=0.01,
        eps=1e-06,
        max_grad_norm=None,
    )
    (train_data, val_data, test_data) = t.load("./data/w2d2/wikitext_tokens_103.pt")
    print("Training data size: ", train_data.shape)
    train_loader = DataLoader(
        TensorDataset(train_data), shuffle=True, batch_size=config_dict["batch_size"], drop_last=True # type: ignore
    )

# %%
def lr_for_step(step: int, max_step: int, max_lr: float, warmup_step_frac: float):
    """Return the learning rate for use at this step of training."""
    warmup_steps =  int(warmup_step_frac * max_step)
    if step <= warmup_steps:
        return 0.01 * max_lr + 0.99 * ((step / warmup_steps) * max_lr)
    return max_lr - ((step - warmup_steps) / (max_step - warmup_steps)) * 0.99 * max_lr
    

if MAIN:
    max_step = int(len(train_loader) * config_dict["epochs"]) # type: ignore
    lrs = [
        lr_for_step(step, max_step, max_lr=config_dict["lr"], warmup_step_frac=config_dict["warmup_step_frac"]) # type: ignore
        for step in range(max_step) # type: ignore
    ]
    (fig, ax) = plt.subplots(figsize=(12, 4))
    ax.plot(lrs)
    ax.set(xlabel="Step", ylabel="Learning Rate", title="Learning Rate Schedule")
    plt.plot()
    plt.show()