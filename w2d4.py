#!/usr/bin/env python

import os
import sys
import requests
from pathlib import Path
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import w2d4_test
from w2d4_attn_only_transformer import AttnOnlyTransformer

pio.renderers.default = "notebook"
device = "cuda" if t.cuda.is_available() else "cpu"
MAIN = __name__ == "__main__"
IS_CI = os.getenv("IS_CI")
if IS_CI:
    sys.exit(0)

WEIGHT_PATH = Path("./data/w2d4/attn_only_2L.pth")
if not WEIGHT_PATH.exists():
    print(
        "Weight file not found. Try manually downloading it from https://drive.google.com/u/0/uc?id=19FQ4UQ-5vw8d-duXR9haks5xyKYPzvS8&export=download"
    )

cfg = {
    "d_model": 768,
    "d_head": 64,
    "n_heads": 12,
    "n_layers": 2,
    "n_ctx": 2048,
    "d_vocab": 50278,
    "lr": 0.001,
    "betas": (0.9, 0.95),
    "weight_decay": 0.01,
    "batch_size": 144,
    "batches_per_step": 2,
    "seed": 398,
    "dataset_name": "the_pile",
    "use_attn_result": True,
}
if MAIN:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = AttnOnlyTransformer(cfg, tokenizer)
    pretrained_weights = t.load(WEIGHT_PATH, map_location=device)
    model.load_state_dict(pretrained_weights)

if MAIN:
    cache_example = {}
    model.cache_all(cache_example, device=device)
    example_text = "IT WAS A BRIGHT cold day in April, and the clocks were striking thirteen. Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust from entering along with him."
    example_tokens = model.to_tokens(example_text)
    print(f"There are {example_tokens.shape[-1]} tokens")
    logits = model(example_tokens)
    model.reset_hooks()
    for activation_name in cache_example:
        activation = cache_example[activation_name]
        print(f"Activation: {activation_name} Shape: {activation.shape}")