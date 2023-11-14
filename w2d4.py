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

from ipdb import set_trace as p

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


def mask_scores(attn_scores):
    """Mask the attention scores so that tokens don't attend to previous tokens."""
    mask = t.tril(t.ones_like(attn_scores)).bool()
    neg_inf = t.tensor(-10000.0).to(attn_scores.device)
    masked_attn_scores = t.where(mask, attn_scores, neg_inf)
    return masked_attn_scores


def QK_attn(W_QK, qk_input):
    """
    W_QK: (query_d_model, key_d_model)
    qk_input: (position, d_model)
    """
    "TODO: YOUR CODE HERE"
    attn_scores = (qk_input @ W_QK @ qk_input.T) / cfg['d_head'] ** 0.5
    attn_scores = mask_scores(attn_scores)
    out = t.softmax(attn_scores, dim=-1)
    return out


def run_QK_attn():
    layer = 0
    head_index = 0
    batch_index = 0
    W_Q = model.blocks[layer].attn.W_Q[head_index]
    W_K = model.blocks[layer].attn.W_K[head_index]
    qk_input = cache_example[f"blocks.{layer}.hook_resid_pre"][batch_index] + cache_example["hook_pos_embed"]
    original_attn_pattern = cache_example[f"blocks.{layer}.attn.hook_attn"][batch_index, head_index, :, :]
    W_QK = W_Q.T @ W_K
    return (QK_attn, W_QK, qk_input, original_attn_pattern)


if MAIN:
    w2d4_test.test_qk_attn(*run_QK_attn())


def OV_result_mix_before(W_OV, residual_stream_pre, attn_pattern):
    """
    Apply attention to the residual stream, and THEN apply W_OV.
    Inputs:
        W_OV: (d_model, d_model)
        residual_stream_pre: (position, d_model)
        attn_pattern: (query_pos, key_pos)
    Returns:
        head output of shape: (position, d_model)
    """
    return attn_pattern @ residual_stream_pre @ W_OV


def run_OV_result_mix_before():
    layer = 0
    head_index = 0
    batch_index = 0
    W_O = model.blocks[layer].attn.W_O[head_index].detach().clone()
    W_V = model.blocks[layer].attn.W_V[head_index].detach().clone()
    W_OV = W_O @ W_V
    residual_stream_pre = cache_example[f"blocks.{layer}.hook_resid_pre"][batch_index].detach().clone()
    original_head_results = (
        cache_example[f"blocks.{layer}.attn.hook_result"][batch_index, :, head_index].detach().clone()
    )
    attn_pattern = cache_example[f"blocks.{layer}.attn.hook_attn"][batch_index, head_index, :, :].detach().clone()
    return (OV_result_mix_before, W_OV, residual_stream_pre, attn_pattern, original_head_results)


if MAIN:
    (OV_result_mix_before, W_OV, residual_stream_pre, attn_pattern, original_head_results) = run_OV_result_mix_before()
    w2d4_test.test_ov_result_mix_before(
        OV_result_mix_before, W_OV, residual_stream_pre, attn_pattern, original_head_results
    )