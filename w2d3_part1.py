#!/usr/bin/env python

import math
import os
from dataclasses import dataclass
from typing import Any, Optional
import torch as t
import transformers
from einops import rearrange
from fancy_einsum import einsum
from torch import nn
import utils
import w2d3_test
from ipdb import set_trace as p

os.environ["TOKENIZERS_PARALLELISM"] = "false"
MAIN = __name__ == "__main__"


@dataclass(frozen=True)
class GPTConfig:
    """Constants used throughout the GPT2 model."""

    activation_function: str = "gelu"
    num_layers: int = 12
    num_heads: int = 12
    vocab_size: int = 50257
    hidden_size: int = 768
    max_position_embeddings: int = 1024
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-05


config = GPTConfig()
if MAIN:
    pretrained_gpt = utils.load_pretrained_gpt()
    # Required notes about differences
    # Layer Norms come before (compared to BERT)
    # Conv1D Layers instead of Linear layers for attention -> can just load weights into linear layers
    # Q, K and V are concatenated into c_attn.weight
    # c_proj.weight is the O matrics for all weights concatenated

class UnidirectionalAttention(nn.Module):
    qkv_proj: nn.Linear
    output_proj: nn.Linear
    attn_dropout: nn.Dropout
    resid_dropout: nn.Dropout

    def __init__(self, hidden_size: int, num_heads: int, head_size: Optional[int] = None, dropout=0.1):
        super().__init__()
        self.hidden_size, self.num_heads = hidden_size, num_heads
        assert hidden_size % num_heads == 0
        self.head_size = head_size if head_size is not None else hidden_size // num_heads
        self.qkv_proj = nn.Linear(hidden_size, 3 * num_heads * self.head_size) # 3 before [Q, K, V]
        self.output_proj = nn.Linear(self.head_size * num_heads, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def attention_pattern_pre_softmax(self, x: t.Tensor) -> tuple[t.Tensor, t.Tensor]:

        QKV = self.qkv_proj(x)
        QKV = rearrange(QKV, "b s (nh hs) -> b s nh hs", nh=self.num_heads * 3) # having Q, K and V is like having 3x the heads
        Q, K, V = QKV[:, :, :self.num_heads, :], QKV[:, :, self.num_heads:2*self.num_heads, :], QKV[:, :, 2*self.num_heads:, :]
        QK = t.einsum("bqhi,bkhi->bhqk", Q, K) / self.head_size**0.5

        mask = t.triu(t.ones(QK.shape[-2], QK.shape[-1], device=x.device), diagonal=1).to(x.device)
        mask = mask.masked_fill(mask==1, float('-1e4'))
        QK += mask
        return QK, V

    def forward(self, x: t.Tensor, cache: Optional[Any] = None) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        attn_sc, V = self.attention_pattern_pre_softmax(x)

        attn_probs = t.softmax(attn_sc, dim=-1)
        attn_drop = self.attn_dropout(attn_probs)

        values = t.einsum("bkhi,bhqk->bqhi", V, attn_probs)
        values = rearrange(values, "b s nh hs -> b s (nh hs)")
        O = self.output_proj(values)
        out = self.resid_dropout(O)
        return out


if MAIN:
    w2d3_test.test_unidirectional_attn(UnidirectionalAttention)


ACTIVATION_FUNCTIONS = dict(relu=nn.ReLU(), gelu=nn.GELU())


class GPT2Block(nn.Module):
    attn: UnidirectionalAttention
    linear1: nn.Linear
    linear2: nn.Linear
    ln1: nn.LayerNorm
    ln2: nn.LayerNorm

    def __init__(
        self, hidden_size: int, num_heads: int, dropout: float, layer_norm_epsilon: float, activation_function: str
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.attn = UnidirectionalAttention(hidden_size, num_heads)
        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.linear1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.linear2 = nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.act = ACTIVATION_FUNCTIONS[activation_function]

    def forward(self, x: t.Tensor, cache: Optional[Any] = None) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        x += self.attn(self.ln1(x))
        x += self.dropout(self.linear2(self.act(self.linear1(self.ln2(x)))))
        return x

if MAIN:
    w2d3_test.test_gpt_block(GPT2Block)