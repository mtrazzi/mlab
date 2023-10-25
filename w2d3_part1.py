#!/usr/bin/env python

import math
import os
from dataclasses import dataclass
from typing import Any, Optional
import torch as t
import transformers
from einops import rearrange, repeat
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

class GPT2(nn.Module):
    token_embedding: nn.Embedding
    pos_embedding: nn.Embedding
    ln: nn.LayerNorm
    blocks: utils.StaticModuleList[GPT2Block]

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.blocks = utils.StaticModuleList([GPT2Block(config.hidden_size, config.num_heads, config.dropout, config.layer_norm_epsilon, config.activation_function) for _ in range(config.num_layers)])
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: t.Tensor, cache: Optional[Any] = None) -> t.Tensor:
        """
        x: shape (batch, seq), dtype t.int64 - the token ids

        Return: shape (batch, seq, vocab_size), dtype t.float32- the output logits
        """
        pos_idxs = t.arange(x.shape[1]).to(x.device)
        pos_idxs = repeat(pos_idxs, "n -> b n", b=x.shape[0])

        pos_emb = self.pos_embedding(pos_idxs)
        tok_emb = self.token_embedding(x)

        x = pos_emb + tok_emb
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)

        x = t.einsum("bnh, vh -> bnv", x, self.token_embedding.weight)
        return x

if MAIN:
    w2d3_test.test_gpt(GPT2)

def _copy_weight_bias(mine, theirs, transpose=False):
    mine.weight.copy_(theirs.weight.T if transpose else theirs.weight)
    if mine.bias is not None:
        mine.bias.copy_(theirs.bias)


def load_pretrained_weights():
    pretrained_gpt = utils.load_pretrained_gpt()
    my_gpt = GPT2(config)
    for p in my_gpt.parameters():
        p.requires_grad = False
    my_gpt.token_embedding.weight.copy_(pretrained_gpt.transformer.wte.weight)
    my_gpt.pos_embedding.weight.copy_(pretrained_gpt.transformer.wpe.weight)
    _copy_weight_bias(my_gpt.ln, pretrained_gpt.transformer.ln_f)
    from transformers.models.gpt2.modeling_gpt2 import GPT2Block as HFGPT2Block

    my_block: GPT2Block
    hf_block: HFGPT2Block
    for (my_block, hf_block) in zip(my_gpt.blocks, pretrained_gpt.transformer.h):
        _copy_weight_bias(my_block.ln1, hf_block.ln_1)
        _copy_weight_bias(my_block.attn.qkv_proj, hf_block.attn.c_attn, transpose=True)
        _copy_weight_bias(my_block.attn.output_proj, hf_block.attn.c_proj, transpose=True)
        _copy_weight_bias(my_block.ln2, hf_block.ln_2)
        _copy_weight_bias(my_block.linear1, hf_block.mlp.c_fc, transpose=True)
        _copy_weight_bias(my_block.linear2, hf_block.mlp.c_proj, transpose=True)
    for p in my_gpt.parameters():
        p.requires_grad_(True)
    return my_gpt

if MAIN:
    my_gpt = load_pretrained_weights()
    my_gpt.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    w2d3_test.test_load_pretrained_weights(my_gpt, tokenizer)