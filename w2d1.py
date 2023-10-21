#!/usr/bin/env python3
import os
from dataclasses import dataclass
from typing import List, Optional, Union
import torch as t
import transformers
from einops import rearrange, repeat
from fancy_einsum import einsum
from torch import nn
from torch.nn import functional as F
import utils
import w2d1_test

MAIN = __name__ == "__main__"
IS_CI = os.getenv("IS_CI")


@dataclass(frozen=True)
class BertConfig:
    """Constants used throughout the Bert model. Most are self-explanatory.

    intermediate_size is the number of hidden neurons in the MLP (see schematic)
    type_vocab_size is only used for pretraining on "next sentence prediction", which we aren't doing.

    Note that the head size happens to be hidden_size // num_heads, but this isn't necessarily true and your code shouldn't assume it.
    """

    vocab_size: int = 28996
    intermediate_size: int = 3072
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    head_size: int = 64
    max_position_embeddings: int = 512
    dropout: float = 0.1
    type_vocab_size: int = 2
    layer_norm_epsilon: float = 1e-12


if MAIN:
    config = BertConfig()


class BertSelfAttention(nn.Module):
    project_query: nn.Linear
    project_key: nn.Linear
    project_value: nn.Linear
    project_output: nn.Linear

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.project_query = nn.Linear(config.hidden_size, config.head_size * config.num_heads)
        self.project_key = nn.Linear(config.hidden_size, config.head_size * config.num_heads)
        self.project_value = nn.Linear(config.hidden_size, config.head_size * config.num_heads)
        self.project_output = nn.Linear(config.head_size * config.num_heads, config.hidden_size)

    def attention_pattern_pre_softmax(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)
        Return the attention pattern after scaling but before softmax.

        pattern[batch, head, q, k] should be the match between a query at sequence position q and a key at sequence position k.
        """

        batch_size = x.shape[0]
        seq_len = x.shape[1]
        Q = self.project_query(x)
        K = self.project_key(x)

        num_heads, head_size = self.config.num_heads, self.config.head_size
        Q = rearrange(Q, "b s (nh hs) -> b s nh hs", nh=num_heads, hs=head_size)
        K = rearrange(K, "b s (nh hs) -> b s nh hs", nh=num_heads, hs=head_size)
        QK = t.einsum("bqhi,bkhi->bhqk", Q, K)
        return QK / self.config.head_size**0.5

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None) -> t.Tensor:
        """
        additive_attention_mask: shape (batch, head=1, seq_q=1, seq_k) - used in training to prevent copying data from padding tokens. Contains 0 for a real input token and a large negative number for a padding token. If provided, add this to the attention pattern (pre softmax).

        Return: (batch, seq, hidden_size)
        """
        V = self.project_value(x)
        attn_sc = self.attention_pattern_pre_softmax(x)
        if additive_attention_mask:
            attn_sc += additive_attention_mask

        attn_probs = t.softmax(attn_sc, dim=-1)

        V = rearrange(V, "b s (nh hs) -> b s nh hs", nh=self.config.num_heads, hs=self.config.head_size)
        # multiply the values and the attention probabilities like in bert
        values = t.einsum("bhqk,bkhi->bqhi", attn_probs, V)
        values = rearrange(values, "b s nh hs -> b s (nh hs)")
        output = self.project_output(values)
        return output


if MAIN:
    batch_size = 2
    seq_len = 5
    hidden_size = 6
    num_heads = 2
    import w2d1_solution

    config = w2d1_solution.BertConfig(hidden_size=hidden_size, num_heads=num_heads, head_size=3)
    ref = w2d1_solution.BertSelfAttention(config)

    yours = BertSelfAttention(config)
    yours.load_state_dict(ref.state_dict())

    w2d1_test.test_attention_pattern_pre_softmax(BertSelfAttention)
    w2d1_test.test_attention(BertSelfAttention)

# %%
