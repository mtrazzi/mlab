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
            attn_sc = attn_sc + additive_attention_mask

        attn_probs = t.softmax(attn_sc, dim=-1)

        V = rearrange(V, "b s (nh hs) -> b nh s hs", nh=self.config.num_heads, hs=self.config.head_size)
        values = t.einsum("bhki,bhqk->bqhi", V, attn_probs)
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


class LayerNorm(nn.Module):
    weight: nn.Parameter
    bias: nn.Parameter

    def __init__(
        self, normalized_shape: Union[int, tuple, t.Size], eps=1e-05, elementwise_affine=True, device=None, dtype=None
    ):
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.device = device
        self.dtype = dtype
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the weight and bias, if applicable."""
        self.weight = nn.Parameter(t.ones(self.normalized_shape, dtype=self.dtype, device=self.device))
        self.bias = nn.Parameter(t.zeros(self.normalized_shape, dtype=self.dtype, device=self.device))

    def forward(self, x: t.Tensor) -> t.Tensor:
        """x and the output should both have shape (batch, *)."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        if self.elementwise_affine:
            y = self.weight * ((x - mean) / t.sqrt(var + self.eps)) + self.bias
        else:
            y = (x - mean) / t.sqrt(var + self.eps)
        return y


if MAIN:
    w2d1_test.test_layernorm_mean_1d(LayerNorm)
    w2d1_test.test_layernorm_mean_2d(LayerNorm)
    w2d1_test.test_layernorm_std(LayerNorm)
    w2d1_test.test_layernorm_exact(LayerNorm)
    w2d1_test.test_layernorm_backward(LayerNorm)


class Embedding(nn.Module):
    num_embeddings: int
    embedding_dim: int
    weight: nn.Parameter

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(t.randn(num_embeddings, embedding_dim) * 0.02)

    def forward(self, x: t.LongTensor) -> t.Tensor:
        """For each integer in the input, return that row of the embedding.

        Don't convert x to one-hot vectors - this works but is too slow.
        """
        return self.weight[x]

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}"


if MAIN:
    assert repr(Embedding(10, 20)) == repr(t.nn.Embedding(10, 20))
    w2d1_test.test_embedding(Embedding)
    w2d1_test.test_embedding_std(Embedding)


class BertMLP(nn.Module):
    first_linear: nn.Linear
    second_linear: nn.Linear
    layer_norm: nn.LayerNorm

    def __init__(self, config: BertConfig):
        super().__init__()
        self.first_linear = nn.Linear(config.hidden_size, config.intermediate_size)
        self.second_linear = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """x has shape (batch, seq, hidden_size)."""
        y = self.first_linear(x)
        y = F.gelu(y)
        y = self.second_linear(y)
        y = self.dropout(y)
        y = self.layer_norm(y + x)
        return y


if MAIN:
    w2d1_test.test_bert_mlp_zero_dropout(BertMLP)
    w2d1_test.test_bert_mlp_one_dropout(BertMLP)


class BertAttention(nn.Module):
    self_attn: BertSelfAttention
    layer_norm: nn.LayerNorm

    def __init__(self, config: BertConfig):
        super().__init__()
        self.self_attn = BertSelfAttention(config)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None) -> t.Tensor:
        y = self.self_attn(x, additive_attention_mask)
        y = self.dropout(y)
        y = self.layer_norm(x + y)
        return y


if MAIN:
    w2d1_test.test_bert_attention_dropout(BertAttention)


class BertBlock(nn.Module):
    attention: BertAttention
    mlp: BertMLP

    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.mlp = BertMLP(config)

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None) -> t.Tensor:
        y = self.attention(x, additive_attention_mask)
        y = self.mlp(y)
        return y


if MAIN:
    w2d1_test.test_bert_block(BertBlock)


def make_additive_attention_mask(one_zero_attention_mask: t.Tensor, big_negative_number: float = -10000) -> t.Tensor:
    """
    one_zero_attention_mask: shape (batch, seq). Contains 1 if this is a valid token and 0 if it is a padding token.
    big_negative_number: Any negative number large enough in magnitude that exp(big_negative_number) is 0.0 for the floating point precision used.

    Out: shape (batch, heads, seq, seq). Contains 0 if attention is allowed, and big_negative_number if it is not allowed.
    """
    # TODO: make sure shape is correct
    return (1 - one_zero_attention_mask) * big_negative_number


class BertCommon(nn.Module):
    token_embedding: Embedding
    pos_embedding: Embedding
    token_type_embedding: Embedding
    layer_norm: nn.LayerNorm
    blocks: utils.StaticModuleList[BertBlock]

    def __init__(self, config: BertConfig):
        super().__init__()
        self.token_embedding = Embedding(config.vocab_size, config.hidden_size)
        self.pos_embedding = Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embedding = Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = utils.StaticModuleList([BertBlock(config) for _ in range(config.num_layers)])

    def forward(
        self,
        input_ids: t.Tensor,
        token_type_ids: Optional[t.Tensor] = None,
        one_zero_attention_mask: Optional[t.Tensor] = None,
    ) -> t.Tensor:
        """
        input_ids: (batch, seq) - the token ids
        token_type_ids: (batch, seq) - only used for next sentence prediction.
        one_zero_attention_mask: (batch, seq) - only used in training. See make_additive_attention_mask.
        """
        pos_idxs = t.arange(0, input_ids.shape[-1])
        pos_emb = self.pos_embedding(pos_idxs)
        tok_emb = self.token_embedding(input_ids)
        x = pos_emb + tok_emb
        mask = make_additive_attention_mask(one_zero_attention_mask) if one_zero_attention_mask else None
        if token_type_ids:
            x += self.token_type_embedding(token_type_ids)
        x = self.layer_norm(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, mask)
        return x


class BertLanguageModel(nn.Module):
    common: BertCommon
    lm_linear: nn.Linear
    lm_layer_norm: nn.LayerNorm
    unembed_bias: nn.Parameter

    def __init__(self, config: BertConfig):
        super().__init__()
        self.common = BertCommon(config)
        self.lm_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.unembed_bias = nn.Parameter(t.zeros(config.vocab_size))

    def forward(
        self,
        input_ids: t.Tensor,
        token_type_ids: Optional[t.Tensor] = None,
        one_zero_attention_mask: Optional[t.Tensor] = None,
    ) -> t.Tensor:
        """Compute logits for each token in the vocabulary.

        Return: shape (batch, seq, vocab_size)
        """
        x = self.common(input_ids, token_type_ids, one_zero_attention_mask)
        x = self.lm_linear(x)
        x = F.gelu(x)
        x = self.lm_layer_norm(x)
        logits = F.linear(x, self.common.token_embedding.weight, self.unembed_bias)
        return logits


if MAIN:
    w2d1_test.test_bert(BertLanguageModel)
