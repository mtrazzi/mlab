#!/usr/bin/env python

import os
import sys
from dataclasses import dataclass
import torch as t
import transformers
from einops import rearrange, repeat
from tqdm.auto import tqdm
import utils
import w2d3_test
from w2d3_part1_loading_solution import GPT2, GPT2Block, load_pretrained_weights
import torch.nn.functional as F


os.environ["TOKENIZERS_PARALLELISM"] = "false"
MAIN = __name__ == "__main__"
IS_CI = os.getenv("IS_CI")
if IS_CI:
    sys.exit(0)
if MAIN:
    my_gpt = load_pretrained_weights().eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

def sample_next_token(
    model: GPT2, input_ids: t.Tensor, temperature=1.0, freq_penalty=0.0, top_k=0, top_p=0.0, cache=None
) -> int:
    """Return the next token, sampled from the model's probability distribution with modifiers.

    input_ids: shape (seq,)
    """
    assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
    assert temperature >= 0
    assert 0 <= top_p <= 1.0, "Top-p must be a probability"
    assert 0 <= top_k, "Top-k must be non-negative"
    assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"
    model.eval()
    with t.inference_mode():
        all_logits = model(input_ids.unsqueeze(0), cache=cache)
    (B, S, V) = all_logits.shape
    assert B == 1
    assert S == len(input_ids)
    logits = all_logits[0, -1]
    if temperature == 0:
        return greedy_search(logits)
    logits = apply_temperature(logits, temperature)
    logits = apply_freq_penalty(input_ids, logits, freq_penalty)
    if top_k > 0:
        return sample_top_k(logits, top_k)
    if top_p > 0:
        return sample_top_p(logits, top_p)
    return sample_basic(logits)


def sample_tokens(
    model: GPT2,
    tokenizer,
    initial_text: str,
    max_tokens_generated=30,
    temperature=1.0,
    freq_penalty=0.0,
    stop_at_eos=True,
    top_k=0,
    top_p=0.0,
    cache=None,
) -> str:
    """Sample tokens using sample_next_token until the model outputs `tokenizer.eos_token_id` or the specified token limit is reached.

    Return: the prompt and continuation concatenated
    """
    model.eval()
    input_ids: list = tokenizer(initial_text).input_ids
    generated = []
    device = next(model.parameters()).device
    for _ in tqdm(range(max_tokens_generated)):
        new_token = sample_next_token(
            model,
            t.tensor(input_ids + generated, dtype=t.int64, device=device),
            temperature=temperature,
            freq_penalty=freq_penalty,
            top_k=top_k,
            top_p=top_p,
            cache=cache,
        )
        generated.append(new_token)
        if stop_at_eos and new_token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(input_ids + generated)

def greedy_search(logits: t.Tensor) -> int:
    """
    logits: shape (vocab_size, )

    Return: the most likely token
    """
    tok = logits.argmax().item()
    assert isinstance(tok, int)
    return tok

# if MAIN:
#     logits = t.ones(100)
#     logits[5] = 10
#     logits[8] = 10
#     assert greedy_search(logits) == 5
#     w2d3_test.test_sample_zero_temperature(my_gpt, tokenizer, sample_tokens)

def apply_temperature(logits: t.Tensor, temperature: float) -> t.Tensor:
    """
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    """
    assert temperature > 0
    return logits / temperature


# if MAIN:
#     logits = t.tensor([1, 2]).log()
#     cold_logits = apply_temperature(logits, 0.001)
#     print('A low temperature "sharpens" or "peaks" the distribution: ', cold_logits)
#     utils.allclose(cold_logits, 1000.0 * logits)
#     hot_logits = apply_temperature(logits, 1000.0)
#     print("A high temperature flattens the distribution: ", hot_logits)
#     utils.allclose(hot_logits, 0.001 * logits)


def apply_freq_penalty(input_ids: t.Tensor, logits: t.Tensor, freq_penalty: float) -> t.Tensor:
    """
    input_ids: shape (seq, )
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    """
    return logits - freq_penalty * t.bincount(input_ids, minlength=logits.shape[0])


#if MAIN:
#    bieber_prompt = "And I was like Baby, baby, baby, oh Like, Baby, baby, baby, no Like, Baby, baby, baby, oh I thought you'd always be mine, mine"
#    input_ids = tokenizer(bieber_prompt, return_tensors="pt")["input_ids"][0]
#    logits = t.ones(tokenizer.vocab_size)
#    penalized_logits = apply_freq_penalty(input_ids, logits, 2.0)
#    assert penalized_logits[5156].item() == -11, "Expected 6 occurrences of ' baby' with leading space"
#    assert penalized_logits[14801].item() == -5, "Expected 3 occurrences of ' Baby' with leading space"


def sample_basic(logits: t.Tensor) -> int:
    """
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    """
    dist = t.distributions.categorical.Categorical(logits=logits)
    tok = dist.sample().item()
    assert isinstance(tok, int)
    return tok

# if MAIN:
#     N = 20000
#     probs = t.linspace(0, 0.4, 5)
#     unnormalized_logits = probs.log() + 1.2345
#     samples = t.tensor([sample_basic(unnormalized_logits) for _ in range(N)])
#     counts = t.bincount(samples, minlength=len(probs)) / N
#     print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
#     utils.allclose_atol(counts, probs, atol=0.01)

# if MAIN:
#     N_RUNS = 5
#     your_prompt = "Barack Obama is the president of the "
#     cases = [
#         ("High freq penalty", dict(freq_penalty=100.0)),
#         ("Negative freq penalty", dict(freq_penalty=-1.0)),
#         ("Too hot!", dict(temperature=2.0)),
#         ("Pleasantly cool", dict(temperature=0.7)),
#         ("Pleasantly warm", dict(temperature=0.9)),
#     ]
#     for (name, kwargs) in cases:
#         for i in range(N_RUNS):
#             output = sample_tokens(my_gpt, tokenizer, your_prompt, max_tokens_generated=24, **kwargs)
#             print(f"Sample {i} with: {name} ({kwargs}):")
#             print(f"Your model said: {repr(output)}")


def sample_top_k(logits: t.Tensor, top_k: int) -> int:
    """
    logits: shape (vocab_size, ) - unnormalized log-probabilities
    top_k: only consider this many of the most likely tokens for sampling

    Return: a sampled token
    """
    values, indices = t.topk(logits, top_k)
    out = t.zeros_like(logits, device=logits.device) - t.inf
    out[indices] = values
    return sample_basic(out)


# if MAIN:
#     k, N = 3, 10000
#     probs = t.linspace(0, 0.4, 5)
#     unnormalized_logits = probs.log() + 1.2345
#     samples = t.tensor([sample_top_k(unnormalized_logits, k) for _ in range(N)])
#     counts = t.bincount(samples, minlength=len(probs)) / N
#     expected = probs.clone()
#     expected[:-k] = 0
#     expected /= expected.sum()
#     print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
#     utils.allclose_atol(counts, expected, atol=0.01)

# if MAIN:
#     your_prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
#     output = sample_tokens(my_gpt, tokenizer, your_prompt, temperature=0.7, top_k=40, max_tokens_generated=64)
#     print(f"Your model said: {repr(output)}")

def sample_top_p(logits: t.Tensor, top_p: float, min_tokens_to_keep: int = 1) -> int:
    """
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    """
    sorted_logits, indices = t.sort(logits)
    # Convert logits to probabilities
    probs = F.softmax(sorted_logits, dim=-1)

    # Compute cumulative probabilities
    cumulative_probs = t.cumsum(probs, dim=-1)

    # Find the cutoff point
    cutoff_index = t.where(cumulative_probs >= top_p)[0][0].item()

    kept_token_indices = indices[min(cutoff_index, indices.shape[0] - min_tokens_to_keep):]
    kept_logits = sorted_logits[kept_token_indices]
    kept_idx = sample_basic(kept_logits)
    out = kept_token_indices[kept_idx].item()
    assert isinstance(out, int)
    return out


if MAIN:
    N = 2000
    unnormalized_logits = t.tensor([0.2, 0.3, 0.5]).log() + 2.3456
    samples = t.tensor([sample_top_p(unnormalized_logits, 0.5) for _ in range(N)])
    counts = t.bincount(samples, minlength=len(unnormalized_logits)) / N
    print("top_p of 0.5 or lower should only return token 2: ", counts)
    assert counts[0] == 0 and counts[1] == 0
if MAIN:
    N = 2000
    unnormalized_logits = t.tensor([0.2, 0.3, 0.5]).log() + 2.3456
    samples = t.tensor([sample_top_p(unnormalized_logits, 0.50001) for _ in range(N)])
    counts = t.bincount(samples, minlength=len(unnormalized_logits)) / N
    print("top_p in (0.5, 0.8] should return tokens 1 and 2: ", counts)
    assert counts[0] == 0
if MAIN:
    N = 2000
    top_p = 0.71
    probs = t.linspace(0, 0.4, 5)
    unnormalized_logits = probs.log() + 1.2345
    samples = t.tensor([sample_top_p(unnormalized_logits, top_p) for _ in range(N)])
    counts = t.bincount(samples, minlength=len(probs)) / N
    expected = probs.clone()
    expected[0:2] = 0
    expected /= expected.sum()
    print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
    utils.allclose_atol(counts, expected, atol=0.01)

#TODO: 1) using caching to make text generation more efficient 2) Beam Search