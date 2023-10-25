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

os.environ["TOKENIZERS_PARALLELISM"] = "false"
MAIN = __name__ == "__main__"
IS_CI = os.getenv("IS_CI")
if IS_CI:
    sys.exit(0)
if MAIN:
    my_gpt = load_pretrained_weights().eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")