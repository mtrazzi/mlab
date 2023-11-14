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
