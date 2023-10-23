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
from ipdb import set_trace as p
from torchsummary import summary

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
    # plt.plot()
    # plt.show()

def make_optimizer(model: BertLanguageModel, config_dict: dict) -> t.optim.AdamW:
    """
    Loop over model parameters and form two parameter groups:

    - The first group includes the weights of each Linear layer and uses the weight decay in config_dict
    - The second has all other parameters and uses weight decay of 0
    """
    linear_weights_ids = set()
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_weights_ids.add(id(module.weight))

    return t.optim.AdamW([
        {'params': [p for p in model.parameters() if id(p) in linear_weights_ids], 'weight_decay': config_dict['weight_decay']},
        {'params': [p for p in model.parameters() if id(p) not in linear_weights_ids]},
    ], lr=config_dict['lr'], eps=config_dict['eps'])
    


if MAIN:
    test_config = BertConfig(
        max_position_embeddings=4, hidden_size=1, intermediate_size=4, num_layers=3, num_heads=1, head_size=1
    )
    optimizer_test_model = BertLanguageModel(test_config)
    opt = make_optimizer(optimizer_test_model, dict(weight_decay=0.1, lr=0.0001, eps=1e-06))
    expected_num_with_weight_decay = test_config.num_layers * 6 + 1
    wd_group = opt.param_groups[0]
    actual = len(wd_group["params"])
    assert (
        actual == expected_num_with_weight_decay
    ), f"Expected 6 linear weights per layer (4 attn, 2 MLP) plus the final lm_linear weight to have weight decay, got {actual}"
    all_params = set()
    for group in opt.param_groups:
        all_params.update(group["params"])
    assert all_params == set(optimizer_test_model.parameters()), "Not all parameters were passed to optimizer!"

def bert_mlm_pretrain(model: BertLanguageModel, config_dict: dict, train_loader: DataLoader) -> None:
    """Train using masked language modelling."""
    opt = make_optimizer(model, config_dict=config_dict)
    def set_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    global_step = 0
    max_step = config_dict['epochs'] * len(train_loader)
    for epoch in range(config_dict['epochs']):  # 10 epochs
        for batch_idx, batch_list in enumerate(train_loader):
            opt.zero_grad()
            new_lr = lr_for_step(global_step, max_step, max_lr=config_dict['lr'], warmup_step_frac=config_dict['warmup_step_frac'])
            set_lr(opt, new_lr)
            model_inputs, was_selected = random_mask(batch_list[0], config_dict['mask_token_id'], model.config.vocab_size)
            output = model(model_inputs)
            loss = cross_entropy_selected(output, batch_list[0], was_selected=was_selected)
            loss.backward()
            opt.step()
            print(f"step = {global_step} loss = {loss}")
            global_step += 1

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


if MAIN:
    model = BertLanguageModel(bert_config_tiny)
    num_params = sum((p.nelement() for p in model.parameters()))
    print("Number of model parameters: ", num_params)
    bert_mlm_pretrain(model, config_dict, train_loader)