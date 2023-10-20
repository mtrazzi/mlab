# %%
import json
import os
import sys
from collections import OrderedDict
from io import BytesIO
from typing import Callable, Optional, Union
from pathlib import Path
import requests
import torch as t
import torchvision
from einops import rearrange
from IPython.display import display
from matplotlib import pyplot as plt
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.nn.functional import conv1d as torch_conv1d
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm.auto import tqdm
import utils
import w1d2_test

MAIN = __name__ == "__main__"
IS_CI = os.getenv("IS_CI")
images: list[Image.Image] = []

# %%
IMAGE_FOLDER = Path("./w1d2_images")
IMAGE_FOLDER.mkdir(exist_ok=True)
IMAGE_FILENAMES = [
    "chimpanzee.jpg",
    "golden_retriever.jpg",
    "platypus.jpg",
    "frogs.jpg",
    "fireworks.jpg",
    "astronaut.jpg",
    "iguana.jpg",
    "volcano.jpg",
    "goofy.jpg",
    "dragonfly.jpg",
]


def download_image(url: str, filename: Optional[str]) -> None:
    """Download the image at url to w1d2_images/{filename}, if not already cached."""
    if filename is None:
        filename = url.rsplit("/", 1)[1].replace("%20", "")
    path = IMAGE_FOLDER / filename
    if not path.exists():
        response = requests.get(url)
        data = response.content
        with path.open("wb") as f:
            f.write(data)


if MAIN:
    images = [Image.open(IMAGE_FOLDER / filename) for filename in tqdm(IMAGE_FILENAMES)]
    if not IS_CI:
        display(images[0])

# %%
preprocess: Callable[[Image.Image], t.Tensor]
preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=(224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# %%
with open("w1d2_imagenet_labels.json") as f:
    imagenet_labels = list(json.load(f).values())
# %%


def predict(model, images: list[Image.Image], print_topk_preds=3) -> list[int]:
    """
    Pass the images through the model and print out the top predictions.

    For each image, `display()` the image and the most likely categories according to the model.

    Return: for each image, the index of the top prediction.
    """
    model.eval()
    # convert list of images into a batch of preprocessed tensors using preprocessed
    x = t.stack([preprocess(image) for image in images])
    with t.inference_mode():
        pretrained_categories = model(x)
    return pretrained_categories.argmax(dim=1)


# %%
images = [Image.open(IMAGE_FOLDER / filename) for filename in tqdm(IMAGE_FILENAMES)]

# %%
if True:  # MAIN and (not IS_CI):
    model = models.resnet34(weights="DEFAULT")
    pretrained_categories = predict(model, images)
    print(pretrained_categories)
# %%
