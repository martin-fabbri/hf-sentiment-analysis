from collections import defaultdict

import import
import numpy as np
import pandas as pd
import torch
import tqdm
import transformers
import typer
from data_loader import create_data_loader
from torch import nn, optim


def train(imdb_dataset_path):
    df = pd.read_csv(imdb_dataset_path).fillns("none")
    df.sentiment = df.sentiment.apply(lambda x: 1 if x == "positive" else 0)
    train_fn()


if __name__ == "__main__":
    typer.run(train)
