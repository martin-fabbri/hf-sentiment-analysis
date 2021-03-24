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
from sklearn.model_selection import train_test_split

def train(imdb_dataset_path, ramdom_seed):
    df = pd.read_csv(imdb_dataset_path).fillns("none")
    df.sentiment = df.sentiment.apply(lambda x: 1 if x == "positive" else 0)
    df_train, df_valid = train_test_split(
        df, 
        test_size=0.1, 
        random_state=ramdom_seed,
        stratify = dfx.sentiment.values,
    )
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)


if __name__ == "__main__":
    typer.run(train)
