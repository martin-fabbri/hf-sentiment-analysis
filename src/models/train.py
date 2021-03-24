from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import tqdm
import transformers
import typer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from torch import nn, optim
from imdb_bert_model import BERTBaseUncased

from dataset import create_data_loader
from transformers import AdamW, get_linear_schedule_with_warmup
from train_fn import train_fn


def train(
    bert_model_name: str,
    train_split_path: str,
    val_split_path: str,
    max_len: int,
    train_batch_size: int,
    val_batch_size: int,
    dropout: float,
    linear_units: int,
    epochs: int,
    classification_model_path: str,
):
    df_train = pd.read_csv(train_split_path).fillna("none")
    df_val = pd.read_csv(val_split_path).fillna("none")

    train_dataset = create_data_loader(
        df_train, bert_model_name, max_len, train_batch_size, num_workers=4
    )

    val_dataset = create_data_loader(
        df_val, bert_model_name, max_len, val_batch_size, num_workers=1
    )

    device = torch.device("cuda")
    model = BERTBaseUncased(bert_model_name, dropout, linear_units)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / train_batch_size * epochs)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps,
    )

    best_accuracy = 0
    for epochs in range(epochs):
        train_fn(train_dataset, model, optimizer, device)
        outputs, targets = engine.eval_fn(val_dataset, model, optimizer, device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy score = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(
                model.state_dict(),
                classification_model_path,
            )
            best_accuracy = accuracy


if __name__ == "__main__":
    typer.run(train)
