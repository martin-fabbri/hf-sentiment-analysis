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

# Function to calcuate the accuracy of the model
def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct

def train_step(epoch, training_loader, model, loss_function, device, optimizer):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()

    for _, data in enumerate(training_loader, 0):
        ids = data["input_ids"].to(device, dtype=torch.long)
        mask = data["attention_mask"].to(device, dtype=torch.long)
        targets = data["targets"].to(device, dtype=torch.long)
        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _%5000==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples 
            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")
    return

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

    train_params = {
        "batch_size": train_batch_size,
        "num_workers": 0,
        "bert_model_name": bert_model_name,
        "max_len": max_len,
    }

    test_params = {
        "batch_size": val_batch_size,
        "num_workers": 0,
        "bert_model_name": bert_model_name,
        "max_len": max_len,
    }

    train_dataset = create_data_loader(df_train, **train_params)
    test_dataset = create_data_loader(df_val, **test_params)

    device = torch.device("cuda:0")
    model = BERTBaseUncased(bert_model_name, dropout, linear_units)
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    LEARNING_RATE = 1e-05
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    for epoch in range(epochs):
        train_step(epoch, train_dataset, model, loss_function, device, optimizer)

if __name__ == "__main__":
    typer.run(train)
