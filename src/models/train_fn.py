from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import transformers
import typer
from torch import nn, optim
from transformers import (
    AdamW,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from dataset import create_data_loader


def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)


def train_fn(data_loader, model, optimizer, device):
    model.train()

    for batch_index, dataset in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = dataset["input_ids"]
        token_type_ids = dataset["token_type_ids"]
        mask = dataset["attention_mask"]
        targets = dataset["targets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids,
        )

        loss = loss_fn(outputs, targets)
        loss.backward()

        optimizer.step()
        scheduler.step()
