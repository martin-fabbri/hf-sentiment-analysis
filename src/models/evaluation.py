from collections import defaultdict
import tqdm import tqdm
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

from data_loader import create_data_loader

def eval_fn(data_loader, model, device):
    model.eval()
    final_targets = []
    final_outputs = []

    with torch.no_grad():
        for batch_index, dataset in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = dataset["input_ids"]
            token_type_ids = dataset["token_type_ids"]
            mask = dataset["attention_mask"]
            targets = dataset["targets"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(
                ids = ids,
                mask = mask,
                token_type_ids = token_type_ids,
            )
            final_targets.extend(targets.cpu().detach().numpy().tolist())
            final_outputs.extend(torch.sigmoid(outputs.cpu()).detach().numpy().tolist())
    return final_outputs, final_targets
        
