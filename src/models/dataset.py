import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from pandas import DataFrame


class BertDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def create_dataset(df: DataFrame, tokenizer: PreTrainedTokenizerBase):
    texts = df.reviews.values.tolist()
    labels = df.sentiment.values.tolist()
    encondings = tokenizer(texts, truncation=True, padding=True)
    ds = BertDataset(encondings, labels)
    return ds