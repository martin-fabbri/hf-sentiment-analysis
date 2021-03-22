import numpy as np
import pandas as pd
import torch
import transformers
import typer
from transformers import (
    AdamW,
    AutoTokenizer,
    BertModel,
    get_linear_schedule_with_warmup,
)


def to_sentiment(rating):
    rating = int(rating)
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2


def prepare(
    random_seed: int,
    pre_trained_model_name: str,
    apps_dataset_path: str,
    reviews_datase_path: str,
):
    df = pd.read_csv(reviews_datase_path)
    df["sentiment"] = df.score.apply(to_sentiment)
    class_name = ["negative", "neutral", "positive"]
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name)

    sample_txt = "When was I last outside? I am stuck at home for 2 weeks."
    tokens = tokenizer.tokenize(sample_txt)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    print(f" Sentence: {sample_txt}")
    print(f"   Tokens: {tokens}")
    print(f" Sentence: {sample_txt}")
    print(f"Token IDs: {token_ids}")
    print(
        f"Special Tokens [SEP] end of a sentence: {tokenizer.sep_token}, {tokenizer.sep_token_id}"
    )
    print(
        f"Special Tokens [CLS] classification task: {tokenizer.cls_token}, {tokenizer.cls_token_id}"
    )
    print(
        f"Special Tokens [PAD] padding: {tokenizer.pad_token}, {tokenizer.pad_token_id}"
    )
    print(
        f"Special Tokens [UNK] unknown: {tokenizer.unk_token}, {tokenizer.unk_token_id}"
    )


if __name__ == "__main__":
    typer.run(prepare)
