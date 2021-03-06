import numpy as np
import pandas as pd
import typer
from sklearn.model_selection import train_test_split


def to_sentiment(rating):
    rating = int(rating)
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2


def split(
    random_seed: int,
    reviews_dataset_path: str,
    train_split_path: str,
    val_split_path: str,
    test_split_path: str,
):
    df = pd.read_csv(reviews_dataset_path)
    df["sentiment"] = df.score.apply(to_sentiment)
    df = df.rename(columns={"content": "reviews"})
    df = df[["reviews", "sentiment"]]
    df = df.dropna(axis=0, how="any")
    df = df.reindex(columns=["reviews", "sentiment"])
    df_train, df_test = train_test_split(
        df,
        test_size=0.1,
        random_state=random_seed,
        stratify=df.sentiment.values,
    )
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=random_seed)

    print("train.shape", df_train.shape)
    print("test.shape ", df_test.shape)
    print("val.shape  ", df_val.shape)

    df_train.to_csv(train_split_path, index=False)
    df_train.to_csv(val_split_path, index=False)
    df_train.to_csv(test_split_path, index=False)


if __name__ == "__main__":
    typer.run(split)
