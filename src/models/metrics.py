from transformers import EvalPrediction
from sklearn.metrics import accuracy_score


def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)