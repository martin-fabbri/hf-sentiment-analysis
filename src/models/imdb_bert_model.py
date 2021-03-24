import torch.nn as nn
import transformers
from transformers import AutoModelForPreTraining


class BERTBaseUncased(nn.Module):
    def __init__(self, bert_model_name, dropout, linear_units):
        super(BERTBaseUncased, self).__init__()
        self.bert = AutoModelForPreTraining.from_pretrained(bert_model_name)
        self.bert_drop = nn.Dropout(dropout)
        self.out = nn.Linear(linear_units, 1)

    def forward(self, ids, mask, token_type_ids):
        outputs = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        print("************")
        print("************")
        print(outputs)
        print("************")
        print(outputs.keys())
        print("************")
        print(outputs.prediction_logits)
        print("************")
        print("************")
        bert_output = self.bert_drop(outputs.prediction_logits)
        print("************")
        print("************")
        print("bert_output.shape", bert_output.shape)
        output = self.out(bert_output)
        print("************")
        print("************")
        return output
