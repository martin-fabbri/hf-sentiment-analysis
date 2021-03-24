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
        _, out2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bert_output = self.bert_drop(out2)
        output = self.out(bert_output)
        return output
