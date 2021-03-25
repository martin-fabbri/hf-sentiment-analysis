import torch.nn as nn
import torch
import transformers
from transformers import AutoModelForPreTraining


class BERTBaseUncased(nn.Module):
    def __init__(self, bert_model_name, dropout, linear_units):
        super(BERTBaseUncased, self).__init__()
        self.bert = AutoModelForPreTraining.from_pretrained(bert_model_name)
        self.pre_classifier = nn.Linear(linear_units, linear_units)
        self.bert_drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(linear_units, 1)

    def forward(self, ids, mask):
        output_1 = self.bert(input_ids=ids, attention_mask=mask)

        print("{}{}{}{}{}{}{")
        print(output_1)

        output_1 = output_1[0]        
        pooler = output_1[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.bert_drop(pooler)
        output = self.classifier(pooler)
        return output
