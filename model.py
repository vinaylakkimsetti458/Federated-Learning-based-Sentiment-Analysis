import torch
from torch import nn
from transformers import DistilBertForSequenceClassification

class DistilBERTSentiment(nn.Module):
    def __init__(self):
        super(DistilBERTSentiment,self).__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels = 2)

    def forward(self,input_ids,attention_mask):
        return self.model(input_ids = input_ids,attention_mask = attention_mask).logits
