import torch as torch
import torch.nn as nn

class SentimentModel(nn.Module):
    def __init__(self, bert_model, hidden_dim, n_classes, num_layers, dropout):
        super().__init__()
        self.bert = bert_model
        
        # architecture
        layers = []
        for _ in range(num_layers):
            if len(layers) == 0:
                layers.append(nn.Linear(bert_model.config.hidden_size, hidden_dim, bias=False))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.Dropout(p=dropout))
                layers.append(nn.LeakyReLU())
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.Dropout(p=dropout))
                layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_dim, n_classes))
        
        # unpack
        self.model = nn.Sequential(*layers)
        
        # initalize weights on new layers
        for m in self.model.children():
            if isinstance(m, (nn.BatchNorm1d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)  
        
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids,
                                  attention_mask=attention_mask)
        bert_pooled_output = bert_output[1]
        return self.model(bert_pooled_output)