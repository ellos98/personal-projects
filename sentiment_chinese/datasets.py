import torch
from torch.utils.data import Dataset
import pandas as pd

class BERTDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_len):
        super().__init__()
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        selected_text = self.X.iloc[index]
        selected_label = torch.tensor(self.y.iloc[index])
        
        inputs = self.tokenizer.encode_plus(
            selected_text,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].flatten()
        attention_mask = inputs['attention_mask'].flatten()
        
        return {'selected_text': selected_text,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'selected_label': selected_label}
