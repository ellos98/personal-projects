import torch
import torch.nn as nn


class Engine:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
    
    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        lr = self.optimizer.param_groups[0]['lr']
        
        for batch_dict in data_loader:
            input_ids = batch_dict['input_ids'].to(self.device)
            attention_mask = batch_dict['attention_mask'].to(self.device)
            label = batch_dict['selected_label'].to(self.device)

            # forward
            with torch.cuda.amp.autocast():
                pred_label = self.model(input_ids,
                                   attention_mask)
                loss = self.criterion(pred_label, label)
                with torch.no_grad():
                    final_loss += loss.item()

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()        
            
        return final_loss / len(data_loader.dataset)
    
    
    def evaluate(self, data_loader):
        self.model.eval()
        final_loss = 0
        correct_prediction = 0
        softmax = nn.Softmax(dim=1)
        
        for batch_dict in data_loader:
            input_ids = batch_dict['input_ids'].to(self.device)
            attention_mask = batch_dict['attention_mask'].to(self.device)
            label = batch_dict['selected_label'].to(self.device)

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    pred_label = self.model(input_ids,
                                       attention_mask)
                    loss = self.criterion(pred_label, label)
                    final_loss += loss.item()
                    pred_label = softmax(pred_label).argmax(dim=1)
                    correct_prediction += torch.sum(pred_label == label).item()
        
        with torch.no_grad():
            eval_loss = final_loss / len(data_loader.dataset)
            eval_accuracy = correct_prediction / len(data_loader.dataset)
            
        self.model.train()
        return {
            'eval_loss':eval_loss,
            'eval_accuracy':eval_accuracy            
        }