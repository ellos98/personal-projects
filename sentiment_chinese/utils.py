import torch
import torch.nn as nn
import os

            
            
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("Saving checkpoint...")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer=None, scheduler=None):
    print("Loading checkpoint...")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler"])

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.BatchNorm1d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            
