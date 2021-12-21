import torch
import torch.nn as nn
import engine
import datasets
import models
import utils
from copy import deepcopy
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

class SummaryWriterFix(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict, step):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()
        
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v, global_step=step)

def cross_validation(fold, df, params, NUM_EPOCHS, bert_model, tokenizer, MAX_LEN, DEVICE):
    #copy bert model to avoid leakage
    bert_model_ = deepcopy(bert_model)
    
    # get X, y from fold
    X_train = df.loc[df.kfold != fold, 'text']
    X_val = df.loc[df.kfold == fold, 'text']
    y_train = df.loc[df.kfold != fold, 'sentiment']
    y_val = df.loc[df.kfold == fold, 'sentiment']
    
    # max fold
    max_fold = df.kfold.unique().shape[0]
    
    # dataset
    train_dataset = datasets.BERTDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_dataset = datasets.BERTDataset(X_val, y_val, tokenizer, MAX_LEN)
    
    # sample weights for imbalalanced class
    label_weights_dict = 1 / (y_train.value_counts() / y_train.shape[0])
    label_weights_dict = label_weights_dict.to_dict()
    label_weights = y_train.map(label_weights_dict).tolist()
    
    # weighted sampler
    gen = torch.Generator()
    gen.manual_seed(17678452212620826457)
    weighted_sampler = WeightedRandomSampler(label_weights,
                                    len(label_weights),
                                    replacement=True,
                                    generator=gen)
    
    # loader
    train_loader = DataLoader(train_dataset,
                              batch_size=params['batch_size'],
                              sampler=weighted_sampler)
    val_loader = DataLoader(val_dataset,
                            batch_size=params['batch_size'])
    
    # model init
    model = models.SentimentModel(bert_model_,
                           bert_model_.config.hidden_size*params['hidden_size_ratio'],
                           y_train.unique().shape[0],
                           num_layers=params['num_layers'],
                           dropout=params['dropout']
                           ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                          factor=0.1,
                                                          patience=500,
                                                          verbose=True)
    
    # engine init
    engine_ = engine.Engine(model, optimizer, criterion, DEVICE)
    
    best_loss = np.inf
    early_stopping_iter = 20
    early_stopping_counter = 0
    
    # tensorboard init
    folder = 'hyp_tune'
    try:
        trial_idx = max([int(i[len("hyp_tune_trial"):-len("_fold0")]) for i in glob(f"{folder}/*")])
        if fold == 0:
            trial_idx += 1
    except:
        trial_idx = 0
    writer = SummaryWriterFix(f"{folder}/trial{trial_idx}_fold{fold}")
    
    # train
    for epoch in range(NUM_EPOCHS):
        train_loss = engine_.train(train_loader)
        valid_dict = engine_.evaluate(val_loader)
        valid_loss = valid_dict['eval_loss']
        valid_accuracy = valid_dict['eval_accuracy']
        
        # update scheduler
        scheduler.step(train_loss)
        
        # update tensorboard
        #writer.add_scalar('Train Loss', train_loss, global_step=epoch)
        #writer.add_scalar('Validation Loss', valid_loss, global_step=epoch)
        #writer.add_scalar('Validation Accuracy', valid_accuracy, global_step=epoch)
        writer.add_hparams(params,
                           {'Train Loss': train_loss,
                            'Validation Loss': valid_loss,
                            'Validation Accuracy': valid_accuracy}, step=epoch)
        
        # if the current valid loss is the lowest
        if valid_loss < best_loss:
            best_loss = valid_loss
            early_stopping_counter = 0 # reset counter
        else:
            early_stopping_counter += 1
            
        if early_stopping_counter > early_stopping_iter:
            break
    
    return valid_loss


def train_loop(fold, df, params, NUM_EPOCHS, bert_model, tokenizer, MAX_LEN, DEVICE):
    #copy bert model to avoid leakage
    bert_model_ = deepcopy(bert_model)
    
    # get X, y from fold
    X_train = df.loc[:, 'text']
    y_train = df.loc[:, 'sentiment']
    
    # dataset
    train_dataset = BERTDataset(X_train, y_train, tokenizer, MAX_LEN)
    
    # sample weights for imbalalanced class
    label_weights_dict = 1 / (y_train.value_counts() / y_train.shape[0])
    label_weights_dict = label_weights_dict.to_dict()
    label_weights = y_train.map(label_weights_dict).tolist()
    
    # weighted sampler
    gen = torch.Generator()
    gen.manual_seed(17678452212620826457)
    weighted_sampler = WeightedRandomSampler(label_weights,
                                    len(label_weights),
                                    replacement=True,
                                    generator=gen)
    
    # loader
    train_loader = DataLoader(train_dataset,
                              batch_size=params['batch_size'],
                              sampler=weighted_sampler)
    
    # model init
    model = models.SentimentModel(bert_model_,
                           bert_model_.config.hidden_size*params['hidden_size_ratio'],
                           y_train.unique().shape[0],
                           num_layers=params['num_layers'],
                           dropout=params['dropout']
                           ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                          factor=0.1,
                                                          patience=500,
                                                          verbose=True)
    
    # engine init
    engine_ = engine.Engine(model, optimizer, criterion, DEVICE)
    
    best_loss = np.inf
    early_stopping_iter = 10
    early_stopping_counter = 0
    
    # tensorboard init
    folder = 'best'
    try:
        trial_idx = max([int(i[len(folder)+1:]) for i in glob(f"{folder}/*")])+1
    except:
        idx = 0
    writer = SummaryWriterFix(f"{folder}/{idx}")
    
    # train
    for epoch in range(NUM_EPOCHS):
        train_loss = engine_.train(train_loader)
        
        # update scheduler
        scheduler.step(train_loss)
        
        # update tensorboard
        writer.add_scalar('Train Loss', train_loss, global_step=epoch)

        # if the current valid loss is the lowest
        if train_loss < best_loss:
            best_loss = train_loss
            early_stopping_counter = 0 # reset counter
            
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            utils.save_checkpoint(checkpoint, filename=f"bert_sentiment_best.pth.tar")
        else:
            early_stopping_counter += 1
            
        if early_stopping_counter > early_stopping_iter:
            break
            
    return best_loss