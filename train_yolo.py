import os
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from dataset import YoloDataset
from model import YoloModel
from loss import YoloLoss

from utils.config import config
from utils.early_stopping import EarlyStopping
from utils.log import create_log


def train_yolo(conf, log, model, train_loader, val_loader, deivce):

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.init_lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=15)
    early_stopping = EarlyStopping(min_delta=conf.min_delta, patience=conf.patience, logger=log, verbose=1)
    loss_fn = YoloLoss(conf.l_coord, conf.l_noobj)

    early_stopping.on_train_begin()
    for epoch in range(1, conf.epochs+1):
        ''' train model '''
        model.train()
        for batch_i, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            train_pred = model(batch_x.to(device))
            train_loss = loss_fn(train_pred, batch_y.to(device))
            train_loss.backward()
            optimizer.step()
            if batch_i % 50 == 0:
                log.info('Train: Epoch {} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, batch_i * len(batch_x), len(train_loader.dataset), train_loss.item()))
    
        ''' valid model '''
        model.eval()
        valid_pred = []
        for _, (batch_x, batch_y) in enumerate(val_loader):
            with torch.no_grad():
                valid_pred.append(model(batch_x.to(device)))
        valid_pred = torch.concat(valid_pred, dim=0)
        valid_loss = loss_fn(valid_pred, val_loader.tgt.to(device))
        lr_scheduler.step(metrics=valid_loss, epoch=epoch)

        log.info(f"Metric: Epoch {epoch} / {conf.epochs}, valid_loss: {valid_loss.item():.4f}")

        save_checkpoint, stop_training = early_stopping.on_epoch_end(epoch=epoch, monitor=valid_loss.item(), monitor_name='valid_loss')
        if save_checkpoint:
            model_file = os.path.join(conf.output_dir, 'YoloModel.pth')
            torch.save({"model_state_dict": model.state_dict(),
                        "best_valid_loss": valid_loss.item(),
                        "opt_state_dict":optimizer.state_dict()},
                        model_file)
        if stop_training:
            break


def test_yolo(model, log, test_loader, device):
    loss_fn = YoloLoss(conf.l_coord, conf.l_noobj)

    tot_pred = []
    model.eval()
    with torch.no_grad():
        for _, (batch_x, _) in enumerate(test_loader):
            pred = model(batch_x.to(device))
            tot_pred.append(pred)

    tot_pred = torch.concat(tot_pred, axis=0)
    test_loss = loss_fn(tot_pred, test_loader.tgt.to(device))

    # mMAP = 

    log.info(f"Test: test_loss={test_loss.item()}")
        

if __name__ == "__main__":
    conf = config(batch_size=256, init_lr=1e-3, epochs=500,
                  min_delta=0., patience=20,
                  S=7, B=2, C=20, l_coord=5, l_noobj=0.5,
                  output_dir='./outputs',
                  pretrained_model=None,
                  tricks='')
    conf.save_config(datetime.now(tz=None).strftime("%m%d%H%M")+'_'+conf.tricks)

    log = create_log(log_dir=conf.output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YoloModel(conf.S, conf.B, conf.C).to(device)

    ''' load pretrained model '''
    if conf.pretrained_model is not None:
        model_state_dict = torch.load(conf.pretrained_model)
        model.load_state_dict(model_state_dict)

    ''' load dataset '''
    root_dir = r'C:/Users/Administrator/Desktop/VOCdevkit/VOC2007'
    train_dataset = YoloDataset(root_dir, 'test')
    val_dataset = YoloDataset(root_dir, 'val')
    test_dataset = YoloDataset(root_dir, 'train')
    log.info(f'train_dataset length:{len(train_dataset)}, val_dataset length:{len(val_dataset)}, test_dataset length:{len(test_dataset)}')

    ''' train model '''
    train_loader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=True)
    train_yolo(conf, log, model, train_loader, val_loader, device)

    ''' test model '''
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    test_yolo(model, log, test_loader, device)

