import os
import numpy as np
from datetime import datetime

import torch
from model import YoloModel
from loss import YoloLoss

from utils.config import config
from utils.early_stopping import EarlyStopping
from utils.log import create_log


def train_yolo(conf, log, model, train_loader, valid_loader):

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
            train_pred = model(batch_x)
            train_loss = loss_fn(train_pred, batch_y)
            train_loss.backward()
            optimizer.step()
            if batch_i % 50 == 0:
                log.info('Train: Epoch {} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, batch_i * len(batch_x), len(train_loader.dataset), train_loss.item()))
    
        ''' valid model '''
        model.eval()
        valid_pred = []
        for _, (batch_x, batch_y) in enumerate(valid_loader):
            with torch.no_grad():
                valid_pred.append(model(batch_x))
        valid_pred = torch.concat(valid_pred, dim=0)
        valid_loss = loss_fn(valid_pred, valid_loader.tgt)
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


def test_yolo(model, log, test_loader):
    for _, (batch_x, batch_y) in enumerate(test_loader):
        pass

if __name__ == "__main__":
    conf = config(batch_size=256, init_lr=1e-3, epochs=500,
                  min_delta=0., patience=20,
                  S=7, B=2, C=20, l_coord=5, l_noobj=0.5,
                  output_dir='./outputs',
                  pretrained_model=None,
                  tricks='')
    conf.save_config(datetime.now(tz=None).strftime("%m%d%H%M")+' '+conf.tricks)

    log = create_log(log_dir=conf.output_dir)

    device = torch.device('GPU' if torch.cuda.is_available() else 'CPU')
    model = YoloModel(conf.S, conf.B, conf.C).to(device)

    ''' load pretrained model '''
    if conf.pretrained_model is not None:
        model_state_dict = torch.load(conf.pretrained_model)
        model.load_state_dict(model_state_dict)

    ''' train model '''
    train_yolo(conf, log, model, train_loader, valid_loader)

    ''' test model '''
    test_yolo(model, log, test_loader)
