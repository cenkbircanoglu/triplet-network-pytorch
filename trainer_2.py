# -*- coding: utf-8 -*-
import logging
import logging.config
import os

import torch
from torch.nn import CrossEntropyLoss
from torchsample.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from torchsample.metrics import CategoricalAccuracy
from torchsample.modules import ModuleTrainer

from torchvision import models
from config import set_config, BaseConfig
from datasets import loaders
from utils.make_dirs import create_dirs

logging.config.fileConfig('etc/logging.conf')


def run():
    device = 0 if torch.cuda.is_available() else -1
    config = BaseConfig()
    logging.info('%s_cross_entropy/ckpt.pth.tar' % config.result_dir)
    if os.path.exists('%s_cross_entropy/ckpt.pth.tar' % config.result_dir):
        return True
    logging.info("Triplet Trainer Not Return")
    create_dirs()
    tr_data_loader, val_data_loader, te_data_loader = loaders.data_loaders(shuffle=True)

    model = getattr(models, config.network)(num_classes=len(tr_data_loader.dataset.y))

    model
    criterion = CrossEntropyLoss()

    if device == 0:
        model.cuda()
        criterion.cuda()
    trainer = ModuleTrainer(model)
    epochs = config.epochs

    callbacks = [EarlyStopping(monitor='val_acc', patience=20),
                 ModelCheckpoint('%s_cross_entropy' % config.result_dir, save_best_only=True, verbose=1),
                 CSVLogger("%s_cross_entropy/logger.csv" % config.result_dir)]

    metrics = [CategoricalAccuracy()]

    trainer.compile(loss=criterion, optimizer='adam', metrics=metrics)
    trainer.set_callbacks(callbacks)

    trainer.fit_loader(tr_data_loader, val_loader=val_data_loader, num_epoch=epochs, verbose=2, cuda_device=device)

    tr_loss = trainer.evaluate_loader(tr_data_loader, cuda_device=device)
    logging.info(tr_loss)
    val_loss = trainer.evaluate_loader(val_data_loader, cuda_device=device)
    logging.info(val_loss)
    te_loss = trainer.evaluate_loader(te_data_loader, cuda_device=device)
    logging.info(te_loss)
    with open('%s_cross_entropy' %config.log_path, "a") as f:
        f.write('Train %s\nVal:%s\nTest:%s\n' % (str(tr_loss), str(val_loss), te_loss))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--feature_name', type=str, default="DETAYGRUP_ADI")
    parser.add_argument('--network', type=str, default="densenet")
    parser.add_argument('--embedding', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()

    kwargs = vars(args)
    set_config(**kwargs)

    run()
