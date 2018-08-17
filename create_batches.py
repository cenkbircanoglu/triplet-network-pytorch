# -*- coding: utf-8 -*-
import logging
import logging.config
import os

import numpy as np
import torch
from torchsample.modules import ModuleTrainer
from tqdm import tqdm

import models
from config import BaseConfig, set_config
from datasets import loaders
from datasets.data_utils import SemihardNegativeTripletSelector
from losses.online_triplet import OnlineTripletLoss
from utils.make_dirs import create_dirs

logging.config.fileConfig('etc/logging.conf')


def run():
    config = BaseConfig()
    logging.info('%s/train_embeddings.csv' % config.result_dir)
    result_dir = config.result_dir
    logging.info('%s/train_embeddings.csv' % result_dir)
    if os.path.exists('%s/train_embeddings.csv' % result_dir) and os.path.exists('%s/test_embeddings.csv' % result_dir):
        return True
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    logging.info("Saved Module Trainer Not Return")
    create_dirs()
    device = 0 if torch.cuda.is_available() else -1

    tr_data_loader, val_data_loader, te_data_loader = loaders.data_loaders()

    model = getattr(models, config.network).get_network()(embedding_size=config.embedding)

    check_point = os.path.join(config.result_dir, "ckpt.pth.tar")
    if os.path.isfile(check_point):
        logging.info("=> loading checkpoint '{}'".format(check_point))
        checkpoint = torch.load(check_point)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(check_point, checkpoint['epoch']))
    else:
        logging.info("=> no checkpoint found at '{}'".format(check_point))
        return
    margin = 1.
    criterion = OnlineTripletLoss(margin, SemihardNegativeTripletSelector(margin))
    if device == 0:
        model.cuda()
        criterion.cuda()
    trainer = ModuleTrainer(model)

    trainer.compile(loss=criterion, optimizer='adam')

    logging.info('Train Prediction')
    tr_y_pred = trainer.predict_loader(tr_data_loader, cuda_device=device)
    logging.info('Train Save Embeddings')
    save_embeddings(tr_y_pred, '%s/train_embeddings.csv' % config.result_dir)
    logging.info('Train Save Labels')
    save_labels(tr_data_loader, '%s/train_labels.csv' % config.result_dir)
    tr_data_loader.dataset.id_list.to_csv('%s/train_ids.csv' % config.result_dir, header=None, index=None)

    logging.info('Validation Prediction')
    val_y_pred = trainer.predict_loader(val_data_loader, cuda_device=device)
    logging.info('Validation Save Embeddings')
    save_embeddings(val_y_pred, '%s/val_embeddings.csv' % config.result_dir)
    logging.info('Validation Save Labels')
    save_labels(val_data_loader, '%s/val_labels.csv' % config.result_dir)
    val_data_loader.dataset.id_list.to_csv('%s/val_ids.csv' % config.result_dir, header=None, index=None)

    logging.info('Test Prediction')
    te_y_pred = trainer.predict_loader(te_data_loader, cuda_device=device)
    logging.info('Test Save Embeddings')
    save_embeddings(te_y_pred, '%s/test_embeddings.csv' % config.result_dir)
    logging.info('Test Save Labels')
    save_labels(te_data_loader, '%s/test_labels.csv' % config.result_dir)
    te_data_loader.dataset.id_list.to_csv('%s/test_ids.csv' % config.result_dir, header=None, index=None)


def save_embeddings(data, outputfile):
    if not os.path.exists(os.path.dirname(outputfile)):
        os.makedirs(os.path.dirname(outputfile))
    with open(outputfile, 'a') as f:
        if type(data) == list:
            data = data[0]
        np.savetxt(f, data.data.cpu().numpy())


def save_labels(loader, outputfile):
    if not os.path.exists(os.path.dirname(outputfile)):
        os.makedirs(os.path.dirname(outputfile))
    for i, data in tqdm(enumerate(loader, 0), total=loader.__len__()):
        img, label = data
        with open(outputfile, 'a') as f:
            np.savetxt(f, label.numpy())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--feature_name', type=str, default="DETAYGRUP_ADI")
    parser.add_argument('--network', type=str, default="alexnet")
    parser.add_argument('--embedding', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)

    args = parser.parse_args()

    kwargs = vars(args)
    set_config(**kwargs)

    run()
