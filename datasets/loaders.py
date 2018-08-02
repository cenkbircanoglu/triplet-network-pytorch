# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 01/12/2016 """

import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from config import BaseConfig
from datasets.batch_sampler import BalancedBatchSampler
from datasets.dataset import LabelDataset

random.seed(1137)


def train_encoder(tr_csv_path, val_csv_path, te_csv_path, feature_name):
    tr_data = pd.read_csv(tr_csv_path)
    val_data = pd.read_csv(val_csv_path)
    te_data = pd.read_csv(te_csv_path)

    le = LabelEncoder()
    y = np.concatenate([tr_data[feature_name].values, val_data[feature_name].values, te_data[feature_name].values],
                       axis=0)
    le.fit(y)

    return le


def data_loaders():
    config = BaseConfig()

    batch_size = config.batch_size

    transform = transforms.Compose(
        [transforms.Scale((config.height, config.width)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    tr_csv_path, val_csv_path, te_csv_path, feature_name = config.tr_csv_path, config.val_csv_path, config.te_csv_path, config.feature_name
    encoder = train_encoder(tr_csv_path, val_csv_path, te_csv_path, feature_name)

    tr_dataset = LabelDataset(
        image_folder=config.img_folder,
        transform=transform,
        le=encoder,
        csv_path=tr_csv_path,
        feature_name=feature_name
    )

    val_dataset = LabelDataset(
        image_folder=config.img_folder,
        transform=transform,
        le=encoder,
        csv_path=tr_csv_path,
        feature_name=feature_name
    )

    te_dataset = LabelDataset(
        image_folder=config.img_folder,
        transform=transform,
        le=encoder,
        csv_path=tr_csv_path,
        feature_name=feature_name
    )

    kwargs = {'num_workers': 16, 'pin_memory': True} if torch.cuda.is_available() else {}
    tr_data_loader = DataLoader(tr_dataset,
                                shuffle=False,
                                batch_size=batch_size, **kwargs)

    val_data_loader = DataLoader(val_dataset,
                                 shuffle=False,
                                 batch_size=batch_size, **kwargs)

    te_data_loader = DataLoader(te_dataset,
                                shuffle=False,
                                batch_size=batch_size, **kwargs)

    return tr_data_loader, val_data_loader, te_data_loader


def online_triplet_loaders():
    config = BaseConfig()
    batch_size = config.batch_size
    transform = transforms.Compose(
        [transforms.Scale((config.height, config.width)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    tr_csv_path, val_csv_path, te_csv_path, feature_name = config.tr_csv_path, config.val_csv_path, config.te_csv_path, config.feature_name
    encoder = train_encoder(tr_csv_path, val_csv_path, te_csv_path, feature_name)

    tr_dataset = LabelDataset(
        image_folder=config.img_folder,
        transform=transform,
        le=encoder,
        csv_path=tr_csv_path,
        feature_name=feature_name
    )

    val_dataset = LabelDataset(
        image_folder=config.img_folder,
        transform=transform,
        le=encoder,
        csv_path=val_csv_path,
        feature_name=feature_name
    )

    te_dataset = LabelDataset(
        image_folder=config.img_folder,
        transform=transform,
        le=encoder,
        csv_path=te_csv_path,
        feature_name=feature_name
    )

    train_batch_sampler = BalancedBatchSampler(tr_dataset.X, tr_dataset.y, n_classes=8, n_samples=int(batch_size / 8))
    val_batch_sampler = BalancedBatchSampler(val_dataset.X, val_dataset.y, n_classes=8, n_samples=int(batch_size / 8))
    test_batch_sampler = BalancedBatchSampler(te_dataset.X, te_dataset.y, n_classes=8, n_samples=int(batch_size / 8))

    kwargs = {'num_workers': 16, 'pin_memory': True} if torch.cuda.is_available() else {}
    tr_data_loader = DataLoader(tr_dataset, batch_sampler=train_batch_sampler, **kwargs)
    val_data_loader = DataLoader(val_dataset, batch_sampler=val_batch_sampler, **kwargs)
    te_data_loader = DataLoader(te_dataset, batch_sampler=test_batch_sampler, **kwargs)

    return tr_data_loader, val_data_loader, te_data_loader
