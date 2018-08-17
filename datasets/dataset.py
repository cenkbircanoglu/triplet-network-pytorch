# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 01/12/2016 """

import os
import random

import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset

random.seed(1137)


class LabelDataset(Dataset):
    def __init__(self, csv_path=None, image_folder=None, feature_name=None, le=None, transform=None):
        df = pd.read_csv(csv_path)
        index_name = 'URUNANA_ID'
        df[index_name] = df[index_name].astype(str)
        img_filenames = os.listdir(image_folder)
        img_urunana_ids = [i.split('_')[0] for i in img_filenames]
        img_df = pd.DataFrame({index_name: img_urunana_ids, 'path': img_filenames})
        tmp_df = pd.merge(df, img_df, on=[index_name])
        self.image_folder = image_folder
        self.transform = transform
        self.num_inputs = 1
        self.num_targets = 1

        self.X = tmp_df['path']
        self.y = le.transform(tmp_df[feature_name].values)
        self.id_list = tmp_df[index_name]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.image_folder, self.X[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y[index]

    def __len__(self):
        print(len(self.X.index))
        return len(self.X.index)
