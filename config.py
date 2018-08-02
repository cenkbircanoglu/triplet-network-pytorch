import os
from sys import platform as _platform

from utils.singleton import singleton

PAR = '/disk2/results/'

Config = None


@singleton
class BaseConfig(object):
    def local_paths(self):
        global PAR
        self.img_folder = './datasets/images'
        self.csv_path = './datasets/'
        PAR = './results/'
        self.tr_csv_path = os.path.join(self.csv_path, 'train.csv')
        self.val_csv_path = os.path.join(self.csv_path, 'val.csv')
        self.te_csv_path = os.path.join(self.csv_path, 'test.csv')

    def __init__(self, feature_name=None, batch_size=256, epochs=20, num_workers=4, embedding=128, network=None,
                 **kwargs):
        self.img_folder = '/disk2/datasets/beymen/images_224/'
        self.csv_path = '/disk2/datasets/beymen/'
        self.tr_csv_path = os.path.join(self.csv_path, 'train.csv')
        self.val_csv_path = os.path.join(self.csv_path, 'val.csv')
        self.te_csv_path = os.path.join(self.csv_path, 'test.csv')

        if _platform == "darwin":
            self.local_paths()
        self.result_dir = os.path.join(PAR, './results/%s/%s/%s' % (feature_name, network, embedding))
        self.log_path = os.path.join(PAR, './results/%s/%s/%s.log' % (feature_name, network, embedding))
        self.batch_size = batch_size
        self.feature_name = feature_name
        self.epochs = epochs
        self.num_workers = num_workers
        self.embedding = embedding
        self.network = network
        self.width = 224
        self.height = 224
        self.channel = 3


def set_config(**kwargs):
    BaseConfig(**kwargs)


def get_config():
    return BaseConfig()
