# -*- coding: utf-8 -*-
import argparse
import os

from PIL import Image


def verify_or_delete(dir):
    for root, dirs, files in os.walk(dir):
        for f in files:
            fname = os.path.join(root, f)
            try:
                img = Image.open(fname)
                img = img.convert("RGB")
                img.load()
            except IOError as ie:
                print(fname)
                os.remove(fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=str,
                        help="Directory of images to partition in-place to 'train' and 'val' directories.")
    args = parser.parse_args()

    verify_or_delete(args.image_dir)
