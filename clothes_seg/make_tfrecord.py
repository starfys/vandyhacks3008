import numpy as np
import os
import sys
import pandas as pd
from matplotlib.path import Path as Tracer
import matplotlib.pyplot as plt
import json

import tensorflow as tf
import nibabel as nib

from sklearn.utils import shuffle

from tqdm import tqdm
from utils.pad import *
from utils import preprocess
from utils.tfrecord_utils import *
from utils.patch_ops import *
from pathlib import Path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Missing cmd line argument")
        sys.exit()

    ######### DIRECTRY SETUP #########

    # pass the preprocessed data directory here
    DATA_DIR = Path("D:/data/deepfashion/{}/{}".format(sys.argv[1], sys.argv[1]))
    IMG_DIR = DATA_DIR / "image"
    SEG_DIR = DATA_DIR / "annos"

    TF_RECORD_FILENAME = DATA_DIR / "ds.tfrecord"

    TARGET_DIMS = (1024, 1024)

    ######### PARSE #######

    def prepare_data(x_filename, y_filename):
        xs = []
        ys = []

        x = plt.imread(str(x_filename))
        with open(str(y_filename), 'r') as f:
            json_data = json.load(f)
        
        item_keys = [x for x in json_data.keys() if "item" in x]
        num_items = len(item_keys)

        dims = x.shape[:2]

        for k in item_keys:
            vertices = json_data[k]["segmentation"][0]
            xc, yc = [], []
            for xpt, ypt in zip(vertices[::2], vertices[1::2]):
                xc.append(xpt)
                yc.append(ypt)
            xc = np.array(xc)
            yc = np.array(yc)

            xycrop = np.vstack((xc, yc)).T
            nr, nc = dims
            ygrid, xgrid = np.mgrid[:nr, :nc]
            xypix = np.vstack((xgrid.ravel(), ygrid.ravel())).T

            pth = Tracer(xycrop, closed=False)
            mask = pth.contains_points(xypix)
            mask = mask.reshape(dims)

            xs.append(pad_crop_image_2D(x, TARGET_DIMS).astype(np.uint8))
            ys.append(pad_crop_image_2D(mask, TARGET_DIMS).astype(np.uint8))

        xs - np.array(xs)
        ys = np.array(ys)

        return xs, ys

    ######### WRITE #######

    with tf.io.TFRecordWriter(str(TF_RECORD_FILENAME)) as writer:
        for img_filename, seg_filename in tqdm(zip(IMG_DIR.iterdir(), SEG_DIR.iterdir()), total=32153):
            xs, ys = prepare_data(img_filename, seg_filename)
            for x, y in zip(xs, ys):
                plt.imshow(x)
                plt.imshow(y, alpha=0.6)
                plt.show()
                input()
                #tf_example = image_example(x, y)
                #writer.write(tf_example.SerializeToString())
