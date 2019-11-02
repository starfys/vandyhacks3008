import numpy as np
import os
import sys
import json

import tensorflow as tf
from tqdm import tqdm

from utils.augmentations import *
from utils.tfrecord_utils import *
from models.unet import *

def running_average(old_average, cur_val, n):
    return old_average * (n-1)/n + cur_val/n

def mil_prediction(pred, n=1):
    # generalized, returns top n 
    i = tf.argsort(pred[:, 1], axis=0)
    i = i[len(pred) - n : len(pred)]

    return (tf.gather(pred, i), i)

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":

    ########## HYPERPARAMETER SETUP ##########

    N_EPOCHS = 10000
    BATCH_SIZE = 128
    BUFFER_SIZE = BATCH_SIZE * 2
    ds = 2
    instance_size = (1024, 1024)
    learning_rate = 1e-4
    progbar_length = 10
    CONVERGENCE_EPOCH_LIMIT = 10
    epsilon = 1e-4

    ########## DIRECTORY SETUP ##########

    MODEL_NAME = "unet"
    WEIGHT_DIR = os.path.join(
        "models", 
        "weights", 
        MODEL_NAME, 
    )

    RESULTS_DIR = os.path.join(
        "results",
    )            

    # files and paths
    for d in [WEIGHT_DIR, RESULTS_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    MODEL_PATH = os.path.join(WEIGHT_DIR, MODEL_NAME + ".json")
    HISTORY_PATH = os.path.join(WEIGHT_DIR, MODEL_NAME + "_history.json")
    
    model = unet(
        MODEL_PATH,
        3,
        ds=ds,
        lr=learning_rate,
        num_gpus=1,
        verbose=1,
    )

    

    ######### DATA IMPORT #########
    augmentations = [flip_dim1, flip_dim2, rotate_2D]

    DATA_DIR = Path("D:/data/deepfashion/validation/validation")
    IMG_DIR = DATA_DIR / "image"
    SEG_DIR = DATA_DIR / "annos"
    
    TARGET_DIMS = (1024, 1024)

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
        
    NUM_TOTAL = 2000
    
    # train
    image_tensor = np.zeros(NUM_TOTAL, *TARGET_DIMS, 3)
    mask_tensor = np.zeros(NUM_TOTAL, *TARGET_DIMS, 1)
    i = 0
    for img_filename, seg_filename in tqdm(zip(IMG_DIR.iterdir(), SEG_DIR.iterdir()), total=NUM_TOTAL):
        xs, ys = prepare_data(img_filename, seg_filename)
        for x, y in zip(xs, ys):
            plt.imshow(x)
            plt.imshow(y, alpha=0.6)
            plt.show()
            image_tensor[i] = x
            mask_tensor[i] = y
            i += 1
            if i > NUM_TOTAL:
                break

    train_dataset = tf.data.Dataset.from_tensor_slices(
            (image_tensor, mask_tensor))
            
    # val
    NUM_VAL = 500
    image_tensor_val = np.zeros(NUM_TOTAL, *TARGET_DIMS, 3)
    mask_tensor_val = np.zeros(NUM_TOTAL, *TARGET_DIMS, 1)
    i = 0
    for img_filename, seg_filename in tqdm(zip(IMG_DIR.iterdir().skip(NUM_TOTAL), 
                                        SEG_DIR.iterdir().skip(NUM_TOTAL)), total=NUM_VAL):
        xs, ys = prepare_data(img_filename, seg_filename)
        for x, y in zip(xs, ys):
            plt.imshow(x)
            plt.imshow(y, alpha=0.6)
            plt.show()
            image_tensor_val[i] = x
            mask_tensor_val[i] = y
            i += 1
            if i > NUM_VAL:
                break

    train_dataset = tf.data.Dataset.from_tensor_slices(
            (image_tensor, mask_tensor))
            
    val_dataset = tf.data.Dataset.from_tensor_slices(
            (image_tensor_val, mask_tensor_val))


    '''
    for f in augmentations:
        train_dataset = train_dataset.map(
                lambda x, y: 
                tf.cond(tf.random.uniform([], 0, 1) > 0.9, 
                    lambda: (f(x), y),
                    lambda: (x, y)
                ), num_parallel_calls=4,)
    '''
    
    history = model.fit(
        train_dataset,
        batch_size=64,
        epochs=3,
        validation_data=val_dataset
    )

  