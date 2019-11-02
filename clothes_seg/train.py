import numpy as np
import os
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.path import Path as Tracer

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tqdm import tqdm
from utils.pad import *
from utils import preprocess
from utils.tfrecord_utils import *
from utils.patch_ops import *
from models.losses import *

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

    if len(sys.argv) > 1:
        pretrained_weights = Path(sys.argv[1]).absolute()
        prev_epoch = int(str(pretrained_weights.name)[9:12])
    else:
        pretrained_weights = None
        prev_epoch = 0
    
    

    N_EPOCHS = 10000
    BATCH_SIZE = 128
    BUFFER_SIZE = BATCH_SIZE * 2
    ds = 2
    TARGET_DIMS = (256, 256)

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
         num_channels=1,
         loss="binary_crossentropy",
         ds=ds,
         lr=learning_rate,
         num_gpus=1,
         verbose=1,)

    if pretrained_weights:
        model.load_weights(str(pretrained_weights))


    ######### DATA IMPORT #########
    augmentations = [flip_dim1, flip_dim2, rotate_2D]

    DATA_DIR = Path("D:/data/deepfashion/validation/validation")
    IMG_DIR = DATA_DIR / "image"
    SEG_DIR = DATA_DIR / "annos"
    
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    def prepare_data(x_filename, y_filename):
        xs = []
        ys = []

        img = plt.imread(str(x_filename))
        img = rgb2gray(img)

        with open(str(y_filename), 'r') as f:
            json_data = json.load(f)
        
        item_keys = [tmp_x for tmp_x in json_data.keys() if "item" in tmp_x]
        num_items = len(item_keys)

        dims = img.shape[:2]

        for k in item_keys:
            vertices = json_data[k]["segmentation"][0]

            xc = []
            yc = []
            for x, y in zip(vertices[::2], vertices[1::2]):
                xc.append(x)
                yc.append(y)
            xc = np.array(xc)
            yc = np.array(yc)

            xycrop = np.vstack((xc, yc)).T
            nr, nc = img.shape[:2]
            ygrid, xgrid = np.mgrid[:nr, :nc]
            xypix = np.vstack((xgrid.ravel(), ygrid.ravel())).T
            pth = Tracer(xycrop, closed=False)
            mask = pth.contains_points(xypix)
            mask = mask.reshape(img.shape[:2])


            x_resize = pad_crop_image_2D(img.astype(np.float32)/255., TARGET_DIMS)
            y_resize = pad_crop_image_2D(mask.astype(np.float32), TARGET_DIMS)

            xs.append(x_resize)
            ys.append(y_resize)

        xs = np.array(xs)
        ys = np.array(ys)

        return xs, ys

    ##### CALLBACKS #####
    callbacks_list = []

    # Checkpoint
    WEIGHT_NAME = MODEL_NAME.replace("model","weights") + f"_{prev_epoch:03}" + "_{epoch:03}" + ".hdf5"
    fpath = os.path.join(WEIGHT_DIR, WEIGHT_NAME)
    checkpoint = ModelCheckpoint(fpath,
                                 verbose=0,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=True)
    callbacks_list.append(checkpoint)


    # Early Stopping, used to quantify convergence
    # convergence is defined as no improvement by 1e-4 for 10 consecutive epochs
    #es = EarlyStopping(monitor='loss', min_delta=0, patience=10)
    #es = EarlyStopping(monitor='loss', min_delta=1e-8, patience=10)
    # The code continues even if the validation/training accuracy reaches 1, but loss is not.
    # For a classification task, accuracy is more important. For a regression task, loss
    # is important
    es = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=100)
    callbacks_list.append(es)


    ##### LOAD DATA #####
    START = 1000
    NUM_TOTAL = 1000

    # get all filenames
    all_filenames = list([x for x in IMG_DIR.iterdir()])
    all_masknames = list([x for x in SEG_DIR.iterdir()])
    
    # train
    image_tensor = np.zeros((NUM_TOTAL, *TARGET_DIMS,), dtype=np.float32)
    mask_tensor = np.zeros((NUM_TOTAL, *TARGET_DIMS), dtype=np.float32)
    i = 0

    print()

    TEMPLATE = "\rLoading data... {}/{} [{:{}<{}}]"
    sys.stdout.write(TEMPLATE.format(
        1, 
        NUM_TOTAL, 
        "=" * 0, 
        '-', 
        progbar_length,
    ))

    for img_filename, seg_filename in zip(all_filenames[START:START + NUM_TOTAL], all_masknames[START:START+NUM_TOTAL]):
        xs, ys = prepare_data(img_filename, seg_filename)
        for x, y in zip(xs, ys):
            image_tensor[i] = x
            mask_tensor[i] = y
            i += 1

            sys.stdout.write(TEMPLATE.format(
                i, 
                NUM_TOTAL,
                "=" * min(int(progbar_length*(i/NUM_TOTAL)), 
                          progbar_length),
                "-",
                progbar_length,
            ))
            sys.stdout.flush()

            if i >= NUM_TOTAL:
                break
        if i >= NUM_TOTAL:
            break

    image_tensor = np.reshape(image_tensor, image_tensor.shape + (1,))
    mask_tensor = np.reshape(mask_tensor, mask_tensor.shape + (1,))

    history = model.fit(
        image_tensor, mask_tensor,
        validation_split=0.2,
        epochs=1000000,
        batch_size=16,
        callbacks=callbacks_list,
        verbose=1,
        shuffle=True,
    )

  