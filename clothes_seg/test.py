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

    N_EPOCHS = 10000
    BATCH_SIZE = 128
    BUFFER_SIZE = BATCH_SIZE * 2
    ds = 1
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

    MODEL_PATH = os.path.join(WEIGHT_DIR, MODEL_NAME + ".json")
    HISTORY_PATH = os.path.join(WEIGHT_DIR, MODEL_NAME + "_history.json")
    
    with open(MODEL_PATH) as json_data:
        model = tf.keras.models.model_from_json(json.load(json_data))

    WEIGHT_NAME = MODEL_NAME.replace("model","weights") + "{:03}.hdf5".format(int(sys.argv[1]))
    fpath = os.path.join(WEIGHT_DIR, WEIGHT_NAME)
    model.load_weights(fpath)
    

    ######### DATA IMPORT #########
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

    ##### LOAD DATA #####
    NUM_TOTAL = 2

    # get all filenames
    all_filenames = list([x for x in IMG_DIR.iterdir()])
    all_masknames = list([x for x in SEG_DIR.iterdir()])
    
    # train
    image_tensor = np.zeros((NUM_TOTAL, *TARGET_DIMS, ), dtype=np.float32)
    mask_tensor = np.zeros((NUM_TOTAL, *TARGET_DIMS), dtype=np.float32)
    i = 0
    for img_filename, seg_filename in tqdm(zip(all_filenames[:NUM_TOTAL], all_masknames[:NUM_TOTAL]), total=NUM_TOTAL):
        xs, ys = prepare_data(img_filename, seg_filename)
        for x, y in zip(xs, ys):
            image_tensor[i] = x
            mask_tensor[i] = y
            i += 1
            if i >= NUM_TOTAL:
                break
        if i >= NUM_TOTAL:
            break

    image_tensor = np.reshape(image_tensor, image_tensor.shape + (1,))
    mask_tensor = np.reshape(mask_tensor, mask_tensor.shape + (1,))

    def make_rgb(img):
        return (255. * img).astype(np.uint8)

    out = model.predict(image_tensor)[0]
    
    #thresh = 0.45
    #out[np.where(out >= thresh)] = 1
    #out[np.where(out < thresh)] = 0

    #in_img = make_rgb(image_tensor[0])
    in_img = image_tensor[0]
    
    plt.imshow(in_img[:, :, 0])
    plt.imshow(out[:, :, 0], alpha=0.6)
    plt.show()

    plt.imshow(in_img[:, :, 0])
    plt.imshow(mask_tensor[0][:, :, 0], alpha=0.6)
    plt.show()

    
    im_prod = out * in_img
    plt.imshow(im_prod[:, :, 0])
    plt.show()



    im_prod = mask_tensor[0] * in_img
    plt.imshow(im_prod[:, :, 0])
    plt.show()
    
  