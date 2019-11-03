#!/usr/bin/env python3

from base64 import b64decode
import cv2
from io import BytesIO
import numpy as np
import argparse
import sys
from PIL import Image, ImageOps
from sanic import Sanic, response
from sanic.log import logger
from StarGAN.solver import Solver
from StarGAN.data_loader import get_loader

# ======================
# Define the app
# ======================
app = Sanic()

# Serve index.html statically
app.static("/", "../static")

def str2bool(v):
    return v.lower() in ('true')

def makeconfig():
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')
    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--celeba_image_dir', type=str, default='data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='StarGAN/data/celeba/list_attr_celeba.txt')
    parser.add_argument('--rafd_image_dir', type=str, default='StarGAN/data/RaFD/train')
    parser.add_argument('--log_dir', type=str, default='stargan/logs')
    parser.add_argument('--model_save_dir', type=str, default='stargan/models')
    parser.add_argument('--sample_dir', type=str, default='stargan/samples')
    parser.add_argument('--result_dir', type=str, default='stargan/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args(
        "--mode test --dataset CelebA --image_size 256 --c_dim 5 --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young --model_save_dir=StarGAN\\stargan_celeba_256\\models --result_dir StarGAN\\stargan_celeba_256\\results --use_tensorboard False".split()
    )

    return config
    
# instantiate
config = makeconfig()
celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers, Image.open("StarGAN\\me.jpg"))
solver = Solver(celeba_loader, rafd_loader=None, config=config)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

import numpy as np


mask = cv2.imread('mask.png')
@app.route('/transform', methods=["POST"])
async def transform(request):
    # Get files from input
    decoded_file = b64decode(request.body.split(b',')[1])
    feature_id = int(request.headers.get("Target", "0"))
    image = Image.open(BytesIO(decoded_file))
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    new_image = image.convert("RGBA")
    changed = False
    for (x, y, w, h) in faces:
        if w * h > (100*100):
            top = int(0.4 * h)
            bottom = int(0.3 * h)
            left = int(0.2 * w)
            right = int(0.2 * w)
            x1 = max(0, x-left)
            y1 = max(0, y-top)
            x2 = min(image.width, x+w+right)
            y2 = min(image.height, y+h+bottom)
            try:
                face = image.crop((x1,y1,x2,y2)).resize((256,256), Image.LANCZOS)
                #face = np.array(face)
                #face = cv2.blur(face,(5,5))
                #face = Image.fromarray(face)
                #new_image = face
                modded_face = solver.test(face, feature_id).resize((x2-x1,y2-y1), Image.LANCZOS).convert("RGBA")
                #smask = mask.resize((x2-x1,y2-y1))
                new_image.paste(modded_face, (x1, y1))
                changed = True
            except KeyboardInterrupt:
                sys.exit()
            #except Exception as e:
                #logger.error(e)
    out_bytes = BytesIO()
    if changed:
        new_image.convert("RGB").save(out_bytes, format='jpeg')
    return response.raw(out_bytes.getvalue())


# ============================
# Run the app
# ============================
if __name__ == '__main__':
    ssl = {'cert': "certs/MyCertificate.crt", 'key': "certs/MyKey.key"}
    app.run(host='0.0.0.0', port=3000, ssl=ssl)
