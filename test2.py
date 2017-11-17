from glob import glob
from PIL import Image
import numpy as np
import os

image_dir = '/home/andy/Github/2018_cvpr_gan/flickr/all_dof/all'
name_list = glob(os.path.join(image_dir, '*.jpg'))

for idx, train_path in enumerate(name_list):
    print('[{:d}/{:d}]'.format(idx, len(name_list)))
    img_name = train_path.split('/')[-1].split('.')[0]
    x = np.array(Image.open(train_path))
    if len(x.shape) < 3:
        continue

    rows = x.shape[0]
    cols = x.shape[1]

    left_shift = np.zeros_like(x, dtype=np.int32)
    left_shift[:, 1:, :] = x[:, :-1, :]

    res_image = np.maximum(np.abs(left_shift - x) * 10, 255)
    x_png = Image.fromarray(res_image.astype(np.uint8))
    x_png.save('{}/{}_res.png'.format(
        '/home/andy/Github/2018_cvpr_gan/flickr/all_dof/filter', img_name), format='PNG')

    x_png = Image.fromarray(x.astype(np.uint8))
    x_png.save('{}/{}.png'.format(
        '/home/andy/Github/2018_cvpr_gan/flickr/all_dof/filter', img_name), format='PNG')

    if idx >= 0:
        break

