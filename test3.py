from glob import glob
from PIL import Image
import numpy as np
import os

image_dir = '/home/andy/Github/2018_cvpr_gan/dataset/test_flower17/valB'
name_list = glob(os.path.join(image_dir, '*.png'))

for idx, train_path in enumerate(name_list):
    print('[{:d}/{:d}]'.format(idx, len(name_list)))
    img_name = train_path.split('/')[-1].split('.')[0]

    image_full_path = os.path.join('/home/andy/Github/2018_cvpr_gan/dataset/test_flower17/valA', img_name+'.jpg')
    x_png = Image.open(image_full_path)
    x_png.save('{}/{}.png'.format(
        '/home/andy/Github/2018_cvpr_gan/dataset/test_flower17/valA_filter', img_name), format='PNG')


