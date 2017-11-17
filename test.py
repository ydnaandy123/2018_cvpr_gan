from glob import glob
from PIL import Image
import numpy as np
import os

image_dir = '/home/andy/Github/2018_cvpr_gan/flickr/all_color/color'
name_list = glob(os.path.join(image_dir, '*.jpg'))

for idx, train_path in enumerate(name_list):
    print('[{:d}/{:d}]'.format(idx, len(name_list)))
    img_name = train_path.split('/')[-1].split('.')[0]
    x = np.array(Image.open(train_path))
    if len(x.shape) < 3:
        continue

    rows = x.shape[0]
    cols = x.shape[1]

    '''
    rows_crop = rows // 8
    cols_crop = cols // 8

    x_center = x[rows_crop:-rows_crop, cols_crop:-cols_crop, :]
    rows_center = float(x_center.shape[0])
    cols_center = float(x_center.shape[1])
    total_pixel = rows_center * cols_center

    [r, g, b] = np.split(x_center, 3, axis=2)
    gray_pixel = ((r == g) & (g == b) & (r == b))
    gray_pixel_num = np.count_nonzero(gray_pixel)
    color_pixel_num = total_pixel - gray_pixel_num
    color_pixel_percent = color_pixel_num / total_pixel
    if color_pixel_percent < 0.3:
        continue
    '''

    [r, g, b] = np.split(x, 3, axis=2)
    gray_pixel = ((r == g) & (g == b) & (r == b))
    gray_pixel_num = np.count_nonzero(gray_pixel)
    gray_pixel_percent = float(gray_pixel_num) / float(rows * cols)
    if gray_pixel_percent < 0.5 or gray_pixel_percent > 0.8:
        continue

    gray = np.ones_like(x[:, :, 0], dtype=np.uint8) * 255
    gray[gray_pixel[:, :, 0]] = 0

    rows_crop = rows // 8
    cols_crop = cols // 8
    gray_pixel[rows_crop:-rows_crop, cols_crop:-cols_crop] = True
    gray_pixel_num = np.count_nonzero(gray_pixel)
    gray_pixel_percent = float(gray_pixel_num) / float(rows * cols)
    if gray_pixel_percent < 0.98:
        continue

    # x_png = Image.fromarray(gray.astype(np.uint8))
    # x_png.save('{}/{}_gray.png'.format(
    #     '/home/andy/Github/2018_cvpr_gan/flickr/all_color/filter_color', img_name), format='PNG')

    x_png = Image.fromarray(x.astype(np.uint8))
    x_png.save('{}/{}.png'.format(
        '/home/andy/Github/2018_cvpr_gan/flickr/all_color/filter_color', img_name), format='PNG')

    #if idx > 50:
    #    break

