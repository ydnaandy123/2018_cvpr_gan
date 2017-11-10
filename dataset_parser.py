import os
import tensorflow as tf
import numpy as np
from PIL import Image
from glob import glob
import random


class ImagePool:
    """ History of generated images
        Same logic as https://github.com/junyanz/CycleGAN/blob/master/util/image_pool.lua
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []

    def query(self, image):
        if self.pool_size == 0:
            return image
        if len(self.images) < self.pool_size:
            self.images.append(image)
            return image
        else:
            p = random.random()
            if p > 0.5:
                # use old image
                random_id = random.randrange(0, self.pool_size)
                tmp = self.images[random_id].copy()
                self.images[random_id] = image.copy()
                return tmp
            else:
                return image


class CycleParser(object):
    def __init__(self, dataset_dir, flags):
        self.image_height, self.image_width = flags.image_height, flags.image_width
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_dir.split('/')[-1]

        self.images_trainA_dir = os.path.join(self.dataset_dir, 'trainA')
        self.images_trainB_dir = os.path.join(self.dataset_dir, 'trainB')
        self.images_valA_dir = os.path.join(self.dataset_dir, 'valA')
        self.images_valB_dir = os.path.join(self.dataset_dir, 'valB')
        self.TFRecord_dir = self.dataset_dir + '_TFRecord'

        self.images_trainA_paths, self.images_trainB_paths = None, None
        self.images_valA_paths, self.images_valB_paths = None, None
        self.images_testA_paths, self.images_testB_paths = None, None

        self.logs_dir = os.path.join(flags.logs_dir, 'events')
        self.checkpoint_dir = os.path.join(flags.logs_dir, 'models')
        self.logs_image_train_dir = os.path.join(flags.logs_dir, 'images_train')
        self.logs_image_val_dir = os.path.join(flags.logs_dir, 'images_val')
        self.dir_check()

    def dir_check(self):
        print('checking directories.')
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.logs_image_train_dir):
            os.makedirs(self.logs_image_train_dir)
        if not os.path.exists(self.logs_image_val_dir):
            os.makedirs(self.logs_image_val_dir)

    def load_paths(self, is_jpg=True, load_val=False):
        extension = '*.jpg' if is_jpg else '*.png'
        self.images_trainA_paths = sorted(glob(os.path.join(self.images_trainA_dir, extension)))
        self.images_trainB_paths = sorted(glob(os.path.join(self.images_trainB_dir, extension)))
        if load_val:
            self.images_valA_paths = sorted(glob(os.path.join(self.images_valA_dir, extension)))
            self.images_valB_paths = sorted(glob(os.path.join(self.images_valB_dir, extension)))
        return self

    def data2record(self, name, set_type, test_num=None):
        if not os.path.exists(self.TFRecord_dir):
            os.makedirs(self.TFRecord_dir)
        filename = os.path.join(self.TFRecord_dir, name)
        print('Writing', filename)
        batch_paths_a, batch_paths_b = [], []
        if set_type == 'train':
            batch_paths_a = self.images_trainA_paths
            batch_paths_b = self.images_trainB_paths
        elif set_type == 'val':
            batch_paths_a = self.images_valA_paths
            batch_paths_b = self.images_valB_paths
        elif set_type == 'test':
            batch_paths_a = self.images_testA_paths
            batch_paths_b = self.images_testB_paths

        # DataA
        writer = tf.python_io.TFRecordWriter(filename.replace('.tfrecords', 'A.tfrecords'))
        for idx, train_path in enumerate(batch_paths_a):
            print('[{:d}/{:d}]'.format(idx, len(batch_paths_a)))
            x = np.array(Image.open(train_path))
            # y = np.array(Image.open(train_path[1]))

            rows = x.shape[0]
            cols = x.shape[1]
            # Some images are gray-scale
            if len(x.shape) < 3:
                x = np.dstack((x, x, x))
            # Label [92, 183] -> [1, 92]
            # y[np.nonzero(y < 92)] = 183
            # y -= 91

            image_raw = x.tostring()
            # label_raw = y.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                # 'label_raw': _bytes_feature(label_raw),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())

            if test_num is not None and idx > test_num:
                break
        writer.close()

        # DataB
        writer = tf.python_io.TFRecordWriter(filename.replace('.tfrecords', 'B.tfrecords'))
        for idx, train_path in enumerate(batch_paths_b):
            print('[{:d}/{:d}]'.format(idx, len(batch_paths_b)))
            x = np.array(Image.open(train_path))
            # y = np.array(Image.open(train_path[1]))

            rows = x.shape[0]
            cols = x.shape[1]
            # Some images are gray-scale
            if len(x.shape) < 3:
                x = np.dstack((x, x, x))
            # Label [92, 183] -> [1, 92]
            # y[np.nonzero(y < 92)] = 183
            # y -= 91

            image_raw = x.tostring()
            # label_raw = y.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                # 'label_raw': _bytes_feature(label_raw),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())

            if test_num is not None and idx > test_num:
                break
        writer.close()

    def tfrecord_get_dataset(self, name, batch_size, need_augmentation=False, shuffle_size=None, need_flip=True):
        image_height, image_width = self.image_height, self.image_width

        def parse_record(record):
            features = tf.parse_single_example(
                record,
                features={
                    'height': tf.FixedLenFeature([], tf.int64),
                    'width': tf.FixedLenFeature([], tf.int64),
                    # 'label_raw': tf.FixedLenFeature([], tf.string),
                    'image_raw': tf.FixedLenFeature([], tf.string)
                })

            height = tf.cast(features['height'], tf.int32)
            width = tf.cast(features['width'], tf.int32)
            image = tf.decode_raw(features['image_raw'], tf.uint8)
            image = tf.reshape(image, [height, width, 3])
            # label = tf.decode_raw(features['label_raw'], tf.uint8)
            # label = tf.reshape(label, [height, width, 1])

            # augmentation:
            if need_augmentation:
                image = tf.cast(image, tf.float32)
                # label = tf.cast(label, tf.float32)
                image = tf.image.random_hue(image, max_delta=0.05)
                image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
                image = tf.image.random_brightness(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
                image = tf.minimum(image, 255.0)
                image = tf.maximum(image, 0.0)

            # combined = tf.concat((image, label), axis=2)
            combined = image
            #################################################################################################
            # random crop
            if need_augmentation:
                image_height_new = tf.maximum(height, image_height)
                image_width_new = tf.maximum(width, image_width)
                offset_height = (image_height_new - height) // 2
                offset_width = (image_width_new - width) // 2

                combined_pad = tf.image.pad_to_bounding_box(
                    combined, offset_height, offset_width,
                    image_height_new,
                    image_width_new)
                combined_crop = tf.random_crop(value=combined_pad, size=(image_height_new, image_width_new, 4))
            else:
                # combined_crop = tf.image.resize_image_with_crop_or_pad(combined, image_height, image_width)
                combined_crop = tf.image.resize_images(combined, (image_height, image_width, 3))
            #################################################################################################
            if need_flip:
                combined_crop = tf.image.random_flip_left_right(combined_crop)

            image = combined_crop
            # label = combined_crop[:, :, -1]
            image = self.preprocess_data(image=image)
            return image

        filename = os.path.join(self.TFRecord_dir, name)
        # filename1 = os.path.join(self.TFRecord_dir, 'coco_stuff2017_train_super.tfrecords')
        # filename2 = os.path.join(self.TFRecord_dir, 'coco_stuff2017_val_super.tfrecords')
        # dataset = tf.contrib.data.TFRecordDataset([filename1, filename2])
        dataset = tf.contrib.data.TFRecordDataset(filename)
        dataset = dataset.map(parse_record)
        if shuffle_size is not None:
            dataset = dataset.shuffle(buffer_size=shuffle_size)
        dataset = dataset.batch(batch_size)
        return dataset

    @staticmethod
    def preprocess_data(image):
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1.0
        return image

    @staticmethod
    def deprocess_data(image):
        image = (np.array(image) + 1.0) * 127.5
        return image

    def visualize_data(self, real_a, real_b, fake_a, fake_b, recon_a, recon_b, shape, global_step, logs_dir):
        real_a = merge(self.deprocess_data(image=real_a), size=shape)
        real_b = merge(self.deprocess_data(image=real_b), size=shape)
        fake_a = merge(self.deprocess_data(image=fake_a), size=shape)
        fake_b = merge(self.deprocess_data(image=fake_b), size=shape)
        recon_a = merge(self.deprocess_data(image=recon_a), size=shape)
        recon_b = merge(self.deprocess_data(image=recon_b), size=shape)

        x_png = Image.fromarray(real_a.astype(np.uint8)).convert('RGB')
        x_png.save('{}/{:d}_0_realA.png'.format(
            logs_dir, global_step), format='PNG')
        x_png = Image.fromarray(fake_a.astype(np.uint8)).convert('RGB')
        x_png.save('{}/{:d}_4_fakeA.png'.format(
            logs_dir, global_step), format='PNG')
        x_png = Image.fromarray(recon_a.astype(np.uint8)).convert('RGB')
        x_png.save('{}/{:d}_2_reconA.png'.format(
            logs_dir, global_step), format='PNG')

        x_png = Image.fromarray(real_b.astype(np.uint8)).convert('RGB')
        x_png.save('{}/{:d}_3_realB.png'.format(
            logs_dir, global_step), format='PNG')
        x_png = Image.fromarray(fake_b.astype(np.uint8)).convert('RGB')
        x_png.save('{}/{:d}_1_fakeB.png'.format(
            logs_dir, global_step), format='PNG')
        x_png = Image.fromarray(recon_b.astype(np.uint8)).convert('RGB')
        x_png.save('{}/{:d}_5_reconB.png'.format(
            logs_dir, global_step), format='PNG')

    def load_data(self, start, end, set_type):
        batch_paths_a, batch_paths_b = [], []
        batch_a, batch_b = [], []
        if set_type == 'train':
            batch_paths_a = self.images_trainA_paths[start:end]
            batch_paths_b = self.images_trainB_paths[start:end]
        elif set_type == 'val':
            batch_paths_a = self.images_valA_paths[start:end]
            batch_paths_b = self.images_valB_paths[start:end]
        elif set_type == 'test':
            batch_paths_a = self.images_testA_paths[start:end]
            batch_paths_b = self.images_testB_paths[start:end]

        for idx in range(end - start):
            print(idx)
            x = Image.open(batch_paths_a[idx])
            y = Image.open(batch_paths_b[idx])
            x = np.array(x.resize((self.image_height, self.image_width), resample=Image.BILINEAR))
            y = np.array(y.resize((self.image_height, self.image_width), resample=Image.BILINEAR))
            batch_a.append(x)
            batch_b.append(y)

        x_png = Image.fromarray(batch_a[0].astype(np.uint8)).convert('RGB')
        x_png.save('./{}_{:d}_{:d}_A.png'.format(
            set_type, start, end), format='PNG')
        x_png = Image.fromarray(batch_b[0].astype(np.uint8)).convert('RGB')
        x_png.save('./{}_{:d}_{:d}_B.png'.format(
            set_type, start, end), format='PNG')

        return batch_a, batch_b


class GANParser(object):
    def __init__(self, flags):
        self.image_height, self.image_width = flags.image_height, flags.image_width
        self.dataset_dir = flags.dataset_dir
        self.dataset_name = flags.dataset_dir.split('/')[-1]

        self.images_trainA_dir = os.path.join(self.dataset_dir, 'trainA')
        self.images_trainB_dir = os.path.join(self.dataset_dir, 'trainB')
        self.images_valA_dir = os.path.join(self.dataset_dir, 'valA')
        self.images_valB_dir = os.path.join(self.dataset_dir, 'valB')
        self.TFRecord_dir = self.dataset_dir + '_TFRecord'

        self.images_trainA_paths, self.images_trainB_paths = None, None
        self.images_valA_paths, self.images_valB_paths = None, None

        self.logs_dir = os.path.join(flags.logs_dir, 'events')
        self.checkpoint_dir = os.path.join(flags.logs_dir, 'models')
        self.logs_image_train_dir = os.path.join(flags.logs_dir, 'images_train')
        self.logs_image_val_dir = os.path.join(flags.logs_dir, 'images_val')
        self.logs_mat_output_dir = os.path.join(flags.logs_dir, 'mat_output')
        self.dir_check()

    def dir_check(self):
        print('checking directories.')
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.logs_image_train_dir):
            os.makedirs(self.logs_image_train_dir)
        if not os.path.exists(self.logs_image_val_dir):
            os.makedirs(self.logs_image_val_dir)
        if not os.path.exists(self.logs_mat_output_dir):
            os.makedirs(self.logs_mat_output_dir)

    def load_paths(self, is_jpg=True, load_val=False):
        extension = '*.jpg' if is_jpg else '*.png'
        self.images_trainA_paths = sorted(glob(os.path.join(self.images_trainA_dir, extension)))
        self.images_trainB_paths = sorted(glob(os.path.join(self.images_trainB_dir, extension)))
        if load_val:
            self.images_valA_paths = sorted(glob(os.path.join(self.images_valA_dir, extension)))
            self.images_valB_paths = sorted(glob(os.path.join(self.images_valB_dir, extension)))
        return self

    def data2record(self, name, set_type, test_num=None):
        if not os.path.exists(self.TFRecord_dir):
            os.makedirs(self.TFRecord_dir)
        filename = os.path.join(self.TFRecord_dir, name)
        print('Writing', filename)
        batch_paths_a, batch_paths_b = [], []
        if set_type == 'train':
            batch_paths_a = self.images_trainA_paths
            batch_paths_b = self.images_trainB_paths
        elif set_type == 'val':
            batch_paths_a = self.images_valA_paths
            batch_paths_b = self.images_valB_paths

        # DataA
        writer = tf.python_io.TFRecordWriter(filename.replace('.tfrecords', 'A.tfrecords'))
        for idx, train_path in enumerate(batch_paths_a):
            print('[{:d}/{:d}]'.format(idx, len(batch_paths_a)))
            img_name = train_path.split('/')[-1].split('.')[0].encode()
            x = np.array(Image.open(train_path))

            rows = x.shape[0]
            cols = x.shape[1]
            # Some images are gray-scale
            if len(x.shape) < 3:
                x = np.dstack((x, x, x))

            image_raw = x.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'image_name': _bytes_feature(img_name),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())

            if test_num is not None and idx > test_num:
                break
        writer.close()

        # DataB
        writer = tf.python_io.TFRecordWriter(filename.replace('.tfrecords', 'B.tfrecords'))
        for idx, train_path in enumerate(batch_paths_b):
            print('[{:d}/{:d}]'.format(idx, len(batch_paths_b)))
            img_name = train_path.split('/')[-1].split('.')[0].encode()
            x = np.array(Image.open(train_path))

            rows = x.shape[0]
            cols = x.shape[1]
            # Some images are gray-scale
            if len(x.shape) < 3:
                x = np.dstack((x, x, x))

            image_raw = x.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'image_name': _bytes_feature(img_name),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())

            if test_num is not None and idx > test_num:
                break
        writer.close()

    def tfrecord_get_dataset(self, name, batch_size, need_augmentation=False, shuffle_size=None, need_flip=True):
        image_height, image_width = self.image_height, self.image_width

        def parse_record(record):
            features = tf.parse_single_example(
                record,
                features={
                    'height': tf.FixedLenFeature([], tf.int64),
                    'width': tf.FixedLenFeature([], tf.int64),
                    'image_raw': tf.FixedLenFeature([], tf.string),
                    'image_name': tf.FixedLenFeature([], tf.string)
                })

            height = tf.cast(features['height'], tf.int32)
            width = tf.cast(features['width'], tf.int32)
            image = tf.decode_raw(features['image_raw'], tf.uint8)
            image = tf.reshape(image, [height, width, 3])
            # min_edge = tf.minimum(height, width)
            # label = tf.decode_raw(features['label_raw'], tf.uint8)
            # label = tf.reshape(label, [height, width, 1])

            # augmentation:
            if need_augmentation:
                image = tf.cast(image, tf.float32)
                # label = tf.cast(label, tf.float32)
                image = tf.image.random_hue(image, max_delta=0.05)
                image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
                image = tf.image.random_brightness(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
                image = tf.minimum(image, 255.0)
                image = tf.maximum(image, 0.0)

            # combined = tf.concat((image, label), axis=2)
            combined = image
            #################################################################################################
            # random crop
            if need_augmentation:
                image_height_new = tf.maximum(height, image_height)
                image_width_new = tf.maximum(width, image_width)
                offset_height = (image_height_new - height) // 2
                offset_width = (image_width_new - width) // 2

                combined_pad = tf.image.pad_to_bounding_box(
                    combined, offset_height, offset_width,
                    image_height_new,
                    image_width_new)
                combined_crop = tf.random_crop(value=combined_pad, size=(image_height_new, image_width_new, 4))
            else:
                # combined_crop = tf.image.resize_image_with_crop_or_pad(combined, image_height, image_width)
                combined_crop = tf.image.resize_images(combined, (image_height, image_width))

                '''
                image_height_new = tf.maximum(height, image_height)
                image_width_new = tf.maximum(width, image_width)
                offset_height = (image_height_new - height) // 2
                offset_width = (image_width_new - width) // 2

                combined_pad = tf.image.pad_to_bounding_box(
                    combined, offset_height, offset_width,
                    image_height_new,
                    image_width_new)
                combined_crop = tf.image.resize_images(combined_pad, (image_height, image_width))
                '''

            #################################################################################################
            if need_flip:
                combined_crop = tf.image.random_flip_left_right(combined_crop)

            image = combined_crop
            # label = combined_crop[:, :, -1]
            image = self.preprocess_data(image=image)
            return image, features['image_name'], [height, width]

        filename = os.path.join(self.TFRecord_dir, name)
        # filename1 = os.path.join(self.TFRecord_dir, 'coco_stuff2017_train_super.tfrecords')
        # filename2 = os.path.join(self.TFRecord_dir, 'coco_stuff2017_val_super.tfrecords')
        # dataset = tf.contrib.data.TFRecordDataset([filename1, filename2])
        dataset = tf.contrib.data.TFRecordDataset(filename)
        dataset = dataset.map(parse_record)
        if shuffle_size is not None:
            dataset = dataset.shuffle(buffer_size=shuffle_size)
        dataset = dataset.batch(batch_size)
        return dataset

    @staticmethod
    def preprocess_data(image):
        image = tf.cast(image, tf.float32)
        # image = (image / 127.5) - 1.0
        return image

    @staticmethod
    def deprocess_data(image):
        # image = (np.array(image) + 1.0) * 127.5
        return image

    def visualize_data(self, real_a, real_b, adjusted_a, segment_a, fake_b, shape, global_step, logs_dir,
                       real_a_name, real_b_name):
        real_a = merge(self.deprocess_data(image=real_a), size=shape)
        real_b = merge(self.deprocess_data(image=real_b), size=shape)
        adjusted_a = merge(self.deprocess_data(image=adjusted_a), size=shape)

        # TODO: only support batch_size=1
        binary_a = np.zeros_like(segment_a, dtype=np.uint8)
        binary_a[segment_a > np.mean(segment_a)] = 255
        binary_a = merge(binary_a, size=shape)

        segment_a = merge(np.array(segment_a) * 255., size=shape)
        fake_b = merge(self.deprocess_data(image=fake_b), size=shape)

        x_png = Image.fromarray(real_a.astype(np.uint8)).convert('RGB')
        x_png.save('{}/{:d}_0_{}_realA.png'.format(
            logs_dir, global_step, real_a_name), format='PNG')
        x_png = Image.fromarray(segment_a.astype(np.uint8)).convert('RGB')
        x_png.save('{}/{:d}_1_{}_segment_A.png'.format(
            logs_dir, global_step, real_a_name), format='PNG')
        x_png = Image.fromarray(binary_a.astype(np.uint8)).convert('RGB')
        x_png.save('{}/{:d}_2_{}_binary_A.png'.format(
            logs_dir, global_step, real_a_name), format='PNG')
        x_png = Image.fromarray(adjusted_a.astype(np.uint8)).convert('RGB')
        x_png.save('{}/{:d}_3_{}_adjustedA.png'.format(
            logs_dir, global_step, real_a_name), format='PNG')
        x_png = Image.fromarray(fake_b.astype(np.uint8)).convert('RGB')
        x_png.save('{}/{:d}_4_{}_fakeB.png'.format(
            logs_dir, global_step, real_a_name), format='PNG')

        x_png = Image.fromarray(real_b.astype(np.uint8)).convert('RGB')
        x_png.save('{}/{:d}_5_{}_realB.png'.format(
            logs_dir, global_step, real_b_name), format='PNG')

    def visualize_data_zero(self, real_a, real_b, adjusted_a, segment_a, fake_b,
                            segment_a_0, segment_a_1, segment_a_2, shape, global_step, logs_dir):
        real_a = merge(self.deprocess_data(image=real_a), size=shape)
        real_b = merge(self.deprocess_data(image=real_b), size=shape)
        # adjusted_a = merge(self.deprocess_data(image=adjusted_a), size=shape)
        # segment_a = merge(np.array(segment_a) * 255., size=shape)
        segment_a_0 = merge(np.array(segment_a_0) * 255., size=shape)
        segment_a_1 = merge(np.array(segment_a_1) * 255., size=shape)
        segment_a_2 = merge(np.array(segment_a_2) * 255., size=shape)
        # fake_b = merge(self.deprocess_data(image=fake_b), size=shape)

        x_png = Image.fromarray(real_a.astype(np.uint8)).convert('RGB')
        x_png.save('{}/{:d}_0_realA.png'.format(
            logs_dir, global_step), format='PNG')

        x_png = Image.fromarray(segment_a_0.astype(np.uint8)).convert('RGB')
        x_png.save('{}/{:d}_1_3_segment0_A.png'.format(
            logs_dir, global_step), format='PNG')
        x_png = Image.fromarray(segment_a_1.astype(np.uint8)).convert('RGB')
        x_png.save('{}/{:d}_1_2_segment1_A.png'.format(
            logs_dir, global_step), format='PNG')
        x_png = Image.fromarray(segment_a_2.astype(np.uint8)).convert('RGB')
        x_png.save('{}/{:d}_1_1_segment2_A.png'.format(
            logs_dir, global_step), format='PNG')

        '''
        x_png = Image.fromarray(segment_a.astype(np.uint8)).convert('RGB')
        x_png.save('{}/{:d}_1_4_segment_A.png'.format(
            logs_dir, global_step), format='PNG')
        x_png = Image.fromarray(adjusted_a.astype(np.uint8)).convert('RGB')
        x_png.save('{}/{:d}_2_adjustedA.png'.format(
            logs_dir, global_step), format='PNG')
        x_png = Image.fromarray(fake_b.astype(np.uint8)).convert('RGB')
        x_png.save('{}/{:d}_3_fakeB.png'.format(
            logs_dir, global_step), format='PNG')
        '''

        x_png = Image.fromarray(real_b.astype(np.uint8)).convert('RGB')
        x_png.save('{}/{:d}_4_realB.png'.format(
            logs_dir, global_step), format='PNG')

    def load_data(self, start, end, set_type):
        batch_paths_a, batch_paths_b = [], []
        batch_a, batch_b = [], []
        if set_type == 'train':
            batch_paths_a = self.images_trainA_paths[start:end]
            batch_paths_b = self.images_trainB_paths[start:end]
        elif set_type == 'val':
            batch_paths_a = self.images_valA_paths[start:end]
            batch_paths_b = self.images_valB_paths[start:end]

        for idx in range(end - start):
            print(idx)
            x = Image.open(batch_paths_a[idx])
            y = Image.open(batch_paths_b[idx])
            x = np.array(x.resize((self.image_height, self.image_width), resample=Image.BILINEAR))
            y = np.array(y.resize((self.image_height, self.image_width), resample=Image.BILINEAR))
            batch_a.append(x)
            batch_b.append(y)

        x_png = Image.fromarray(batch_a[0].astype(np.uint8)).convert('RGB')
        x_png.save('./{}_{:d}_{:d}_A.png'.format(
            set_type, start, end), format='PNG')
        x_png = Image.fromarray(batch_b[0].astype(np.uint8)).convert('RGB')
        x_png.save('./{}_{:d}_{:d}_B.png'.format(
            set_type, start, end), format='PNG')

        return batch_a, batch_b


class SemanticParser(object):
    def __init__(self, flags):
        self.image_height, self.image_width = flags.image_height, flags.image_width
        self.dataset_dir = flags.dataset_dir
        self.dataset_name = flags.dataset_dir.split('/')[-1]

        self.images_trainA_dir = os.path.join(self.dataset_dir, 'trainA')
        self.images_trainB_dir = os.path.join(self.dataset_dir, 'trainB')
        self.images_valA_dir = os.path.join(self.dataset_dir, 'valA')
        self.images_valB_dir = os.path.join(self.dataset_dir, 'valB')
        self.TFRecord_dir = self.dataset_dir + '_TFRecord'

        self.images_trainA_paths, self.images_trainB_paths = None, None
        self.images_valA_paths, self.images_valB_paths = None, None

        self.logs_dir = os.path.join(flags.logs_dir, 'events')
        self.checkpoint_dir = os.path.join(flags.logs_dir, 'models')
        self.logs_image_train_dir = os.path.join(flags.logs_dir, 'images_train')
        self.logs_image_val_dir = os.path.join(flags.logs_dir, 'images_val')
        self.logs_mat_output_dir = os.path.join(flags.logs_dir, 'mat_output')
        self.dir_check()

    def dir_check(self):
        print('checking directories.')
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.logs_image_train_dir):
            os.makedirs(self.logs_image_train_dir)
        if not os.path.exists(self.logs_image_val_dir):
            os.makedirs(self.logs_image_val_dir)
        if not os.path.exists(self.logs_mat_output_dir):
            os.makedirs(self.logs_mat_output_dir)

    def load_paths(self, is_jpg=True, load_val=False):
        extension = '*.jpg' if is_jpg else '*.png'
        self.images_trainA_paths = sorted(glob(os.path.join(self.images_trainA_dir, extension)))
        self.images_trainB_paths = sorted(glob(os.path.join(self.images_trainB_dir, extension)))
        if load_val:
            self.images_valA_paths = sorted(glob(os.path.join(self.images_valA_dir, extension)))
            self.images_valB_paths = sorted(glob(os.path.join(self.images_valB_dir, extension)))
        return self

    def data2record(self, name, set_type, test_num=None):
        if not os.path.exists(self.TFRecord_dir):
            os.makedirs(self.TFRecord_dir)
        filename = os.path.join(self.TFRecord_dir, name)
        print('Writing', filename)
        batch_paths_a, batch_paths_b = [], []
        if set_type == 'train':
            batch_paths_a = self.images_trainA_paths
            batch_paths_b = self.images_trainB_paths
        elif set_type == 'val':
            batch_paths_a = self.images_valA_paths
            batch_paths_b = self.images_valB_paths

        # DataA
        writer = tf.python_io.TFRecordWriter(filename.replace('.tfrecords', 'A.tfrecords'))
        for idx, train_path in enumerate(batch_paths_a):
            print('[{:d}/{:d}]'.format(idx, len(batch_paths_a)))
            img_name = train_path.split('/')[-1].split('.')[0].encode()
            x = np.array(Image.open(train_path))

            rows = x.shape[0]
            cols = x.shape[1]
            # Some images are gray-scale
            if len(x.shape) < 3:
                x = np.dstack((x, x, x))

            image_raw = x.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'image_name': _bytes_feature(img_name),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())

            if test_num is not None and idx > test_num:
                break
        writer.close()

        # DataB
        writer = tf.python_io.TFRecordWriter(filename.replace('.tfrecords', 'B.tfrecords'))
        for idx, train_path in enumerate(batch_paths_b):
            print('[{:d}/{:d}]'.format(idx, len(batch_paths_b)))
            img_name = train_path.split('/')[-1].split('.')[0].encode()
            # x = np.array(Image.open(train_path))[:, :, 0]
            x = np.array(Image.open(train_path))

            rows = x.shape[0]
            cols = x.shape[1]
            # All labels are gray-scale

            image_raw = x.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'image_name': _bytes_feature(img_name),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())

            if test_num is not None and idx > test_num:
                break
        writer.close()

    def tfrecord_get_dataset(self, name, batch_size, need_augmentation=False,
                             shuffle_size=None, is_label=False, need_flip=True):
        image_height, image_width = self.image_height, self.image_width

        def parse_record(record):
            features = tf.parse_single_example(
                record,
                features={
                    'height': tf.FixedLenFeature([], tf.int64),
                    'width': tf.FixedLenFeature([], tf.int64),
                    'image_raw': tf.FixedLenFeature([], tf.string),
                    'image_name': tf.FixedLenFeature([], tf.string)
                })

            height = tf.cast(features['height'], tf.int32)
            width = tf.cast(features['width'], tf.int32)
            image = tf.decode_raw(features['image_raw'], tf.uint8)
            if is_label:
                image = tf.reshape(image, [height, width, 1])
            else:
                image = tf.reshape(image, [height, width, 3])
            # min_edge = tf.minimum(height, width)
            # label = tf.decode_raw(features['label_raw'], tf.uint8)
            # label = tf.reshape(label, [height, width, 1])

            # augmentation:
            if need_augmentation:
                image = tf.cast(image, tf.float32)
                # label = tf.cast(label, tf.float32)
                image = tf.image.random_hue(image, max_delta=0.05)
                image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
                image = tf.image.random_brightness(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
                image = tf.minimum(image, 255.0)
                image = tf.maximum(image, 0.0)

            # combined = tf.concat((image, label), axis=2)
            combined = image
            #################################################################################################
            # random crop
            if need_augmentation:
                image_height_new = tf.maximum(height, image_height)
                image_width_new = tf.maximum(width, image_width)
                offset_height = (image_height_new - height) // 2
                offset_width = (image_width_new - width) // 2

                combined_pad = tf.image.pad_to_bounding_box(
                    combined, offset_height, offset_width,
                    image_height_new,
                    image_width_new)
                combined_crop = tf.random_crop(value=combined_pad, size=(image_height_new, image_width_new, 4))
            else:
                # combined_crop = tf.image.resize_image_with_crop_or_pad(combined, image_height, image_width)
                combined_crop = tf.image.resize_images(combined, (image_height, image_width))

            #################################################################################################
            if need_flip:
                combined_crop = tf.image.random_flip_left_right(combined_crop)

            image = combined_crop
            image = self.preprocess_data(image=image)
            return image, features['image_name'], [height, width]

        filename = os.path.join(self.TFRecord_dir, name)
        dataset = tf.contrib.data.TFRecordDataset(filename)
        dataset = dataset.map(parse_record)
        if shuffle_size is not None:
            dataset = dataset.shuffle(buffer_size=shuffle_size)
        dataset = dataset.batch(batch_size)
        return dataset

    @staticmethod
    def preprocess_data(image):
        image = tf.cast(image, tf.float32)
        # image = (image / 127.5) - 1.0
        return image

    @staticmethod
    def deprocess_data(image):
        # image = (np.array(image) + 1.0) * 127.5
        return image

    def visualize_data(self, real_a, real_b, segment_a, shape, global_step, logs_dir, real_a_name):
        real_a = merge(self.deprocess_data(image=real_a), size=shape)
        real_b = merge(self.deprocess_data(image=real_b), size=shape)
        segment_a = merge(np.array(segment_a) * 255., size=shape)

        x_png = Image.fromarray(real_a.astype(np.uint8)).convert('RGB')
        x_png.save('{}/{:d}_0_{}_realA.png'.format(
            logs_dir, global_step, real_a_name), format='PNG')
        x_png = Image.fromarray(segment_a.astype(np.uint8)).convert('RGB')
        x_png.save('{}/{:d}_1_{}_segment_A.png'.format(
            logs_dir, global_step, real_a_name), format='PNG')

        x_png = Image.fromarray(real_b.astype(np.uint8)).convert('RGB')
        x_png.save('{}/{:d}_4_{}_realB.png'.format(
            logs_dir, global_step, real_a_name), format='PNG')


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if images.shape[3] in (3, 4):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
