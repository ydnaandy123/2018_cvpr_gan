from ops import *
import numpy as np


def train_sgd_op(loss, learning_rate, flags, var_list, name):
    with tf.variable_scope(name):
        optimizer = tf.train.AdamOptimizer(learning_rate, flags.beta1, flags.beta2, name='optimizer')
        grads = optimizer.compute_gradients(loss, var_list=var_list)
        '''
        if flags.debug:
            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + "/gradient", grad)
        '''
        return optimizer.apply_gradients(grads, name='train_op')


def mae_criterion(in_, target, name):
    return tf.reduce_mean((in_ - target) ** 2, name=name)


def sce_criterion(logits, labels, name):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels), name=name)


def generator_unet(image, flags, reuse, drop_probability, name):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # image is (256 x 256 x input_c_dim)
        e1 = instance_normalization(conv2d(image, flags.gf_dim, name='g_e1_conv'))
        # e1 is (128 x 128 x self.gf_dim)
        e2 = instance_normalization(conv2d(lrelu(e1), flags.gf_dim*2, name='g_e2_conv'), 'g_bn_e2')
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = instance_normalization(conv2d(lrelu(e2), flags.gf_dim*4, name='g_e3_conv'), 'g_bn_e3')
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = instance_normalization(conv2d(lrelu(e3), flags.gf_dim*8, name='g_e4_conv'), 'g_bn_e4')
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = instance_normalization(conv2d(lrelu(e4), flags.gf_dim*8, name='g_e5_conv'), 'g_bn_e5')
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = instance_normalization(conv2d(lrelu(e5), flags.gf_dim*8, name='g_e6_conv'), 'g_bn_e6')
        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = instance_normalization(conv2d(lrelu(e6), flags.gf_dim*8, name='g_e7_conv'), 'g_bn_e7')
        # e7 is (2 x 2 x self.gf_dim*8)
        e8 = instance_normalization(conv2d(lrelu(e7), flags.gf_dim*8, name='g_e8_conv'), 'g_bn_e8')
        # e8 is (1 x 1 x self.gf_dim*8)

        d1 = deconv2d(tf.nn.relu(e8), flags.gf_dim*8, name='g_d1')
        d1 = tf.nn.dropout(d1, drop_probability)
        d1 = tf.concat([instance_normalization(d1, 'g_bn_d1'), e7], 3)
        # d1 is (2 x 2 x self.gf_dim*8*2)

        d2 = deconv2d(tf.nn.relu(d1), flags.gf_dim*8, name='g_d2')
        d2 = tf.nn.dropout(d2, drop_probability)
        d2 = tf.concat([instance_normalization(d2, 'g_bn_d2'), e6], 3)
        # d2 is (4 x 4 x self.gf_dim*8*2)

        d3 = deconv2d(tf.nn.relu(d2), flags.gf_dim*8, name='g_d3')
        d3 = tf.nn.dropout(d3, drop_probability)
        d3 = tf.concat([instance_normalization(d3, 'g_bn_d3'), e5], 3)
        # d3 is (8 x 8 x self.gf_dim*8*2)

        d4 = deconv2d(tf.nn.relu(d3), flags.gf_dim*8, name='g_d4')
        d4 = tf.concat([instance_normalization(d4, 'g_bn_d4'), e4], 3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        d5 = deconv2d(tf.nn.relu(d4), flags.gf_dim*4, name='g_d5')
        d5 = tf.concat([instance_normalization(d5, 'g_bn_d5'), e3], 3)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        d6 = deconv2d(tf.nn.relu(d5), flags.gf_dim*2, name='g_d6')
        d6 = tf.concat([instance_normalization(d6, 'g_bn_d6'), e2], 3)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        d7 = deconv2d(tf.nn.relu(d6), flags.gf_dim, name='g_d7')
        d7 = tf.concat([instance_normalization(d7, 'g_bn_d7'), e1], 3)
        # d7 is (128 x 128 x self.gf_dim*1*2)

        d8 = deconv2d(tf.nn.relu(d7), flags.c_dim, name='g_d8')
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(d8)


def discriminator(image, flags, reuse, name):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(image, flags.df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(instance_normalization(conv2d(h0, flags.df_dim * 2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_normalization(conv2d(h1, flags.df_dim * 4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_normalization(conv2d(h2, flags.df_dim * 8, s=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
        # h4 is (32 x 32 x 1)
        return h4


def gaussian_blur(image, name):
    with tf.variable_scope(name):
        # weights = [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]]
        # weights = np.array([[1, 4, 7, 4, 1],
        #                     [4, 16, 26, 16, 4],
        #                     [7, 26, 41, 26, 7],
        #                     [4, 16, 26, 16, 4],
        #                     [1, 4, 7, 4, 1]], dtype=np.float32) / 273.
        weights = np.array([[2, 4, 5, 4, 2],
                            [4, 9, 12, 9, 4],
                            [5, 12, 15, 12, 5],
                            [4, 9, 12, 9, 4],
                            [2, 4, 5, 4, 2]], dtype=np.float32) / 159.
        init = tf.constant_initializer(weights, dtype=tf.float32)

        image_r, image_g, image_b = tf.unstack(image, axis=3)
        image_r = expand_conv(image_r, init=init, k=5, name='image_r')
        image_g = expand_conv(image_g, init=init, k=5, name='image_g')
        image_b = expand_conv(image_b, init=init, k=5, name='image_b')

        image = tf.concat([image_r, image_g, image_b], axis=3)
        return image


def high_light(image, name):
    with tf.variable_scope(name):
        image_r, image_g, image_b = tf.unstack(image, axis=3)
        image_r = tf.multiply(image_r, 0.2989, name='image_r')
        image_g = tf.multiply(image_g, 0.5870, name='image_g')
        image_b = tf.multiply(image_b, 0.1140, name='image_b')
        adjusted_image = tf.expand_dims(image_r + image_g + image_b, axis=3, name='adjusted_image')
        return adjusted_image


def generator_resnet_se(image, options, reuse, name):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        image = (image / 127.5) - 1.0

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_normalization(conv2d(c0, options.gf_dim,
                                                      7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_normalization(conv2d(c1, options.gf_dim*2,
                                                      3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_normalization(conv2d(c2, options.gf_dim*4,
                                                      3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block_se(c3, options.gf_dim*4, name='g_r1')
        r2 = residule_block_se(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block_se(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block_se(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block_se(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block_se(r5, options.gf_dim*4, name='g_r6')
        r7 = residule_block_se(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block_se(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block_se(r8, options.gf_dim*4, name='g_r9')

        d1 = deconv2d(r9, options.gf_dim*2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_normalization(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_normalization(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        # pred = tf.nn.tanh(conv2d(d2, options.c_out_dim, 7, 1, padding='VALID', name='g_pred_c'))
        pred = tf.nn.sigmoid(conv2d(d2, options.c_out_dim, 7, 1, padding='VALID', name='g_pred_c'))

        return pred


def discriminator_se(image, flags, reuse, name):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        image = (image / 127.5) - 1.0
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(image, flags.df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(instance_normalization(conv2d(h0, flags.df_dim * 2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_normalization(conv2d(h1, flags.df_dim * 4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_normalization(conv2d(h2, flags.df_dim * 8, s=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
        # h4 is (32 x 32 x 1)
        return h4


def discriminator_patch142(image, flags, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        # image is 256 x 256 x input_c_dim
        image_normalization = image_normalization_one(image, name='image_normalization_one')

        # PatchGAN n=4, https://github.com/phillipi/pix2pix/blob/master/models.lua#L131
        # Receptive field=(14x142), https://github.com/phillipi/pix2pix/blob/master/scripts/receptive_field_sizes.m
        # 128x128,
        n1 = conv2d(image_normalization, flags.df_dim, ks=4, s=2, padding='SAME', name='n1')
        # 64x64,
        n2 = lrelu_instance_normalization_conv2d(n1, flags.df_dim * 2, ks=4, s=2, padding='SAME', name='n2')
        # 32x32,
        n3 = lrelu_instance_normalization_conv2d(n2, flags.df_dim * 4, ks=4, s=2, padding='SAME', name='n3')
        # 16x16,
        n4 = lrelu_instance_normalization_conv2d(n3, flags.df_dim * 8, ks=4, s=2, padding='SAME', name='n4')

        # 16x16
        s1 = lrelu_instance_normalization_conv2d(n4, flags.df_dim * 16, ks=4, s=1, padding='SAME', name='s1')
        # 16x16
        s2 = lrelu_instance_normalization_conv2d(s1, flags.df_dim * 16, ks=4, s=1, padding='SAME', name='s2')
        # 16x16
        s3 = lrelu_instance_normalization_conv2d(s2, 1, ks=4, s=1, padding='SAME', name='s3')

        return s3


def discriminator_patch_all(image, flags, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        # image is 256 x 256 x input_c_dim
        image_normalization = image_normalization_one(image, name='image_normalization_one')

        # PatchGAN n=4, https://github.com/phillipi/pix2pix/blob/master/models.lua#L131
        # Receptive field=(14x142), https://github.com/phillipi/pix2pix/blob/master/scripts/receptive_field_sizes.m
        # 128x128,
        n1 = conv2d(image_normalization, flags.df_dim, ks=4, s=2, padding='SAME', name='n1')
        # 64x64,
        n2 = lrelu_instance_normalization_conv2d(n1, flags.df_dim * 2, ks=4, s=2, padding='SAME', name='n2')
        # 32x32,
        n3 = lrelu_instance_normalization_conv2d(n2, flags.df_dim * 4, ks=4, s=2, padding='SAME', name='n3')
        # 16x16,
        n4 = lrelu_instance_normalization_conv2d(n3, flags.df_dim * 8, ks=4, s=2, padding='SAME', name='n4')
        # 8x8
        n5 = lrelu_instance_normalization_conv2d(n4, flags.df_dim * 16, ks=4, s=2, padding='SAME', name='n5')
        # 1x1
        s1 = lrelu_instance_normalization_conv2d(n5, 1, ks=8, s=1, padding='VALID', name='s1')

        return s1


def generator_resnet_se_fixed(image, flags, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        # image is 256 x 256 x input_c_dim
        image_normalization = image_normalization_one(image, name='image_normalization_one')

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        image_pad = tf.pad(image_normalization, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = conv2d(image_pad, flags.gf_dim, ks=7, s=1, padding='VALID', name='c1')
        # 256x256
        c2 = relu_instance_normalization_conv2d(c1, flags.gf_dim * 2, ks=3, s=2, padding='SAME', name='c2')
        # 128x128
        c3 = relu_instance_normalization_conv2d(c2, flags.gf_dim * 4, ks=3, s=2, padding='SAME', name='c3')
        # 64x64

        # define G network with 9 resnet blocks
        r1 = residule_block_se_fixed(c3, flags.gf_dim * 4, name='r1')
        r2 = residule_block_se_fixed(r1, flags.gf_dim * 4, name='r2')
        r3 = residule_block_se_fixed(r2, flags.gf_dim * 4, name='r3')
        r4 = residule_block_se_fixed(r3, flags.gf_dim * 4, name='r4')
        r5 = residule_block_se_fixed(r4, flags.gf_dim * 4, name='r5')
        r6 = residule_block_se_fixed(r5, flags.gf_dim * 4, name='r6')
        r7 = residule_block_se_fixed(r6, flags.gf_dim * 4, name='r7')
        r8 = residule_block_se_fixed(r7, flags.gf_dim * 4, name='r8')
        r9 = residule_block_se_fixed(r8, flags.gf_dim * 4, name='r9')

        d2 = relu_instance_normalization_deconv2d(r9, flags.gf_dim * 2, 3, 2, name='d2')
        # 128x128
        c2_skip_scale = skip_squeeze_excitation_layer(c2, dim=flags.gf_dim * 2, name='c2_skip_scale')
        d2_fuse = tf.add(d2, c2_skip_scale, name='d2_fuse')

        d1 = relu_instance_normalization_deconv2d(d2_fuse, flags.gf_dim * 2, 3, 2, name='d1')
        # 256x256
        c1_skip_scale = skip_squeeze_excitation_layer(c1, dim=flags.gf_dim * 2, name='c1_skip_scale')
        d1_fuse = tf.add(d1, c1_skip_scale, name='d1_fuse')
        d1_pad = tf.pad(d1_fuse, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

        # Pred
        # pred = tf.nn.tanh(conv2d(d2, options.c_out_dim, 7, 1, padding='VALID', name='g_pred_c'))
        pred = relu_instance_normalization_conv2d(d1_pad, flags.c_out_dim, 7, 1, padding='VALID', name='pred')
        pred_sigmoid = tf.nn.sigmoid(pred)

        return pred_sigmoid


def discriminator_wgangp(image, flags, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        # image is 256 x 256 x input_c_dim
        image_reshape = tf.reshape(image, [-1, flags.image_height, flags.image_width, flags.c_in_dim])
        image_normalization = image_normalization_one(image_reshape, name='image_normalization_one')

        # PatchGAN n=4, https://github.com/phillipi/pix2pix/blob/master/models.lua#L131
        # Receptive field=(14x142), https://github.com/phillipi/pix2pix/blob/master/scripts/receptive_field_sizes.m
        # 128x128,
        n1 = conv2d(image_normalization, flags.df_dim, ks=4, s=2, padding='SAME', name='n1')
        # 64x64,
        n2 = lrelu_conv2d(n1, flags.df_dim * 2, ks=4, s=2, padding='SAME', name='n2')
        # 32x32,
        n3 = instance_normalization_lrelu_conv2d(n2, flags.df_dim * 4, ks=4, s=2, padding='SAME', name='n3')
        # 16x16,
        n4 = instance_normalization_lrelu_conv2d(n3, flags.df_dim * 8, ks=4, s=2, padding='SAME', name='n4')
        # 8x8
        n5 = instance_normalization_lrelu_conv2d(n4, flags.df_dim * 16, ks=4, s=2, padding='SAME', name='n5')
        # 1x1
        s1 = instance_normalization_lrelu_conv2d(n5, 1, ks=7, s=1, padding='VALID', name='s1')

        return s1


def discriminator_wgangp_reverse(image, flags, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        # image is 256 x 256 x input_c_dim
        image_reshape = tf.reshape(image, [-1, flags.image_height, flags.image_width, flags.c_in_dim])
        image_normalization = image_normalization_one(image_reshape, name='image_normalization_one')

        # PatchGAN n=4, https://github.com/phillipi/pix2pix/blob/master/models.lua#L131
        # Receptive field=(14x142), https://github.com/phillipi/pix2pix/blob/master/scripts/receptive_field_sizes.m
        # 128x128,
        n1 = conv2d(image_normalization, flags.df_dim, ks=4, s=2, padding='SAME', name='n1')
        # 64x64,
        n2 = instance_normalization_lrelu_conv2d(n1, flags.df_dim * 2, ks=4, s=2, padding='SAME', name='n2')
        # 32x32,
        n3 = instance_normalization_lrelu_conv2d(n2, flags.df_dim * 4, ks=4, s=2, padding='SAME', name='n3')
        # 16x16,
        n4 = instance_normalization_lrelu_conv2d(n3, flags.df_dim * 8, ks=4, s=2, padding='SAME', name='n4')
        # 8x8
        n5 = instance_normalization_lrelu_conv2d(n4, flags.df_dim * 16, ks=4, s=2, padding='SAME', name='n5')
        # 1x1
        s1 = instance_normalization_lrelu_conv2d(n5, 1, ks=8, s=1, padding='VALID', name='s1')

        return s1


def discriminator_wgangp_lazy(image, flags, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        # image is 256 x 256 x input_c_dim
        image_reshape = tf.reshape(image, [-1, flags.image_height, flags.image_width, flags.c_in_dim])
        image_normalization = image_normalization_one(image_reshape, name='image_normalization_one')

        # PatchGAN n=4, https://github.com/phillipi/pix2pix/blob/master/models.lua#L131
        # Receptive field=(14x142), https://github.com/phillipi/pix2pix/blob/master/scripts/receptive_field_sizes.m
        # 128x128,
        n1 = conv2d_lrelu(image_normalization, flags.df_dim, ks=5, s=2, padding='SAME', name='n1')
        # 64x64,
        n2 = conv2d_lrelu(n1, flags.df_dim * 2, ks=5, s=2, padding='SAME', name='n2')
        # 32x32,
        n3 = conv2d_lrelu(n2, flags.df_dim * 4, ks=5, s=2, padding='SAME', name='n3')
        # 16x16,
        n4 = conv2d_lrelu(n3, flags.df_dim * 8, ks=5, s=2, padding='SAME', name='n4')
        # 8x8
        n5 = conv2d_lrelu(n4, flags.df_dim * 16, ks=5, s=2, padding='SAME', name='n5')
        # 1x1
        s1 = conv2d(n5, 1, ks=8, s=1, padding='VALID', name='s1')

        return s1


def generator_resnet_lazy(image, flags, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        # image is 256 x 256 x input_c_dim
        image_normalization = image_normalization_one(image, name='image_normalization_one')

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        image_pad = tf.pad(image_normalization, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = conv2d(image_pad, flags.gf_dim, ks=7, s=1, padding='VALID', name='c1')
        # 256x256
        c2 = relu_instance_normalization_conv2d(c1, flags.gf_dim * 2, ks=3, s=2, padding='SAME', name='c2')
        # 128x128
        c3 = relu_instance_normalization_conv2d(c2, flags.gf_dim * 4, ks=3, s=2, padding='SAME', name='c3')
        # 64x64

        # define G network with 9 resnet blocks
        r1 = residule_block_se_fixed(c3, flags.gf_dim * 4, name='r1')
        r2 = residule_block_se_fixed(r1, flags.gf_dim * 4, name='r2')
        r3 = residule_block_se_fixed(r2, flags.gf_dim * 4, name='r3')
        r4 = residule_block_se_fixed(r3, flags.gf_dim * 4, name='r4')
        r5 = residule_block_se_fixed(r4, flags.gf_dim * 4, name='r5')
        r6 = residule_block_se_fixed(r5, flags.gf_dim * 4, name='r6')
        r7 = residule_block_se_fixed(r6, flags.gf_dim * 4, name='r7')
        r8 = residule_block_se_fixed(r7, flags.gf_dim * 4, name='r8')
        r9 = residule_block_se_fixed(r8, flags.gf_dim * 4, name='r9')

        d2 = relu_instance_normalization_deconv2d(r9, flags.gf_dim * 2, 3, 2, name='d2')
        # 128x128
        # c2_skip_scale = skip_squeeze_excitation_layer(c2, dim=flags.gf_dim * 2, name='c2_skip_scale')
        # d2_fuse = tf.add(d2, c2_skip_scale, name='d2_fuse')

        d1 = relu_instance_normalization_deconv2d(d2, flags.gf_dim * 2, 3, 2, name='d1')
        # 256x256
        # c1_skip_scale = skip_squeeze_excitation_layer(c1, dim=flags.gf_dim * 2, name='c1_skip_scale')
        # d1_fuse = tf.add(d1, c1_skip_scale, name='d1_fuse')
        d1_pad = tf.pad(d1, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

        # Pred
        # pred = tf.nn.tanh(conv2d(d2, options.c_out_dim, 7, 1, padding='VALID', name='g_pred_c'))
        pred = relu_instance_normalization_conv2d(d1_pad, flags.c_out_dim, 7, 1, padding='VALID', name='pred')
        pred_sigmoid = tf.nn.sigmoid(pred)

        return pred_sigmoid


def generator_resnet_lazy_reverse(image, flags, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        # image is 256 x 256 x input_c_dim
        image_normalization = image_normalization_one(image, name='image_normalization_one')

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        image_pad = tf.pad(image_normalization, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = conv2d(image_pad, flags.gf_dim, ks=7, s=1, padding='VALID', name='c1')
        # 256x256
        c2 = instance_normalization_relu_conv2d(c1, flags.gf_dim * 2, ks=3, s=2, padding='SAME', name='c2')
        # 128x128
        c3 = instance_normalization_relu_conv2d(c2, flags.gf_dim * 4, ks=3, s=2, padding='SAME', name='c3')
        # 64x64

        # define G network with 9 resnet blocks
        r1 = residule_block_se_fixed_reverse(c3, flags.gf_dim * 4, name='r1')
        r2 = residule_block_se_fixed_reverse(r1, flags.gf_dim * 4, name='r2')
        r3 = residule_block_se_fixed_reverse(r2, flags.gf_dim * 4, name='r3')
        r4 = residule_block_se_fixed_reverse(r3, flags.gf_dim * 4, name='r4')
        r5 = residule_block_se_fixed_reverse(r4, flags.gf_dim * 4, name='r5')
        r6 = residule_block_se_fixed_reverse(r5, flags.gf_dim * 4, name='r6')
        r7 = residule_block_se_fixed_reverse(r6, flags.gf_dim * 4, name='r7')
        r8 = residule_block_se_fixed_reverse(r7, flags.gf_dim * 4, name='r8')
        r9 = residule_block_se_fixed_reverse(r8, flags.gf_dim * 4, name='r9')

        d2 = instance_normalization_relu_deconv2d(r9, flags.gf_dim * 2, 3, 2, name='d2')
        # 128x128
        # c2_skip_scale = skip_squeeze_excitation_layer(c2, dim=flags.gf_dim * 2, name='c2_skip_scale')
        # d2_fuse = tf.add(d2, c2_skip_scale, name='d2_fuse')

        d1 = instance_normalization_relu_deconv2d(d2, flags.gf_dim * 2, 3, 2, name='d1')
        # 256x256
        # c1_skip_scale = skip_squeeze_excitation_layer(c1, dim=flags.gf_dim * 2, name='c1_skip_scale')
        # d1_fuse = tf.add(d1, c1_skip_scale, name='d1_fuse')
        d1_pad = tf.pad(d1, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

        # Pred
        # pred = tf.nn.tanh(conv2d(d2, options.c_out_dim, 7, 1, padding='VALID', name='g_pred_c'))
        pred = instance_normalization_relu_conv2d(d1_pad, flags.c_out_dim, 7, 1, padding='VALID', name='pred')
        pred_sigmoid = tf.nn.sigmoid(pred)

        return pred_sigmoid


def discriminator_wgangp2(image, flags, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        # image is 256 x 256 x input_c_dim
        image_reshape = tf.reshape(image, [-1, flags.image_height, flags.image_width, flags.c_in_dim])
        image_normalization = image_normalization_one(image_reshape, name='image_normalization_one')

        # PatchGAN n=4, https://github.com/phillipi/pix2pix/blob/master/models.lua#L131
        # Receptive field=(14x142), https://github.com/phillipi/pix2pix/blob/master/scripts/receptive_field_sizes.m
        # 128x128,
        n1 = conv2d(image_normalization, flags.df_dim, ks=4, s=2, padding='SAME', name='n1')
        # 64x64,
        n2 = lrelu_instance_normalization_conv2d(n1, flags.df_dim * 2, ks=4, s=2, padding='SAME', name='n2')
        # 32x32,
        n3 = lrelu_instance_normalization_conv2d(n2, flags.df_dim * 4, ks=4, s=2, padding='SAME', name='n3')
        # 16x16,
        n4 = lrelu_instance_normalization_conv2d(n3, flags.df_dim * 8, ks=4, s=2, padding='SAME', name='n4')

        # 16x16
        s1 = lrelu_instance_normalization_conv2d(n4, flags.df_dim * 16, ks=4, s=1, padding='SAME', name='s1')
        # 16x16
        s2 = lrelu_instance_normalization_conv2d(s1, flags.df_dim * 16, ks=4, s=1, padding='SAME', name='s2')
        # 16x16
        s3 = lrelu_instance_normalization_conv2d(s2, 1, ks=4, s=1, padding='SAME', name='s3')

        return s3


def generator_resnet(image, options, reuse, name):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        image_normalization = image_normalization_sub(image, name='image_normalization_one')

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image_normalization, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_normalization(conv2d(c0, options.gf_dim,
                                                      7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_normalization(conv2d(c1, options.gf_dim*2,
                                                      3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_normalization(conv2d(c2, options.gf_dim*4,
                                                      3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')

        d1 = deconv2d(r9, options.gf_dim*2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_normalization(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_normalization(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        # pred = tf.nn.tanh(conv2d(d2, options.c_out_dim, 7, 1, padding='VALID', name='g_pred_c'))
        pred = conv2d(d2, options.c_out_dim, 7, 1, padding='VALID', name='g_pred_c')

        return pred


def generator_resnet_big(image, options, reuse, name):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_normalization(conv2d(c0, options.gf_dim,
                                                      7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_normalization(conv2d(c1, options.gf_dim*2,
                                                      3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_normalization(conv2d(c2, options.gf_dim*4,
                                                      3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')

        d1 = deconv2d(r9, options.gf_dim*2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_normalization(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim*2, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_normalization(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(conv2d(d2, options.c_out_dim, 7, 1, padding='VALID', name='g_pred_c'))

        return pred


def generator_resnet_sigmoid(image, options, reuse, name):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_normalization(conv2d(c0, options.gf_dim,
                                                      7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_normalization(conv2d(c1, options.gf_dim*2,
                                                      3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_normalization(conv2d(c2, options.gf_dim*4,
                                                      3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')

        d1 = deconv2d(r9, options.gf_dim*2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_normalization(d1, 'g_d1_bn'))

        d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_normalization(d2, 'g_d2_bn'))

        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        logits = conv2d(d2, options.c_out_dim, 7, 1, padding='VALID', name='g_pred_c')

        return logits


def generator_resnet_u(image, options, reuse, name):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_normalization(conv2d(c0, options.gf_dim,
                                                      7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_normalization(conv2d(c1, options.gf_dim*2,
                                                      3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_normalization(conv2d(c2, options.gf_dim*4,
                                                      3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')

        d1 = deconv2d(r9, options.gf_dim*2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_normalization(d1, 'g_d1_bn'))

        d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_normalization(d2, 'g_d2_bn'))

        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.sigmoid(conv2d(d2, options.c_out_dim, 7, 1, padding='VALID', name='g_pred_c'))

        return pred


def generator_resnet_sigmoid_big(image, options, reuse, name):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_normalization(conv2d(c0, options.gf_dim,
                                                      7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_normalization(conv2d(c1, options.gf_dim*2,
                                                      3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_normalization(conv2d(c2, options.gf_dim*4,
                                                      3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')

        d1 = deconv2d(r9, options.gf_dim*2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_normalization(d1, 'g_d1_bn'))

        d2 = deconv2d(d1, options.gf_dim*2, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_normalization(d2, 'g_d2_bn'))

        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.sigmoid(conv2d(d2, options.c_out_dim, 7, 1, padding='VALID', name='g_pred_c'))

        return pred


def generator_resnet_small(image, options, reuse, name):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_normalization(conv2d(c0, options.gf_dim,
                                                      7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_normalization(conv2d(c1, options.gf_dim*2,
                                                      3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_normalization(conv2d(c2, options.gf_dim*4,
                                                      3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')

        r9_up = tf.image.resize_bilinear(r9, (options.image_height, options.image_width), name='r9_up')
        d2 = tf.pad(r9_up, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.sigmoid(conv2d(d2, options.c_out_dim, 7, 1, padding='VALID', name='g_pred_c'))

        return pred


def generator_resnet_sigmoid_skip(image, options, reuse, name):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_normalization(conv2d(c0, options.gf_dim,
                                                      7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_normalization(conv2d(c1, options.gf_dim*2,
                                                      3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_normalization(conv2d(c2, options.gf_dim*4,
                                                      3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')

        d1 = deconv2d(r9, options.gf_dim*2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_normalization(d1, 'g_d1_bn'))
        c2_skip = tf.nn.relu(instance_normalization(conv2d(c2, options.gf_dim*2,
                                                           1, 1, name='g_c2_skip'), 'g_c2_skip_bn'))

        d2 = deconv2d(d1+c2_skip, options.gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_normalization(d2, 'g_d2_bn'))
        c1_skip = tf.nn.relu(instance_normalization(conv2d(c1, options.gf_dim,
                                                           1, 1, name='g_c1_skip'), 'g_c1_skip_bn'))

        d2 = tf.pad(d2+c1_skip, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.sigmoid(conv2d(d2, options.c_out_dim, 7, 1, padding='VALID', name='g_pred_c'))

        return pred


def discriminator_se_wgangp(image, flags, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        # image is 256 x 256 x input_c_dim
        image_reshape = tf.reshape(image, [-1, flags.image_height, flags.image_width, flags.c_in_dim])
        image_normalization = image_normalization_one(image_reshape, name='image_normalization_one')

        h0 = lrelu(conv2d(image_normalization, flags.df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(instance_normalization(conv2d(h0, flags.df_dim * 2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_normalization(conv2d(h1, flags.df_dim * 4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_normalization(conv2d(h2, flags.df_dim * 8, s=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
        # h4 is (32 x 32 x 1)

        # h4_average = tf.reduce_mean(h4, axis=[1, 2, 3])
        # return h4_average
        # hf_f = tf.reshape(h4, [-1, flags.image_height * flags.image_width // ((2 ** 3) ** 2)], name='hf_f')
        return h4


def discriminator_se_wgangp_sub(image, flags, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        # image is 256 x 256 x input_c_dim
        image_reshape = tf.reshape(image, [-1, flags.image_height, flags.image_width, flags.c_in_dim])
        image_normalization = image_normalization_sub(image_reshape, name='image_normalization_one')

        h0 = lrelu(conv2d(image_normalization, flags.df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(instance_normalization(conv2d(h0, flags.df_dim * 2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_normalization(conv2d(h1, flags.df_dim * 4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_normalization(conv2d(h2, flags.df_dim * 8, s=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
        # h4 is (32 x 32 x 1)

        # h4_average = tf.reduce_mean(h4, axis=[1, 2, 3])
        # return h4_average
        # hf_f = tf.reshape(h4, [-1, flags.image_height * flags.image_width // ((2 ** 3) ** 2)], name='hf_f')
        return h4


def discriminator_rule(image, flags, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        # image is 256 x 256 x input_c_dim
        image_reshape = tf.reshape(image, [-1, flags.image_height, flags.image_width, flags.c_in_dim])
        image_normalization = image_normalization_one(image_reshape, name='image_normalization_one')

        # PatchGAN n=3, https://github.com/phillipi/pix2pix/blob/master/models.lua#L131
        # Receptive field=(70, 70), https://github.com/phillipi/pix2pix/blob/master/scripts/receptive_field_sizes.m
        n1 = conv2d(image_normalization, flags.df_dim, ks=4, s=2, name='n1')
        # 128x128,
        n2 = lrelu_conv2d(n1, flags.df_dim * 2, ks=4, s=2, name='n2')  # First layer no normalization
        # 64x64,
        n3 = instance_normalization_lrelu_conv2d(n2, flags.df_dim * 4, ks=4, s=2, padding='SAME', name='n3')
        # 32x32,
        # n4 = instance_normalization_lrelu_conv2d(n3, flags.df_dim * 8, ks=4, s=2, padding='SAME', name='n4')
        # # 16x16,

        s1 = instance_normalization_lrelu_conv2d(n3, flags.df_dim * 8, ks=4, s=1, padding='SAME', name='s1')
        # 32x32
        s2 = instance_normalization_lrelu_conv2d(s1, 1, ks=4, s=1, padding='SAME', name='s2')
        # 32x32

        return s2


def generator_rule(image, flags, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        # image is 256 x 256 x input_c_dim
        image_normalization = image_normalization_one(image, name='image_normalization_one')

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c1 = reflect_pad_conv(image_normalization, flags.gf_dim, ks=7, s=1, ps=3, padding='VALID', name='c1')
        # 256x256
        c2 = instance_normalization_relu_conv2d(c1, flags.gf_dim * 2, ks=3, s=2, padding='SAME', name='c2')
        # 128x128
        c3 = instance_normalization_relu_conv2d(c2, flags.gf_dim * 4, ks=3, s=2, padding='SAME', name='c3')
        # 64x64

        # define G network with 9 resnet blocks
        r1 = residule_block_rule(c3, flags.gf_dim * 4, name='r1')
        r2 = residule_block_rule(r1, flags.gf_dim * 4, name='r2')
        r3 = residule_block_rule(r2, flags.gf_dim * 4, name='r3')
        r4 = residule_block_rule(r3, flags.gf_dim * 4, name='r4')
        r5 = residule_block_rule(r4, flags.gf_dim * 4, name='r5')
        r6 = residule_block_rule(r5, flags.gf_dim * 4, name='r6')
        r7 = residule_block_rule(r6, flags.gf_dim * 4, name='r7')
        r8 = residule_block_rule(r7, flags.gf_dim * 4, name='r8')
        r9 = residule_block_rule(r8, flags.gf_dim * 4, name='r9')

        pred_s2 = reflect_pad_instance_normalization_relu_conv(r9, flags.c_out_dim, 3, 1,
                                                               ps=1, padding='VALID', name='pred_s2')
        pred_s2_activate = tf.nn.tanh(pred_s2, name='pred_s2_activate')

        d2 = instance_normalization_relu_deconv2d(r9, flags.gf_dim * 2, 3, 2, name='d2')
        # 128x128
        # c2_skip_scale = skip_squeeze_excitation_layer(c2, dim=flags.gf_dim * 2, name='c2_skip_scale')
        # d2_fuse = tf.add(d2, c2_skip_scale, name='d2_fuse')
        pred_s1 = reflect_pad_instance_normalization_relu_conv(d2, flags.c_out_dim, 3, 1,
                                                               ps=1, padding='VALID', name='pred_s1')
        pred_s1_activate = tf.nn.tanh(pred_s1, name='pred_s1_activate')


        d1 = instance_normalization_relu_deconv2d(d2, flags.gf_dim * 2, 3, 2, name='d1')
        # 256x256
        # c1_skip_scale = skip_squeeze_excitation_layer(c1, dim=flags.gf_dim * 2, name='c1_skip_scale')
        # d1_fuse = tf.add(d1, c1_skip_scale, name='d1_fuse')

        # Pred
        pred = reflect_pad_instance_normalization_relu_conv(d1, flags.c_out_dim, 7, 1,
                                                            ps=3, padding='VALID', name='pred')
        pred_activate = tf.nn.tanh(pred, name='pred_activate')

        return pred_activate


def generator_rule_sig(image, flags, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        # image is 256 x 256 x input_c_dim
        image_normalization = image_normalization_one(image, name='image_normalization_one')

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c1 = reflect_pad_conv(image_normalization, flags.gf_dim, ks=7, s=1, ps=3, padding='VALID', name='c1')
        # 256x256
        c2 = instance_normalization_relu_conv2d(c1, flags.gf_dim * 2, ks=3, s=2, padding='SAME', name='c2')
        # 128x128
        c3 = instance_normalization_relu_conv2d(c2, flags.gf_dim * 4, ks=3, s=2, padding='SAME', name='c3')
        # 64x64

        # define G network with 9 resnet blocks
        r1 = residule_block_rule(c3, flags.gf_dim * 4, name='r1')
        r2 = residule_block_rule(r1, flags.gf_dim * 4, name='r2')
        r3 = residule_block_rule(r2, flags.gf_dim * 4, name='r3')
        r4 = residule_block_rule(r3, flags.gf_dim * 4, name='r4')
        r5 = residule_block_rule(r4, flags.gf_dim * 4, name='r5')
        r6 = residule_block_rule(r5, flags.gf_dim * 4, name='r6')
        r7 = residule_block_rule(r6, flags.gf_dim * 4, name='r7')
        r8 = residule_block_rule(r7, flags.gf_dim * 4, name='r8')
        r9 = residule_block_rule(r8, flags.gf_dim * 4, name='r9')

        d2 = instance_normalization_relu_deconv2d(r9, flags.gf_dim * 2, 3, 2, name='d2')
        # 128x128
        c2_skip_scale = skip_squeeze_excitation_layer(c2, dim=flags.gf_dim * 2, name='c2_skip_scale')
        # d2_fuse = tf.concat([d2, c2_skip_scale], axis=3,  name='d2_fuse')
        d2_fuse = tf.add(d2, c2_skip_scale, name='d2_fuse')

        d1 = instance_normalization_relu_deconv2d(d2_fuse, flags.gf_dim * 2, 3, 2, name='d1')
        # 256x256
        c1_skip_scale = skip_squeeze_excitation_layer(c1, dim=flags.gf_dim * 2, name='c1_skip_scale')
        # d1_fuse = tf.concat([d1, c1_skip_scale], axis=3, name='d1_fuse')
        d1_fuse = tf.add(d1, c1_skip_scale, name='d1_fuse')

        # Pred
        pred_s2 = reflect_pad_instance_normalization_relu_conv(r9, flags.c_out_dim, 3, 1,
                                                               ps=1, padding='VALID', name='pred_s2')
        pred_s2_activate = tf.nn.tanh(pred_s2, name='pred_s2_activate')

        pred_s1 = reflect_pad_instance_normalization_relu_conv(d2_fuse, flags.c_out_dim, 3, 1,
                                                               ps=1, padding='VALID', name='pred_s1')
        pred_s1_activate = tf.nn.tanh(pred_s1, name='pred_s1_activate')

        pred = reflect_pad_instance_normalization_relu_conv(d1_fuse, flags.c_out_dim, 7, 1,
                                                            ps=3, padding='VALID', name='pred')
        pred_activate = tf.nn.sigmoid(pred, name='pred_activate')

        return pred_activate


def generator_zero(image, flags, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        # image is 256 x 256 x input_c_dim
        image_normalization = image_normalization_one(image, name='image_normalization_one')

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c1 = reflect_pad_conv(image_normalization, flags.gf_dim, ks=7, s=1, ps=3, padding='VALID', name='c1')
        # 256x256
        c2 = instance_normalization_relu_conv2d(c1, flags.gf_dim * 2, ks=3, s=2, padding='SAME', name='c2')
        # 128x128
        c3 = instance_normalization_relu_conv2d(c2, flags.gf_dim * 4, ks=3, s=2, padding='SAME', name='c3')
        # 64x64

        # define G network with 9 resnet blocks
        r1 = residule_block_zero(c3, flags.gf_dim * 4, name='r1')
        r2 = residule_block_zero(r1, flags.gf_dim * 4, name='r2')
        r3 = residule_block_zero(r2, flags.gf_dim * 4, name='r3')
        r4 = residule_block_zero(r3, flags.gf_dim * 4, name='r4')
        r5 = residule_block_zero(r4, flags.gf_dim * 4, name='r5')
        r6 = residule_block_zero(r5, flags.gf_dim * 4, name='r6')
        r7 = residule_block_zero(r6, flags.gf_dim * 4, name='r7')
        r8 = residule_block_zero(r7, flags.gf_dim * 4, name='r8')
        r9 = residule_block_zero(r8, flags.gf_dim * 4, name='r9')

        d2 = instance_normalization_relu_deconv2d(r9, flags.gf_dim * 2, 3, 2, name='d2')
        d1 = instance_normalization_relu_deconv2d(d2, flags.gf_dim, 3, 2, name='d1')

        # Pred
        logits = relu_reflect_pad_conv(d1, flags.c_out_dim, 7, 1, 3, 'VALID', name='logits')

        return logits


def generator_why(image, flags, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        # image is 256 x 256 x input_c_dim
        image_normalization = image_normalization_one(image, name='image_normalization_one')

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c1 = reflect_pad_conv(image_normalization, flags.gf_dim, ks=7, s=1, ps=3, padding='VALID', name='c1')
        # 256x256
        c2 = instance_normalization_relu_conv2d(c1, flags.gf_dim * 2, ks=3, s=2, padding='SAME', name='c2')
        # 128x128
        c3 = instance_normalization_relu_conv2d(c2, flags.gf_dim * 4, ks=3, s=2, padding='SAME', name='c3')
        # 64x64

        # define G network with 9 resnet blocks
        r1 = residule_block_why(c3, flags.gf_dim * 4, name='r1')
        r2 = residule_block_why(r1, flags.gf_dim * 4, name='r2')
        r3 = residule_block_why(r2, flags.gf_dim * 4, name='r3')
        r4 = residule_block_why(r3, flags.gf_dim * 4, name='r4')
        r5 = residule_block_why(r4, flags.gf_dim * 4, name='r5')
        r6 = residule_block_why(r5, flags.gf_dim * 4, name='r6')
        r7 = residule_block_why(r6, flags.gf_dim * 4, name='r7')
        r8 = residule_block_why(r7, flags.gf_dim * 4, name='r8')
        r9 = residule_block_why(r8, flags.gf_dim * 4, name='r9')

        d2 = skip_combine(r9, c2, flags.gf_dim * 2, name='d2')
        d1 = skip_combine(d2, c1, flags.gf_dim, name='d1')

        # Pred
        logits = relu_reflect_pad_conv(d1, flags.c_out_dim, 7, 1, 3, 'VALID', name='logits')

        return logits


def discriminator_zero(image, flags, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        # image is 256 x 256 x input_c_dim
        image_reshape = tf.reshape(image, [-1, flags.image_height, flags.image_width, flags.c_in_dim])
        image_normalization = image_normalization_one(image_reshape, name='image_normalization_one')

        # PatchGAN n=4, https://github.com/phillipi/pix2pix/blob/master/models.lua#L131
        # Receptive field=(14x142), https://github.com/phillipi/pix2pix/blob/master/scripts/receptive_field_sizes.m
        # 128x128,
        n1 = conv2d(image_normalization, flags.df_dim, ks=4, s=2, padding='SAME', name='n1')
        # 64x64,
        n2 = lrelu_conv2d(n1, flags.df_dim * 2, ks=4, s=2, padding='SAME', name='n2')
        # 32x32,
        n3 = instance_normalization_lrelu_conv2d(n2, flags.df_dim * 4, ks=4, s=2, padding='SAME', name='n3')
        # 16x16,
        n4 = instance_normalization_lrelu_conv2d(n3, flags.df_dim * 8, ks=4, s=2, padding='SAME', name='n4')
        # 8x8
        n5 = instance_normalization_lrelu_conv2d(n4, flags.df_dim * 16, ks=4, s=2, padding='SAME', name='n5')
        # 1x1
        s1 = instance_normalization_lrelu_conv2d(n5, 1, ks=7, s=1, padding='VALID', name='s1')

        return s1


def generator_fpn(image, flags, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        # image is 256 x 256 x input_c_dim
        image_normalization = image_normalization_one(image, name='image_normalization_one')

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c1 = reflect_pad_conv(image_normalization, flags.gf_dim, ks=7, s=1, ps=3, padding='VALID', name='c1')
        # 256x256
        c2 = instance_normalization_relu_conv2d(c1, flags.gf_dim * 2, ks=3, s=2, padding='SAME', name='c2')
        # 128x128
        c3 = instance_normalization_relu_conv2d(c2, flags.gf_dim * 4, ks=3, s=2, padding='SAME', name='c3')
        # 64x64

        # define G network with 9 resnet blocks
        r1 = residule_block_zero(c3, flags.gf_dim * 4, name='r1')
        r2 = residule_block_zero(r1, flags.gf_dim * 4, name='r2')
        r3 = residule_block_zero(r2, flags.gf_dim * 4, name='r3')
        r4 = residule_block_zero(r3, flags.gf_dim * 4, name='r4')
        r5 = residule_block_zero(r4, flags.gf_dim * 4, name='r5')
        r6 = residule_block_zero(r5, flags.gf_dim * 4, name='r6')
        r7 = residule_block_zero(r6, flags.gf_dim * 4, name='r7')
        r8 = residule_block_zero(r7, flags.gf_dim * 4, name='r8')
        r9 = residule_block_zero(r8, flags.gf_dim * 4, name='r9')

        d2 = fpn_up(r9, flags.image_height // 2, c2, flags.gf_dim * 4, name='d2')
        d1 = fpn_up(d2, flags.image_height, c1, flags.gf_dim * 4, name='d1')

        # Pred
        # pred_s2 = pred_output(r9, flags.c_out_dim, 3, 1, ps=1, name='pred_s2', flags=flags)
        # pred_s1 = pred_output(d2_concat, flags.c_out_dim, 3, 1, ps=1, name='pred_s1', flags=flags)
        pred_s0 = pred_output(d1, flags.c_out_dim, 7, 1, ps=3, name='pred_s0')

        return pred_s0


def generator_data(image, flags, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        # image is 256 x 256 x input_c_dim
        image_normalization = image_normalization_one(image, name='image_normalization_one')

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c1 = reflect_pad_conv(image_normalization, flags.gf_dim, ks=7, s=1, ps=3, padding='VALID', name='c1')
        # 256x256
        c2 = instance_normalization_relu_conv2d(c1, flags.gf_dim * 2, ks=3, s=2, padding='SAME', name='c2')
        # 128x128
        c3 = instance_normalization_relu_conv2d(c2, flags.gf_dim * 4, ks=3, s=2, padding='SAME', name='c3')
        # 64x64

        # define G network with 9 resnet blocks
        r1 = residule_block_zero(c3, flags.gf_dim * 4, name='r1')
        r2 = residule_block_zero(r1, flags.gf_dim * 4, name='r2')
        r3 = residule_block_zero(r2, flags.gf_dim * 4, name='r3')
        r4 = residule_block_zero(r3, flags.gf_dim * 4, name='r4')
        r5 = residule_block_zero(r4, flags.gf_dim * 4, name='r5')
        r6 = residule_block_zero(r5, flags.gf_dim * 4, name='r6')
        r7 = residule_block_zero(r6, flags.gf_dim * 4, name='r7')
        r8 = residule_block_zero(r7, flags.gf_dim * 4, name='r8')
        r9 = residule_block_zero(r8, flags.gf_dim * 4, name='r9')

        # d2 = data_up(r9, c3, flags.gf_dim * 4, name='d2')
        # d1 = data_up(d2, c2, flags.gf_dim * 4, name='d1')
        d2 = instance_normalization_relu_deconv2d(r9, flags.gf_dim * 4, 3, 2, name='d2')
        d1 = instance_normalization_relu_deconv2d(d2, flags.gf_dim * 4, 3, 2, name='d1')

        # Pred
        # pred_s2 = pred_output(r9, flags.c_out_dim, 3, 1, ps=1, name='pred_s2', flags=flags)
        # pred_s1 = pred_output(d2_concat, flags.c_out_dim, 3, 1, ps=1, name='pred_s1', flags=flags)
        pred_s0 = pred_output(d1, flags.c_out_dim, 7, 1, ps=3, name='pred_s0')

        return pred_s0


def generator_final(image, flags, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        # image is 256 x 256 x input_c_dim
        image_normalization = image_normalization_one(image, name='image_normalization_one')

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c1 = reflect_pad_conv(image_normalization, flags.gf_dim, ks=7, s=1, ps=3, padding='VALID', name='c1')
        # 256x256
        c2 = instance_normalization_relu_conv2d(c1, flags.gf_dim * 2, ks=3, s=2, padding='SAME', name='c2')
        # 128x128
        c3 = instance_normalization_relu_conv2d(c2, flags.gf_dim * 4, ks=3, s=2, padding='SAME', name='c3')
        # 64x64

        # define G network with 9 resnet blocks
        r1 = residule_block_rule(c3, flags.gf_dim * 4, name='r1')
        r2 = residule_block_rule(r1, flags.gf_dim * 4, name='r2')
        r3 = residule_block_rule(r2, flags.gf_dim * 4, name='r3')
        r4 = residule_block_rule(r3, flags.gf_dim * 4, name='r4')
        r5 = residule_block_rule(r4, flags.gf_dim * 4, name='r5')
        r6 = residule_block_rule(r5, flags.gf_dim * 4, name='r6')
        r7 = residule_block_rule(r6, flags.gf_dim * 4, name='r7')
        r8 = residule_block_rule(r7, flags.gf_dim * 4, name='r8')
        r9 = residule_block_rule(r8, flags.gf_dim * 4, name='r9')

        # d2 = data_up(r9, c3, flags.gf_dim * 4, name='d2')
        # d1 = data_up(d2, c2, flags.gf_dim * 4, name='d1')
        d2 = skip_combine(r9, c2, flags.gf_dim * 2, name='d2')
        d1 = skip_combine(d2, c1, flags.gf_dim * 2, name='d1')

        # Pred
        # pred_s2 = pred_output(r9, flags.c_out_dim, 3, 1, ps=1, name='pred_s2', flags=flags)
        # pred_s1 = pred_output(d2_concat, flags.c_out_dim, 3, 1, ps=1, name='pred_s1', flags=flags)
        pred_s0 = pred_output(d1, flags.c_out_dim, 7, 1, ps=3, name='pred_s0')

        return pred_s0


def generator_z(image, flags, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        # image is 256 x 256 x input_c_dim
        image_normalization = image_normalization_one(image, name='image_normalization_one')

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c1 = reflect_pad_conv(image_normalization, flags.gf_dim, ks=7, s=1, ps=3, padding='VALID', name='c1')
        # 256x256
        c2 = instance_normalization_relu_conv2d(c1, flags.gf_dim * 2, ks=3, s=2, padding='SAME', name='c2')
        # 128x128
        c3 = instance_normalization_relu_conv2d(c2, flags.gf_dim * 4, ks=3, s=2, padding='SAME', name='c3')
        # 64x64

        # define G network with 9 resnet blocks
        r1 = residule_block_zero(c3, flags.gf_dim * 4, name='r1')
        r2 = residule_block_zero(r1, flags.gf_dim * 4, name='r2')
        r3 = residule_block_zero(r2, flags.gf_dim * 4, name='r3')
        r4 = residule_block_zero(r3, flags.gf_dim * 4, name='r4')
        r5 = residule_block_zero(r4, flags.gf_dim * 4, name='r5')
        r6 = residule_block_zero(r5, flags.gf_dim * 4, name='r6')
        r7 = residule_block_zero(r6, flags.gf_dim * 4, name='r7')
        r8 = residule_block_zero(r7, flags.gf_dim * 4, name='r8')
        r9 = residule_block_zero(r8, flags.gf_dim * 4, name='r9')

        d2 = z_up(r9, flags.image_height // 2, c2, flags.gf_dim * 2, name='d2')
        d1 = z_up(d2, flags.image_height,  c1, flags.gf_dim, name='d1')
        # Pred
        pred_s0 = pred_output(d1, flags.c_out_dim, 7, 1, ps=3, name='pred_s0')

        return pred_s0


def generator_zero_simple(image, flags, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        # image is 256 x 256 x input_c_dim
        image_normalization = image_normalization_one(image, name='image_normalization_one')

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c1 = reflect_pad_conv(image_normalization, flags.gf_dim, ks=7, s=1, ps=3, padding='VALID', name='c1')
        # 256x256
        c2 = instance_normalization_relu_conv2d(c1, flags.gf_dim * 2, ks=3, s=2, padding='SAME', name='c2')
        # 128x128
        c3 = instance_normalization_relu_conv2d(c2, flags.gf_dim * 4, ks=3, s=2, padding='SAME', name='c3')
        # 64x64

        # define G network with 9 resnet blocks
        r1 = residule_block_zero(c3, flags.gf_dim * 4, name='r1')
        r2 = residule_block_zero(r1, flags.gf_dim * 4, name='r2')
        r3 = residule_block_zero(r2, flags.gf_dim * 4, name='r3')
        r4 = residule_block_zero(r3, flags.gf_dim * 4, name='r4')
        r5 = residule_block_zero(r4, flags.gf_dim * 4, name='r5')
        r6 = residule_block_zero(r5, flags.gf_dim * 4, name='r6')
        r7 = residule_block_zero(r6, flags.gf_dim * 4, name='r7')
        r8 = residule_block_zero(r7, flags.gf_dim * 4, name='r8')
        r9 = residule_block_zero(r8, flags.gf_dim * 4, name='r9')

        d2 = instance_normalization_relu_deconv2d(r9, flags.gf_dim * 2, 3, 2, name='d2')
        # 128x128
        d2_concat = skip_concat(c2, d2, dim=flags.gf_dim * 2, name='d2_concat')

        d1 = instance_normalization_relu_deconv2d(d2_concat, flags.gf_dim * 2, 3, 2, name='d1')
        # 256x256
        d1_concat = skip_concat(c1, d1, dim=flags.gf_dim, name='d1_concat')

        # Pred
        pred_s2 = pred_output(r9, flags.c_out_dim, 3, 1, ps=1, name='pred_s2', flags=flags)
        pred_s1 = pred_output(d2_concat, flags.c_out_dim, 3, 1, ps=1, name='pred_s1', flags=flags)
        pred_s0 = pred_output(d1_concat, flags.c_out_dim, 7, 1, ps=3, name='pred_s0')

        return (0.2 * pred_s0) + (0.3 * pred_s1) + (0.5 * pred_s2)


def generator_combine(image, flags, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        # image is 256 x 256 x input_c_dim
        image_normalization = image_normalization_one(image, name='image_normalization_one')

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c1 = reflect_pad_conv(image_normalization, flags.gf_dim, ks=7, s=1, ps=3, padding='VALID', name='c1')
        # 256x256
        c2 = instance_normalization_relu_conv2d(c1, flags.gf_dim * 2, ks=3, s=2, padding='SAME', name='c2')
        # 128x128
        c3 = instance_normalization_relu_conv2d(c2, flags.gf_dim * 4, ks=3, s=2, padding='SAME', name='c3')
        # 64x64

        # define G network with 9 resnet blocks
        r1 = residule_block_rule(c3, flags.gf_dim * 4, name='r1')
        r2 = residule_block_rule(r1, flags.gf_dim * 4, name='r2')
        r3 = residule_block_rule(r2, flags.gf_dim * 4, name='r3')
        r4 = residule_block_rule(r3, flags.gf_dim * 4, name='r4')
        r5 = residule_block_rule(r4, flags.gf_dim * 4, name='r5')
        r6 = residule_block_rule(r5, flags.gf_dim * 4, name='r6')
        r7 = residule_block_rule(r6, flags.gf_dim * 4, name='r7')
        r8 = residule_block_rule(r7, flags.gf_dim * 4, name='r8')
        r9 = residule_block_rule(r8, flags.gf_dim * 4, name='r9')

        '''
        d2 = relu_instance_normalization_deconv2d(r9, flags.gf_dim * 2, 3, 2, name='d2')
        # 128x128
        c2_skip_scale = skip_squeeze_excitation_layer(c2, dim=flags.gf_dim * 2, name='c2_skip_scale')
        d2_fuse = tf.add(d2, c2_skip_scale, name='d2_fuse')

        d1 = relu_instance_normalization_deconv2d(d2_fuse, flags.gf_dim * 2, 3, 2, name='d1')
        # 256x256
        c1_skip_scale = skip_squeeze_excitation_layer(c1, dim=flags.gf_dim * 2, name='c1_skip_scale')
        d1_fuse = tf.add(d1, c1_skip_scale, name='d1_fuse')
        d1_pad = tf.pad(d1_fuse, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        '''

        # Pred
        pred_s2 = pred_output(r9, flags.c_out_dim, 3, 1, ps=1, name='pred_s2', flags=flags)
        # pred_s1 = pred_output(d2_concat, flags.c_out_dim, 3, 1, ps=1, name='pred_s1', flags=flags)
        # pred_s0 = pred_output(d1_concat, flags.c_out_dim, 7, 1, ps=3, name='pred_s0')

        return pred_s2


def blend_fake_b(real_a, adjusted_a, segment_a, name):
    with tf.variable_scope(name):
        foreground = tf.multiply(real_a, segment_a, name='foreground')
        background = tf.multiply(adjusted_a, (1 - segment_a), name='background')
        fake_b = tf.add(foreground, background, name='fake_b')
        return fake_b
