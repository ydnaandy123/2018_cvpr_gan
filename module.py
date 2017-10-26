from ops import *


def abs_criterion(in_, target, name):
    return tf.reduce_mean(tf.abs(in_ - target), name=name)


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


def residule_block(x, dim, ks=3, s=1, name='res'):
    with tf.variable_scope(name):
        p = int((ks - 1) / 2)
        y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y = instance_normalization(conv2d(y, dim, ks, s, padding='VALID', name='_c1'), name='_bn1')
        y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y = instance_normalization(conv2d(y, dim, ks, s, padding='VALID', name='_c2'), name='_bn2')
    return y + x


def generator_resnet(image, options, reuse, name):
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
        pred = tf.nn.tanh(conv2d(d2, options.c_out_dim, 7, 1, padding='VALID', name='g_pred_c'))

        return pred


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


def g_loss(dis_fake, flags, real_a, recon_a, real_b, recon_b, name):
    with tf.variable_scope(name):
        adversarial_loss = mae_criterion(dis_fake, tf.ones_like(dis_fake), name='adversarial')
        with tf.variable_scope('cycle'):
            cycle = flags.lambda_rec * (abs_criterion(real_a, recon_a) + abs_criterion(real_b, recon_b))
        return adversarial_loss + cycle


def d_loss(dis_real, dis_fake_pool, name):
    with tf.variable_scope(name):
        d_loss_real = mae_criterion(dis_real, tf.ones_like(dis_real), name='adversarial_real')
        d_loss_fake = mae_criterion(dis_fake_pool, tf.zeros_like(dis_fake_pool), name='adversarial_fake_pool')
        return d_loss_real + d_loss_fake


def train_op(loss, learning_rate, flags, var_list, name):
    with tf.variable_scope(name):
        optimizer = tf.train.AdamOptimizer(learning_rate, flags.beta1, flags.beta2, name='optimizer')
        grads = optimizer.compute_gradients(loss, var_list=var_list)
        '''
        if FLAGS.debug:
            # print(len(var_list))
            for grad, var in grads:
                utils.add_gradient_summary(grad, var)
        '''
        return optimizer.apply_gradients(grads, name='train_op')
