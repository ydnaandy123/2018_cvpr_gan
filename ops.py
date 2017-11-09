import tensorflow as tf


def train_op(loss, learning_rate, flags, var_list, name):
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
"""
Combinations
"""


def skip_why(input_tensor, y, scale_size, dim, name):
    with tf.variable_scope(name):
        conv = instance_normalization_relu_conv2d(input_tensor, dim, ks=3, s=1, padding='SAME', name='conv')
        up = tf.image.resize_bilinear(conv, (scale_size, scale_size), name='up_scale')
        skip = instance_normalization_relu_conv2d(y, dim, ks=1, s=1, padding='SAME', name='skip')
        combine = tf.concat([up, skip], axis=3, name='combine')
        fuse = instance_normalization_relu_conv2d(combine, dim, ks=3, s=1, padding='SAME', name='fuse')
        return fuse


def z_up(input_tensor, scale_size, y, dim, name):
    with tf.variable_scope(name):
        up = tf.image.resize_bilinear(input_tensor, (scale_size, scale_size), name='up_scale')
        skip = instance_normalization_relu_conv2d(y, dim*2, ks=1, s=1, padding='SAME', name='skip')
        fuse = tf.add(up, skip, name='fuse')
        conv = instance_normalization_relu_conv2d(fuse, dim, ks=3, s=1, padding='SAME', name='conv')
        return conv


def deconv_skip_fuse(input_tensor, y, dim, name):
    with tf.variable_scope(name):
        deconv = instance_normalization_relu_deconv2d(input_tensor, dim, 3, 2, name='deconv')
        skip = instance_normalization_relu_conv2d(y, dim, ks=1, s=1, padding='SAME', name='skip')
        fuse = tf.add(deconv, skip, name='sum')
        conv = instance_normalization_relu_conv2d(fuse, dim, ks=3, s=1, padding='SAME', name='conv')
        return conv


def data_up(input_tensor, y, dim, name):
    with tf.variable_scope(name):
        concat = tf.concat([input_tensor, y], axis=3, name='concat')
        up = instance_normalization_relu_deconv2d(concat, dim, 3, 2, name='deconv')
        return up


def fpn_up(input_tensor, scale_size, y, dim, name):
    with tf.variable_scope(name):
        up = tf.image.resize_bilinear(input_tensor, (scale_size, scale_size), name='up_scale')
        skip = instance_normalization_relu_conv2d(y, dim, ks=1, s=1, padding='SAME', name='skip')
        sum = tf.add(up, skip, name='sum')
        fuse = instance_normalization_relu_conv2d(sum, dim, ks=3, s=1, padding='SAME', name='fuse')
        return fuse


def dssd_up(input_tensor, scale_size, y, dim, name):
    with tf.variable_scope(name):
        up = instance_normalization_relu_deconv2d(input_tensor, dim, 3, 2, name='d2')
        up = tf.image.resize_bilinear(input_tensor, (scale_size, scale_size), name='up_scale')
        skip = instance_normalization_relu_conv2d(y, dim, ks=1, s=1, padding='SAME', name='skip')
        fuse = tf.add(up, skip, name='fuse')

        return fuse


def pred_output(input_tensor, dim, ks, s, ps, name, flags=None):
    with tf.variable_scope(name):
        pred = reflect_pad_instance_normalization_relu_conv(input_tensor, dim, ks, s,
                                                            ps=ps, padding='VALID', name='pred')
        pred_activate = tf.nn.sigmoid(pred, name='pred_activate')
        if flags is None:
            return pred_activate
        else:
            return tf.image.resize_bilinear(pred_activate, (flags.image_height, flags.image_width))


def reflect_pad_conv(input_tensor, output_dim, ks, s, ps, padding, name):
    with tf.variable_scope(name):
        image_pad = tf.pad(input_tensor, [[0, 0], [ps, ps], [ps, ps], [0, 0]], "REFLECT", name='pad')
        conv = conv2d(image_pad, output_dim, ks=ks, s=s, padding=padding, name='conv')
        return conv


def relu_reflect_pad_conv(input_tensor, output_dim, ks, s, ps, padding, name):
    with tf.variable_scope(name):
        relu = tf.nn.relu(input_tensor, name='relu')
        image_pad = tf.pad(relu, [[0, 0], [ps, ps], [ps, ps], [0, 0]], "REFLECT", name='pad')
        conv = conv2d(image_pad, output_dim, ks=ks, s=s, padding=padding, name='conv')
        return conv


def reflect_pad_instance_normalization_relu_conv(input_tensor, output_dim, ks, s, ps, padding, name):
    with tf.variable_scope(name):
        image_pad = tf.pad(input_tensor, [[0, 0], [ps, ps], [ps, ps], [0, 0]], "REFLECT", name='pad')
        normal = instance_normalization(image_pad, name='instance_normalization')
        relu = tf.nn.relu(normal, name='relu')
        conv = conv2d(relu, output_dim=output_dim, ks=ks, s=s, padding=padding, name='conv')
        return conv


def instance_normalization_relu_reflect_pad_conv(input_tensor, output_dim, ks, s, ps, padding, name):
    with tf.variable_scope(name):
        normal = instance_normalization(input_tensor, name='instance_normalization')
        relu = tf.nn.relu(normal, name='relu')
        relu_pad = tf.pad(relu, [[0, 0], [ps, ps], [ps, ps], [0, 0]], "REFLECT", name='pad')
        conv = conv2d(relu_pad, output_dim=output_dim, ks=ks, s=s, padding=padding, name='conv')
        return conv


def instance_normalization_relu_conv2d(input_tensor, output_dim, ks, s, padding, name):
    with tf.variable_scope(name):
        normal = instance_normalization(input_tensor, name='instance_normalization')
        relu = tf.nn.relu(normal, name='relu')
        conv = conv2d(relu, output_dim=output_dim, ks=ks, s=s, padding=padding, name='conv')
        return conv


def instance_normalization_lrelu_conv2d(input_tensor, output_dim, ks, s, padding, name):
    with tf.variable_scope(name):
        normal = instance_normalization(input_tensor, name='instance_normalization')
        leaky_relu = lrelu(normal, name='leaky_relu')
        conv = conv2d(leaky_relu, output_dim=output_dim, ks=ks, s=s, padding=padding, name='conv')
        return conv


def instance_normalization_relu_deconv2d(input_tensor, output_dim, ks, s, name):
    with tf.variable_scope(name):
        normal = instance_normalization(input_tensor, name='instance_normalization')
        relu = tf.nn.relu(normal, name='relu')
        deconv = deconv2d(relu, output_dim=output_dim, ks=ks, s=s, name='deconv')
        return deconv


def relu_instance_normalization_conv2d(input_tensor, output_dim, ks, s, padding, name):
    with tf.variable_scope(name):
        relu = tf.nn.relu(input_tensor, name='relu')
        normal = instance_normalization(relu, name='instance_normalization')
        conv = conv2d(normal, output_dim=output_dim, ks=ks, s=s, padding=padding, name='conv')
        return conv


def relu_instance_normalization_deconv2d(input_tensor, output_dim, ks, s, name):
    with tf.variable_scope(name):
        relu = tf.nn.relu(input_tensor, name='relu')
        normal = instance_normalization(relu, name='instance_normalization')
        deconv = deconv2d(normal, output_dim=output_dim, ks=ks, s=s, name='deconv')
        return deconv


def lrelu_instance_normalization_conv2d(input_tensor, output_dim, ks, s, padding, name):
    with tf.variable_scope(name):
        leaky_relu = lrelu(input_tensor, name='leaky_relu')
        normal = instance_normalization(leaky_relu, name='instance_normalization')
        conv = conv2d(normal, output_dim=output_dim, ks=ks, s=s, padding=padding, name='conv')
        return conv


def conv2d_lrelu(input_tensor, output_dim, ks, s, padding, name):
    with tf.variable_scope(name):
        conv = conv2d(input_tensor, output_dim=output_dim, ks=ks, s=s, padding=padding, name='conv')
        leaky_relu = lrelu(conv, name='leaky_relu')
        return leaky_relu


def lrelu_conv2d(input_tensor, output_dim, ks, s, padding, name):
    with tf.variable_scope(name):
        leaky_relu = lrelu(input_tensor, name='leaky_relu')
        conv = conv2d(leaky_relu, output_dim=output_dim, ks=ks, s=s, padding=padding, name='conv')
        return conv

"""
Basic operations
"""


def image_normalization_one(image, name='image_normalization_one'):
    with tf.variable_scope(name):
        return (image / 127.5) - 1.0


def image_normalization_sub(image, name='image_normalization_sub'):
    with tf.variable_scope(name):
        return image - 127.5


def get_shape(tensor):
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0]
            for s in zip(static_shape, dynamic_shape)]
    return dims


def fully_connected(x, units, name='fully_connected'):
    return tf.layers.dense(inputs=x, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           units=units, name=name)


def global_average_pooling(x, name):
    return tf.reduce_mean(input_tensor=x, axis=[1, 2], keep_dims=True, name=name)


def skip_squeeze_excitation_layer(x, dim, name):
    with tf.variable_scope(name):
        skip = instance_normalization_relu_conv2d(x, dim, ks=3, s=1, padding='SAME', name='skip')
        residual = residule_block_rule(skip, dim=dim, name='residual')
        scale = squeeze_excitation_layer(residual, ratio=16, name='scale')
        return scale


def skip_combine(x, y, dim, name):
    with tf.variable_scope(name):
        x_conv = instance_normalization_relu_deconv2d(x, dim, 3, 2, name='x_conv')
        y_skip = instance_normalization_relu_conv2d(y, dim, ks=3, s=1, padding='SAME', name='y_skip')
        y_residual = residule_block_why(y_skip, dim=dim, name='y_residual')
        y_scale = squeeze_excitation_layer(y_residual, ratio=16, name='y_scale')
        fuse = tf.add(x_conv, y_scale, name='fuse')
        return fuse


def skip_zero(x, dim, name):
    with tf.variable_scope(name):
        skip = instance_normalization_relu_conv2d(x, dim, ks=3, s=1, padding='SAME', name='skip')
        residual = residule_block_zero(skip, dim=dim, name='residual')
        return residual


def skip_concat(x, y, dim, name):
    with tf.variable_scope(name):
        skip = skip_zero(x, dim=dim, name='skip')
        concat = tf.concat([y, skip], axis=3, name='concat')
        return concat


def squeeze_excitation_layer(x, ratio, name):
    with tf.variable_scope(name):
        out_dim = get_shape(x)[3]
        squeeze = global_average_pooling(x, name='squeeze')
        excitation = fully_connected(squeeze, units=out_dim / ratio, name='fully_connected1')
        excitation = tf.nn.relu(excitation, name='relu')
        excitation = fully_connected(excitation, units=out_dim, name='fully_connected2')
        excitation = tf.nn.sigmoid(excitation, name='sigmoid')
        scale = tf.multiply(x, excitation, name='scale')
        return scale


def instance_normalization_relu_squeeze_excitation_layer(x, ratio, name):
        with tf.variable_scope(name):
            out_dim = get_shape(x)[3]
            normal = instance_normalization(x, name='instance_normalization')
            relu = tf.nn.relu(normal, name='relu')
            squeeze = global_average_pooling(relu, name='squeeze')
            excitation = fully_connected(squeeze, units=out_dim / ratio, name='fully_connected1')
            excitation = tf.nn.relu(excitation, name='relu')
            excitation = fully_connected(excitation, units=out_dim, name='fully_connected2')
            excitation = tf.nn.sigmoid(excitation, name='sigmoid')
            scale = tf.multiply(x, excitation, name='scale')
            return scale


def residule_block_se(x, dim, ks=3, s=1, name='res_se'):
    with tf.variable_scope(name):
        p = int((ks - 1) / 2)
        y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y = instance_normalization(conv2d(y, dim, ks, s, padding='VALID', name='_c1'), name='_bn1')
        y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y = instance_normalization(conv2d(y, dim, ks, s, padding='VALID', name='_c2'), name='_bn2')
        scale = squeeze_excitation_layer(y, 16, 'squeeze_excitation_layer')
        residule_out = tf.add(x, scale, name='residule_out')
    return residule_out


def residule_block_se_fixed(x, dim, ks=3, s=1, name='res_se'):
    with tf.variable_scope(name):
        p = int((ks - 1) / 2)
        x_pad = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y1 = relu_instance_normalization_conv2d(x_pad, dim, ks, s, padding='VALID', name='y1')
        y1_pad = tf.pad(tf.nn.relu(y1), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y2 = relu_instance_normalization_conv2d(y1_pad, dim, ks, s, padding='VALID', name='y2')
        scale = squeeze_excitation_layer(y2, 16, name='squeeze_excitation_layer')
        residule_out = tf.add(x, scale, name='residule_out')
    return residule_out


def residule_block_rule(x, dim, ks=3, s=1, name='res_se'):
    with tf.variable_scope(name):
        p = int((ks - 1) / 2)
        y1 = reflect_pad_instance_normalization_relu_conv(x, dim, ks, s, ps=p, padding='VALID', name='y1')
        y2 = reflect_pad_instance_normalization_relu_conv(y1, dim, ks, s, ps=p, padding='VALID', name='y2')
        scale = squeeze_excitation_layer(y2, 16, name='squeeze_excitation_layer')
        residule_out = tf.add(x, scale, name='residule_out')
    return residule_out


def residule_block_zero(x, dim, ks=3, s=1, name='res_se'):
    with tf.variable_scope(name):
        p = int((ks - 1) / 2)
        y1 = instance_normalization_relu_reflect_pad_conv(x, dim, ks, s, ps=p, padding='VALID', name='y1')
        y2 = instance_normalization_relu_reflect_pad_conv(y1, dim, ks, s, ps=p, padding='VALID', name='y2')
        residule_out = tf.add(x, y2, name='residule_out')
    return residule_out


def residule_block_why(x, dim, ks=3, s=1, name='res_se'):
    with tf.variable_scope(name):
        p = int((ks - 1) / 2)
        y1 = instance_normalization_relu_reflect_pad_conv(x, dim, ks, s, ps=p, padding='VALID', name='y1')
        y2 = instance_normalization_relu_reflect_pad_conv(y1, dim, ks, s, ps=p, padding='VALID', name='y2')
        scale = instance_normalization_relu_squeeze_excitation_layer(y2, 16, name='squeeze_excitation_layer')
        residule_out = tf.add(x, scale, name='residule_out')
    return residule_out


def residule_block_se_fixed_reverse(x, dim, ks=3, s=1, name='res_se'):
    with tf.variable_scope(name):
        p = int((ks - 1) / 2)
        x_pad = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y1 = instance_normalization_relu_conv2d(x_pad, dim, ks, s, padding='VALID', name='y1')
        y1_pad = tf.pad(tf.nn.relu(y1), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y2 = instance_normalization_relu_conv2d(y1_pad, dim, ks, s, padding='VALID', name='y2')
        scale = squeeze_excitation_layer(y2, 16, name='squeeze_excitation_layer')
        residule_out = tf.add(x, scale, name='residule_out')
    return residule_out


def residule_block(x, dim, ks=3, s=1, name='res'):
    with tf.variable_scope(name):
        p = int((ks - 1) / 2)
        y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y = instance_normalization(conv2d(y, dim, ks, s, padding='VALID', name='_c1'), name='_bn1')
        y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y = instance_normalization(conv2d(y, dim, ks, s, padding='VALID', name='_c2'), name='_bn2')
    return y + x


def instance_normalization(input_tensor, name="instance_normalization"):
    with tf.variable_scope(name):
        depth = input_tensor.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input_tensor, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input_tensor-mean)*inv
        return scale*normalized + offset


def conv2d(input_tensor, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        # return slim.conv2d(input_tensor, output_dim, ks, s, padding=padding, biases_initializer=None,
        #                    activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=stddev))
        return tf.layers.conv2d(input_tensor, output_dim, ks, s, padding=padding, bias_initializer=None,
                                activation=None, kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))


def deconv2d(input_tensor, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        # return slim.conv2d_transpose(input_tensor, output_dim, ks, s, padding='SAME', activation_fn=None,
        #                              weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
        #                              biases_initializer=None)
        return tf.layers.conv2d_transpose(input_tensor, output_dim, ks, s, padding='SAME', activation=None,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                          bias_initializer=None)


def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        return tf.maximum(x, leak*x, name=name)


def linear(input_tensor, output_size, name=None, stddev=0.02, bias_start=0.0, with_w=False):
    with tf.variable_scope(name or "Linear"):
        matrix = tf.get_variable("Matrix", [input_tensor.get_shape()[-1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_tensor, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_tensor, matrix) + bias


def expand_conv(image_slice, init, k, name):
    with tf.variable_scope(name):
        image_expand = tf.expand_dims(image_slice, axis=3)
        image_blur = tf.layers.conv2d(image_expand, 1, k, 1, padding='same', bias_initializer=None,
                                      trainable=False, activation=None, kernel_initializer=init)
        return image_blur
