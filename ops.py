import tensorflow as tf
"""
Combinations
"""


def lrelu_instance_normalization_conv2d(input_tensor, name):
    return input_tensor
"""
Basic operations
"""


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
