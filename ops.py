import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope
"""
Combination with default value
"""


def relu_batch_conv(input_tensor, filters, is_training, name, k_h=3, k_w=3, d_h=1, d_w=1):
    with tf.variable_scope(name):
        x_relu = relu(input_tensor=input_tensor)
        x_batch = batch_normalization(input_tensor=x_relu, training=is_training, scope='batch')
        x_conv = conv2d(input_tensor=x_batch, filters=filters, activation=None, name='conv2d',
                        k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w)
    return x_conv


def pool2d_relu_batch_conv(input_tensor, filters, is_training, name, k_h=3, k_w=3, d_h=1, d_w=1):
    with tf.variable_scope(name):
        x_pool = max_pooling(input_tensor=input_tensor)
        x_relu = relu(input_tensor=x_pool)
        x_batch = batch_normalization(input_tensor=x_relu, training=is_training, name='batch')
        x_conv = conv2d(input_tensor=x_batch, filters=filters, activation=None, name='conv2d',
                        k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w)
    return x_conv
"""
Convolution
"""


def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            biases_initializer=None)

def conv2d(input_tensor, filters, k_h=3, k_w=3, d_h=1, d_w=1, activation=None, name="conv2d"):
    conv = tf.layers.conv2d(inputs=input_tensor, filters=filters, kernel_size=[k_h, k_w],
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            strides=[d_h, d_w], padding='same', activation=activation, name=name)
    return conv


def relu(input_tensor):
    return tf.nn.relu(features=input_tensor)
"""
Generalize
"""


def batch_normalization(input_tensor, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=input_tensor, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=input_tensor, is_training=training, reuse=True))

    # batch = tf.layers.batch_normalization(inputs=input_tensor, training=training, name=name)
    # return batch


def drop_out(input_tensor, rate, training):
    return tf.layers.dropout(inputs=input_tensor, rate=rate, training=training)
"""
Pooling
"""


def average_pooling(input_tensor, pool_size_h=2, pool_size_w=2, stride_h=2, stride_w=2, padding='VALID'):
    return tf.layers.average_pooling2d(
        inputs=input_tensor, pool_size=[pool_size_h, pool_size_w], strides=[stride_h, stride_w], padding=padding)


def max_pooling(input_tensor, pool_size_h=2, pool_size_w=2, stride_h=2, stride_w=2, padding='VALID'):
    return tf.layers.max_pooling2d(
        inputs=input_tensor, pool_size=[pool_size_h, pool_size_w], strides=[stride_h, stride_w], padding=padding)


def global_average_pooling(input_tensor):
    return tf.reduce_mean(input_tensor=input_tensor, axis=[1, 2], keep_dims=True)
"""
Others
"""


def concatenation(layers):
    return tf.concat(layers, axis=3)


def flatten(input_tensor):
    dim = tf.reduce_prod(tf.shape(input_tensor)[1:])
    return tf.reshape(input_tensor, [-1, dim])


def linear(x, units):
    return tf.layers.dense(inputs=x, units=units, name='linear')


def get_shape(tensor):
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0]
            for s in zip(static_shape, dynamic_shape)]
    return dims