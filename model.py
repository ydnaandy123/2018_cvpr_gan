from ops import *


class VAE:
    def __init__(self, input_x, training, dropout_rate, class_num):
        self.training = training
        self.dropout_rate = dropout_rate
        self.class_num = class_num
        self.model = self.vae_net(input_x)

    def vae_net(self, input_x):
        x_normal = (input_x / 127.5) - 1.0

        """ en0 448x448"""
        en0 = conv2d(input_tensor=x_normal, filters=32, name='en0')

        en1 = pool2d_relu_batch_conv(input_tensor=en0, filters=64, is_training=self.training, name='en1')


        return en1