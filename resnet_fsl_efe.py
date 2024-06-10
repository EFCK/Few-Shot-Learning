import tensorflow as tf

from keras.models import Model
from keras.layers import Layer
from keras.layers import (GlobalAveragePooling2D, Activation, MaxPooling2D, Add, Conv2D, MaxPool2D, Dense,
                                     Flatten, InputLayer, BatchNormalization, Input, Embedding, Permute,
                                     Dropout, RandomFlip, RandomRotation, LayerNormalization, MultiHeadAttention,
                                     RandomContrast, Rescaling, Resizing, Reshape)

class MyConv2D(Layer):
    """
    MyConv2D is a custom layer that combines a Conv2D layer and adds BatchNormalization layer.
    """
    def __init__(self, in_n_filters, in_kernel_size, in_n_strides, in_padding="valid"):
        super(MyConv2D, self).__init__()

        # create a custom conv2d layer
        self.conv = Conv2D(
            filters=in_n_filters,
            kernel_size=in_kernel_size,
            strides=in_n_strides,
            padding=in_padding,
        )

        # create a batch normalization layer
        self.bn = BatchNormalization()

    def call(self, in_x, training=False):
        """
        Call method of the custom layer. it applies the convolution and batch normalization to the input data.
        """
        x = self.conv(in_x)
        x = self.bn(x, training=training)
        return x

class ResidualBlock(Layer):
    def __init__(self, in_n_filters, in_n_strides=1):
        super(ResidualBlock, self).__init__()

        # check if the input data will be down-sized
        self.down_sized = (in_n_strides != 1)

        # create custom conv2d layers
        self.conv_1 = MyConv2D(in_n_filters, 3, in_n_strides, in_padding="same")
        self.conv_2 = MyConv2D(in_n_filters, 3, 1, in_padding="same")

        self.activation = Activation("relu")

        # create a conv2d layer for down-sizing the input data
        if self.down_sized:
            self.conv_3 = MyConv2D(in_n_filters, 1, in_n_strides)

    def call(self, input, training=False):
        x = self.conv_1(input, training=training)
        x = self.conv_2(x, training=training)

        if self.down_sized:
            x_down = self.conv_3(input, training=training)
            x_final = Add()([x, x_down])
        else:
            x_final = Add()([x, input])

        return self.activation(x_final)

class ResNet18_fsl(Model):
    def __init__(self):
        super(ResNet18_fsl, self).__init__()

        # first layer
        self.conv_1 = MyConv2D(64, 7, 2, in_padding="same")
        self.max_pool = MaxPool2D(pool_size=3, strides=2)

        # residual layer 1
        self.res_1_1 = ResidualBlock(64)
        self.res_1_2 = ResidualBlock(64)

        # residual layer 2
        self.res_2_1 = ResidualBlock(128, 2)
        self.res_2_2 = ResidualBlock(128)

        # residual layer 3
        self.res_3_1 = ResidualBlock(256, 2)
        self.res_3_2 = ResidualBlock(256)

        # residual layer 4
        self.res_4_1 = ResidualBlock(512, 2)
        self.res_4_2 = ResidualBlock(512)

        # global average pooling
        self.global_pool = GlobalAveragePooling2D()

    def call(self, inputs, training=False):
        x = self.conv_1(inputs, training=training)
        x = self.max_pool(x)

        x = self.res_1_1(x, training=training)
        x = self.res_1_2(x, training=training)

        x = self.res_2_1(x, training=training)
        x = self.res_2_2(x, training=training)

        x = self.res_3_1(x, training=training)
        x = self.res_3_2(x, training=training)

        x = self.res_4_1(x, training=training)
        x = self.res_4_2(x, training=training)

        x = self.global_pool(x)

        return x