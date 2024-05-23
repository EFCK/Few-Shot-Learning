from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Add, Flatten, Dense, GlobalAveragePooling2D, Softmax
from keras.utils import set_random_seed


def residual_block(inputs, filters ,stride=1):
    set_random_seed(1)
    
    out = Conv2D(filters= filters, kernel_size=(3,3), padding='same', strides=stride)(inputs)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(filters= filters, kernel_size = (3,3), padding= 'same', strides=1)(out)
    out = BatchNormalization()(out)

    if stride != 1 or inputs.shape[-1] != filters:
        # fix input shape to match output shape
        shortcut = Conv2D(filters=filters, kernel_size=(1, 1), strides=stride, padding='same')(inputs)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = inputs
    
    x = Activation('relu')(out)
    x = Add()([x, shortcut])

    return x


def build_resnet(input_shape, num_classes):
    set_random_seed(1)
    
    inputs = Input(shape= input_shape)

    hidden = Conv2D(filters= 64, kernel_size= (25,25), strides= 10, padding= 'same')(inputs)
    hidden = Activation('relu')(hidden)
    hidden= BatchNormalization()(hidden)
    hidden = MaxPooling2D(pool_size=(3,3), padding= 'same', strides=2)(hidden)

    hidden = residual_block(hidden, 64, 2)
    hidden = residual_block(hidden, 64, 2)

    hidden = MaxPooling2D(pool_size= (3,3), strides= 2, padding= 'same')(hidden)
    hidden = residual_block(hidden, 64, 2)
    hidden = residual_block(hidden, 64, 2)

    hidden = GlobalAveragePooling2D()(hidden)

    hidden = Flatten()(hidden)
    hidden= Dense(units=num_classes, activation= 'sigmoid')(hidden)

    hidden = Softmax()(hidden)

    model = Model(inputs= inputs, outputs= hidden)

    return model
