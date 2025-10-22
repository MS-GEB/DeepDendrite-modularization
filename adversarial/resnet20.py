import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
np.random.seed(1)
tf.random.set_seed(1)


def residual_block(inputs, filter_num, stride, l2_reg):
    x = layers.BatchNormalization()(inputs)
    x = layers.ReLU()(x)
    if stride != 1:
        shortcut = layers.Conv2D(filters=filter_num,
                                 kernel_size=(1, 1),
                                 strides=stride,
                                 use_bias=False,
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
        # residual = layers.BatchNormalization()(residual)
    else:
        shortcut = inputs

    x = layers.Conv2D(filters=filter_num,
                      kernel_size=(3, 3),
                      strides=stride,
                      padding='same',
                      kernel_initializer='he_normal',
                      use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters=filter_num,
                      kernel_size=(3, 3),
                      strides=1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=keras.regularizers.l2(l2_reg))(x)

    output = layers.Add()([x, shortcut])

    return output


def make_resnet(input_shape, num_classes, n, l2_reg):
    inputs = keras.Input(shape=input_shape, name='img')
    x = layers.Conv2D(filters=16,
                      kernel_size=(3, 3),
                      strides=1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=keras.regularizers.l2(l2_reg))(inputs)
    # x = layers.BatchNormalization()(x)
    # x = layers.ReLU()(x)

    for _ in range(n):
        x = residual_block(x, 16, 1, l2_reg)

    x = residual_block(x, 32, 2, l2_reg)
    for _ in range(n - 1):
        x = residual_block(x, 32, 1, l2_reg)

    x = residual_block(x, 64, 2, l2_reg)
    for _ in range(n - 1):
        x = residual_block(x, 64, 1, l2_reg)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)

    outputs = layers.Dense(num_classes,
                           activation='softmax',
                           kernel_regularizer=keras.regularizers.l2(l2_reg))(x)

    model = keras.Model(inputs, outputs, name='resnet')

    return model


class DataAug(keras.layers.Layer):
    def __init__(self, pad_size):
        super(DataAug, self).__init__()
        self.pad_size = pad_size
        self.rand_flip = keras.layers.experimental.preprocessing.RandomFlip("horizontal")
        self.rand_crop = keras.layers.experimental.preprocessing.RandomCrop(32, 32)

    def build(self, input_shape):
        #super(DataAug, self).build(input_shape)
        self.in_shape = input_shape

    def call(self, inputs, training = None):
        if training is True:
            x = tf.image.resize_with_crop_or_pad(inputs, 40, 40)
            x = self.rand_flip(x)
            x = self.rand_crop(x)
            return x
        else:
            return inputs


if __name__ == '__main__':
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    y_train = y_train.astype('int32')
    x_test = x_test.astype('float32') / 255
    y_test = y_test.astype('int32')
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print(x_train.shape)
    batch_size = 64

    model = make_resnet(x_train.shape[1:], 10, 3, 0.0001)
    # model = make_cnn()
    # data_augmentation.summary()
    model.summary()
    optim = keras.optimizers.SGD(0.1, momentum=0.9, nesterov=True)
    # optim = keras.optimizers.Adam(0.01)
    # tsb_callback = keras.callbacks.TensorBoard('./logs', 1)
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=optim, metrics=["acc"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=60, validation_data=(x_test, y_test))
    model.evaluate(x_test, y_test)
    model.save_weights("resnet20_mnist_60epochs_sgd01.h5")
