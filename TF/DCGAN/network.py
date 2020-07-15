
import tensorflow as tf
from tensorflow.keras import layers

from .config import DCGAN_Config as conf

def make_generator_model():
    model = tf.keras.Sequential()
    # model.add(layers.Dense(4*4*conf['ngf']*16, use_bias=False, input_shape=(100,)))
    # model.add(layers.BatchNormalization())
    # model.add(layers.ReLU())

    # model.add(layers.Reshape((4, 4, conf['ngf']*16)))
    # assert model.output_shape == (None, 4, 4, conf['ngf']*16) # 注意：batch size 没有限制

    # model.add(layers.Conv2DTranspose(conf['ngf']*8, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.Conv2DTranspose(64*8, (4, 4), strides=(4, 4), padding='same', use_bias=False, input_shape=(1, 1, 100)))
    assert model.output_shape == (None, 4, 4, conf['ngf']*8)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(conf['ngf']*4, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, conf['ngf']*4)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(conf['ngf']*2, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, conf['ngf']*2)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(conf['ngf'], (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, conf['ngf'])
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 64, 64, 1)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(conf['ndf'], (4, 4), strides=(2, 2), padding='same',
                                     input_shape=[64, 64, 1]))
    model.add(layers.LeakyReLU(0.2))
    # model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(conf['ndf']*2, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    # model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(conf['ndf']*4, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    # model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(conf['ndf']*8, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    # model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model