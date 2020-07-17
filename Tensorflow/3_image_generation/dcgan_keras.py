# keras实现，训练效果不好，超参数可能需要调整

import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time

NOISE_DIM = 100
WIDTH = 28
HEIGHT = 28
CHANNELS = 1
EPOCHS = 20
BATCH_SIZE = 64
ALPHA = 2e-4

# build generator
def build_generator():
    generator_input = keras.Input(shape=(NOISE_DIM,))
    x = layers.Dense(256*7*7)(generator_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Reshape((7, 7, 256))(x)

    x = layers.Conv2DTranspose(128, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, 5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(1, 5, strides=2, padding='same', activation='tanh')(x)
    x = layers.LeakyReLU()(x)

    generator = keras.models.Model(generator_input, x)
    return generator

# build discriminator
def build_discriminator():
    discriminator_input = layers.Input(shape=(HEIGHT, WIDTH, CHANNELS))
    x = layers.Conv2D(64, 5, strides=2, padding='same')(discriminator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, 5, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    discriminator = keras.models.Model(discriminator_input, x)
    return discriminator

generator = build_generator()
generator.summary()

discriminator = build_discriminator()
discriminator.summary()

# discriminator_optimizer = keras.optimizers.Adam(lr=ALPHA)
discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
discriminator.compile(
    optimizer=discriminator_optimizer,
    loss='binary_crossentropy'
)

# build gan network
discriminator.trainable = False
gan_input = keras.Input(shape=(NOISE_DIM, ))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)

# gan_optimizer = keras.optimizers.Adam(learning_rate=ALPHA)
gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(
    optimizer=gan_optimizer,
    loss='binary_crossentropy'
)


# load dataset mnist
(train_images, train_labels), (_, _) = keras.datasets.mnist.load_data()
# SAMPLES = train_images.shape[0]
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 255 # 将图片标准化到 [-0.5, 0.5] 区间内
# choose a number, 5
train_x = train_images[train_labels.flatten() == 5]
SAMPLES = train_x.shape[0]

seed = np.random.normal(size=(BATCH_SIZE, NOISE_DIM))

def generate_and_save_images(model, epoch):
    predictions = model.predict(seed)
    plt.figure(figsize = (8, 8))
    for i in range(predictions.shape[0]):
        plt.subplot(8, 8, i+1)
        plt.imshow(predictions[i, :, :, 0]*255+127.5, cmap='gray')
        plt.axis('off')
    plt.savefig('pics/image_at_epoch_{:02d}.png'.format(epoch))

# iterations of batches in an epoch
ITRS = SAMPLES // BATCH_SIZE

for ep in range(EPOCHS):
    start = 0
    total_a_loss = 0
    total_d_loss = 0
    
    for step in range(ITRS):
        random_vector = np.random.normal(size=(BATCH_SIZE, NOISE_DIM))
        generated_images = generator.predict(random_vector)
        stop = start + BATCH_SIZE
        real_images = train_images[start: stop]
        combined_images = np.concatenate([generated_images, real_images])

        labels = np.concatenate([np.ones((BATCH_SIZE, 1)), np.zeros((BATCH_SIZE, 1))])
        labels += 0.05 * np.random.random(labels.shape)

        # train discriminator
        d_loss = discriminator.train_on_batch(combined_images, labels)
        total_d_loss += d_loss

        random_vector = np.random.normal(size=(BATCH_SIZE, NOISE_DIM))
        misleading_targets = np.zeros((BATCH_SIZE, 1))  # lie about true images
        
        # train generator
        a_loss = gan.train_on_batch(random_vector, misleading_targets)
        total_a_loss += a_loss

        start += BATCH_SIZE

        '''
        if step % 100 == 0:
            gan.save_weights('models/gan.h5')
            print('discrimination loss: ', d_loss)
            print('gan loss: ', a_loss)
        '''
    
    # print losses
    print('discrimination loss: %.5f' % (total_d_loss/SAMPLES))
    print('gan loss: %.5f' % (total_a_loss/SAMPLES))
    # save a figure every epoch
    generate_and_save_images(generator, ep+1)