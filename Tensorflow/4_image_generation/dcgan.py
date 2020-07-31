import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers

import matplotlib.pyplot as plt
import numpy as np
import time
import os

BATCH_SIZE = 128
BUFFER_SIZE = 60000
EPOCHS = 20
NOISE_DIM = 100
NGF = 64
NDF = 64
NUM_EXAMPLES = 64
ALPHA = 2e-4
BEAT1 = 0.5

# load mnist dataset
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
SAMPLES = train_images.shape[0]
train_images = train_images.reshape(SAMPLES, 28, 28, 1).astype('float32')
# image process
train_images = (train_images - 127.5) / 255
train_images = tf.image.resize(train_images, [64, 64])

# 批量化和打乱数据
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# build models: generator and discriminator
conv_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
bn_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# bn中，默认 beta_initializer="zeros"
# bn_init_b = tf.keras.initializers.Zeros()

def make_generator_model():
    model = tf.keras.Sequential()

    model.add(layers.Conv2DTranspose(NGF*8, (4,4), strides=(1,1), padding='valid', use_bias=False, input_shape=(1, 1, NOISE_DIM)))
    model.add(layers.BatchNormalization(gamma_initializer=bn_init))
    model.add(layers.ReLU())
    
    model.add(layers.Conv2DTranspose(NGF*4, (4,4), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization(gamma_initializer=bn_init))
    model.add(layers.ReLU())
    
    model.add(layers.Conv2DTranspose(NGF*2, (4,4), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization(gamma_initializer=bn_init))
    model.add(layers.ReLU())
    
    model.add(layers.Conv2DTranspose(NGF, (4,4), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization(gamma_initializer=bn_init))
    model.add(layers.ReLU())
    
    model.add(layers.Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', use_bias=False, activation='tanh'))
    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(NDF, (4,4), strides=(2,2), padding='same', use_bias=False, kernel_initializer=conv_init, input_shape=(64, 64, 1)))
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv2D(NDF*2, (4,4), strides=(2,2), padding='same', use_bias=False, kernel_initializer=conv_init))
    model.add(layers.BatchNormalization(gamma_initializer=bn_init))
    model.add(layers.LeakyReLU(0.2))
    
    model.add(layers.Conv2D(NDF*4, (4,4), strides=(2,2), padding='same', use_bias=False, kernel_initializer=conv_init))
    model.add(layers.BatchNormalization(gamma_initializer=bn_init))
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv2D(NDF*8, (4,4), strides=(2,2), padding='same', use_bias=False, kernel_initializer=conv_init))
    model.add(layers.BatchNormalization(gamma_initializer=bn_init))
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv2D(1, (4,4), strides=(1,1), padding='valid', use_bias=False, kernel_initializer=conv_init, activation='sigmoid'))
    return model

generator = make_generator_model()
generator.summary()
discriminator = make_discriminator_model()
discriminator.summary()

# loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# optimizers
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA, beta_1=BEAT1)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA, beta_1=BEAT1)

# add checkpoints
checkpoint_dir = 'checkpoints/dcgan'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator)

def generate_and_save_images(model, epoch, test_input):
  # 注意 training` 设定为 False
    predictions = model(test_input, training=False)
    plt.figure(figsize=(8, 8))

    for i in range(predictions.shape[0]):
        plt.subplot(8, 8, i+1)
        plt.imshow(predictions[i, :, :, 0] * 255 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('pics/dcgan/image_at_epoch_{:02d}.png'.format(epoch))

# define training
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 1, 1, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = tf.reshape(discriminator(images, training=True), [-1])
      fake_output = tf.reshape(discriminator(generated_images, training=True), [-1])

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

seed = tf.random.normal([NUM_EXAMPLES, 1, 1, NOISE_DIM])

def train(dataset, epochs):
    for epoch in range(epochs):
        gen_losses = []
        disc_losses = []
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            gen_losses.append(gen_loss.numpy())
            disc_losses.append(disc_loss.numpy())
        # 每5个epoch保存一次模型，生成一次图片
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            generate_and_save_images(generator, epoch+1, seed)
        # 打印损失函数
        print('Epoch %d: generator loss, %.3f, discriminator loss, %.3f' % (
            epoch+1, sum(gen_losses)/len(gen_losses), sum(disc_losses)/len(disc_losses)))
        
start = time.time()
train(train_dataset, EPOCHS)
print('Total training time: %.3f mins' % ((time.time()-start)/60))