import tensorflow as tf
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import time

from tensorflow_examples.models.pix2pix import pix2pix
AUTOTUNE = tf.data.experimental.AUTOTUNE

EPOCHS = 100
ALPHA = 2e-4
BETA = 0.5
REGTERM = 10
BATCH_SIZE = 8
BUFFER_SIZE = 1000
IMG_WIDTH = 256
IMG_HEIGHT = 256
CHANNELS = 3

# load dataset
dataset, metadata = tfds.load('cycle_gan/horse2zebra', with_info=True, as_supervised=True)

train_horses, train_zebras = dataset['trainA'], dataset['trainB']
test_horses, test_zebras = dataset['testA'], dataset['testB']

# image transforms

# normalize to [-0.5, 0.5]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 255) - 0.5
  return image

def random_crop(image):
  cropped_img = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
  return cropped_img

def random_jitter(image):
  image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  image = random_crop(image)
  image = tf.image.random_flip_left_right(image)
  return image

def process_train(image, label):
  image = random_jitter(image)
  image = normalize(image)
  return image

def process_test(image, label):
  image = normalize(image)
  return image

train_horses = train_horses.map(process_train, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_zebras = train_zebras.map(process_train, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_horses = test_horses.map(process_test, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_zebras = test_zebras.map(process_test, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

sample_horse = next(iter(train_horses))
sample_zebra = next(iter(train_zebras))


# import model from pix2pix
# generator gen_g: x -> y
gen_g = pix2pix.unet_generator(CHANNELS, norm_type='instancenorm')
# generator gen_f: y -> x
gen_f = pix2pix.unet_generator(CHANNELS, norm_type='instancenorm')
# discriminator x: real x or generated x
disc_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
# discriminator y: real y or generated y
disc_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

# loss function
loss_f = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, fake):
  total_loss = loss_f(tf.ones_like(real), real) + loss_f(tf.zeros_like(fake), fake)
  return total_loss*0.5

def generator_loss(fake):
  return loss_f(tf.ones_like(fake), fake)

LAMBDA = 10

def cycle_loss(real, cycled):
  c_loss = tf.reduce_mean(tf.abs(real - cycled))
  return LAMBDA*c_loss

def identity_loss(real, same):
  i_loss = tf.reduce_mean(tf.abs(real - same))
  return LAMBDA*0.5*i_loss

# set optimizers for generators and discriminators
gen_g_opt = tf.keras.optimizers.Adam(lr=ALPHA, beta_1=BETA)
gen_f_opt = tf.keras.optimizers.Adam(lr=ALPHA, beta_1=BETA)

disc_x_opt = tf.keras.optimizers.Adam(lr=ALPHA, beta_1=BETA)
disc_y_opt = tf.keras.optimizers.Adam(lr=ALPHA, beta_1=BETA)

# training
@tf.function
def train_step(real_x, real_y):
  with tf.GradientTape(persistent=True) as tape:
    fake_y = gen_g(real_x, training=True)
    cycled_x = gen_f(fake_y, training=True)
    
    fake_x = gen_f(real_y, training=True)
    cycled_y = gen_g(fake_x, training=True)
    
    same_y = gen_g(real_y, training=True)
    same_x = gen_f(real_x, training=True)

    disc_real_x = disc_x(real_x, training=True)
    disc_real_y = disc_y(real_y, training=True)
    
    disc_fake_x = disc_x(fake_x, training=True)
    disc_fake_y = disc_y(fake_y, training=True)

    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)

    total_cycle_loss = cycle_loss(real_x, cycled_x) + cycle_loss(real_y, cycled_y)
    total_g_loss = total_cycle_loss + gen_g_loss + identity_loss(real_y, same_y)
    total_f_loss = total_cycle_loss + gen_f_loss + identity_loss(real_x, same_x)

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

  gen_g_grads = tape.gradient(total_g_loss, gen_g.trainable_variables)
  gen_f_grads = tape.gradient(total_f_loss, gen_f.trainable_variables)
  disc_x_grads = tape.gradient(disc_x_loss, disc_x.trainable_variables)
  disc_y_grads = tape.gradient(disc_y_loss, disc_y.trainable_variables)

  gen_g_opt.apply_gradients(zip(gen_g_grads, gen_g.trainable_variables))
  gen_f_opt.apply_gradients(zip(gen_f_grads, gen_f.trainable_variables))
  disc_x_opt.apply_gradients(zip(disc_x_grads, disc_x.trainable_variables))
  disc_y_opt.apply_gradients(zip(disc_y_grads, disc_y.trainable_variables))

  return total_g_loss, total_f_loss, disc_x_loss, disc_y_loss


start = time.time()

for epoch in range(EPOCHS):
    gloss = 0
    floss = 0
    dxloss = 0
    dyloss = 0
    for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
        total_g_loss, total_f_loss, disc_x_loss, disc_y_loss = train_step(image_x, image_y)
        gloss += total_g_loss
        floss += total_f_loss
        dxloss += disc_x_loss
        dyloss += disc_y_loss
    print('Epoch %d: generator losses: %.3f, %.3f; discriminator losses: %.3f, %.3f' % (epoch+1, gloss, floss, dxloss, dyloss))

print('Toal training time: %.3f' % (time.time()-start))
'''
gen_g.save('models/dlf5_gg.h5')
gen_f.save('models/dlf5_gf.h5')
disc_x.save('models/dlf5_dx.h5')
disc_y.save('models/dlf5_dy.h5')
'''
# test model
def generate_images(model, input, idx):
    prediction = model(input)
    plt.figure(figsize=(12, 8))

    display_list = [input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] + 0.5)
        plt.axis('off')
    
    plt.savefig('pics/cyclegan/image_{:02d}.png'.format(idx))

idx = 1
for inp in test_horses.take(5):
    generate_images(gen_g, inp, idx)
    idx += 1

for inp in test_zebras.take(5):
    generate_images(gen_f, inp, idx)
    idx += 1