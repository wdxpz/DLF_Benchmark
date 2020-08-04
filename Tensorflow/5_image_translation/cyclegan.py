import time
import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE

dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                              with_info=True, as_supervised=True)

train_horses, train_zebras = dataset['trainA'], dataset['trainB']
test_horses, test_zebras = dataset['testA'], dataset['testB']


BUFFER_SIZE = 1000
# BATCH_SIZE = 4
BATCH_SIZE = 8
IMG_WIDTH = 256
IMG_HEIGHT = 256


def random_crop(image):
    cropped_image = tf.image.random_crop(
        image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image


# normalizing the images to [-1, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def random_jitter(image):
    # resizing to 286 x 286 x 3
    image = tf.image.resize(image, [286, 286],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # randomly cropping to 256 x 256 x 3
    image = random_crop(image)

    # random mirroring
    image = tf.image.random_flip_left_right(image)

    return image


def random_jitter(image):
    # resizing to 286 x 286 x 3
    image = tf.image.resize(image, [286, 286],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # randomly cropping to 256 x 256 x 3
    image = random_crop(image)

    # random mirroring
    image = tf.image.random_flip_left_right(image)

    return image


def preprocess_image_train(image, label):
    image = random_jitter(image)
    image = normalize(image)
    return image


def preprocess_image_test(image, label):
    image = normalize(image)
    return image


train_horses = train_horses.map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

train_zebras = train_zebras.map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_horses = test_horses.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_zebras = test_zebras.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)


OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)


LAMBDA = 10

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)

    generated_loss = loss_obj(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1


def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss



boundaries = [50, 150]
values = [0.0002, 0.00002, 0.000002]

learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)


generator_g_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn, beta_1=0.5)


# generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
# generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
# discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('*** Latest checkpoint restored!! ***')


@tf.function
def train_step(real_x, real_y):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.

        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(
            real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + \
            total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + \
            total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss,
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss,
                                          generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss,
                                              discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss,
                                              discriminator_y.trainable_variables)

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                              generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                              generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                  discriminator_x.trainable_variables))

    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                  discriminator_y.trainable_variables))



def generate_images(model, test_input, name):
    prediction = model(test_input)
    display_list = [test_input[0], prediction[0]]
    plt.imshow(test_input[0] * 0.5 + 0.5)
    plt.axis('off')
    plt.savefig("{}.png".format(name + "_org"), bbox_inches='tight')
    # plt.title('Horse')
    plt.imshow(prediction[0] * 0.5 + 0.5)
    plt.axis('off')
    plt.savefig("{}.png".format(name), bbox_inches='tight')


if __name__ == "__main__":
    args = sys.argv
    print(args)
    if len(args) == 1:
        print("please specify train / test")
        exit(0)
    else:
        purpose = args[1]
        if purpose.lower() == 'train':
            #  training process
            print('training...')
            EPOCHS = 200
            for epoch in range(EPOCHS):
                print("Epoch {} / {} ...".format(epoch + 1, EPOCHS))
                start = time.time()

                n = 0
                for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
                    train_step(image_x, image_y)
                    if n % 10 == 0:
                        print('.', end='')
                    n += 1
                if (epoch + 1) % 5 == 0:
                    ckpt_save_path = ckpt_manager.save()
                    print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                        ckpt_save_path))

                print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                                time.time()-start))
            print("*****testing horse to zebra*******")
            TEST_DIR = "test_data/A"
            IMG_EXT = ["jpg", "jpeg", "png"]
            images = os.listdir(TEST_DIR)
            images = [img for img in images if img.split('.')[-1].lower() in IMG_EXT]
            for i, img in enumerate(images, 1):
                img_path = os.path.join(TEST_DIR, img)
                # print(img_path)
                image_bytes = open(img_path, 'rb')
                image = tf.io.decode_jpeg(image_bytes.read(), channels=3)
                image = tf.expand_dims(image, 0)
                image = preprocess_image_test(image, None)
                # print(image.shape)
                generate_images(generator_g, image, "horse_" + str(i))
            
            print("testing zebra to horse")
            TEST_DIR = "test_data/B"
            IMG_EXT = ["jpg", "jpeg", "png"]
            images = os.listdir(TEST_DIR)
            images = [img for img in images if img.split('.')[-1].lower() in IMG_EXT]
            for i, img in enumerate(images, 1):
                img_path = os.path.join(TEST_DIR, img)
                # print(img_path)
                image_bytes = open(img_path, 'rb')
                image = tf.io.decode_jpeg(image_bytes.read(), channels=3)
                image = tf.expand_dims(image, 0)
                image = preprocess_image_test(image, None)
                # print(image.shape)
                generate_images(generator_f, image, "zebra_" + str(i))
        elif purpose.lower() == 'test':
            # testing process
            print("testing ....")
            TEST_DIR = "test_data/B"
            IMG_EXT = ["jpg", "jpeg", "png"]
            images = os.listdir(TEST_DIR)
            images = [img for img in images if img.split('.')[-1].lower() in IMG_EXT]
            for i, img in enumerate(images, 1):
                img_path = os.path.join(TEST_DIR, img)
                # print(img_path)
                image_bytes = open(img_path, 'rb')
                image = tf.io.decode_jpeg(image_bytes.read(), channels=3)
                image = tf.expand_dims(image, 0)
                image = preprocess_image_test(image, None)
                # print(image.shape)
                generate_images(generator_g, image, "horse_" + str(i))
            
        else:
            print("invalid param!")
            exit(0)