import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import gridspec

from config import DCGAN_Config
from network import make_generator_model, make_discriminator_model 


# Set random seed for reproducibility
# seed = tf.random.normal([DCGAN_Config['num_examples_to_generate'], DCGAN_Config['nz']])
tf.random.set_seed(999)
seed = tf.random.normal([DCGAN_Config['num_examples_to_generate'], 1, 1, DCGAN_Config['nz']])
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_FILE = 'minst_dcgan_checkpoint_{}.pth'
RESULT_DIR = os.path.join(BASE_DIR, 'result')
RESULT_FILE = os.path.join(RESULT_DIR, 'minst_dcgan_result.txt')


class DCGAN_TF(object):
    def __init__(self, config=DCGAN_Config):
        self.image_size = config['image_size']
        self.workers = config['workers']
        self.nz = config['nz']
        self.num_epochs = config['num_epochs']
        self.lr = config['lr']
        self.beta1 = config['beta1']
        self.batch_size = config['batch_size']

        self.real_data_loader = self._get_real_data()
        self.generator, self.discriminator, self.generator_optimizer, self.discriminator_optimizer= self._init_network()


        checkpoint_dir = MODEL_DIR
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "tf_dcgan_ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                        discriminator_optimizer=self.discriminator_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)

    def _get_real_data(self):
        (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data(path=os.path.join(DATA_DIR, 'mnist.npz'))
        print('training set: {}'.format(len(train_images)))
        print('taining set size before resize: {}'.format(train_images.shape))
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        # train_images = tf.image.resize(train_images, [64, 64])
        # print('taining set size after resize: {}'.format(resized_train_images.shape))
        # train_images = (train_images - 127.5) / 127.5 # normalized to [-1, 1]
        #shuffle and batch
        BUFFER_SIZE=6000
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(self.batch_size)

        return train_dataset


    def _init_network(self):
        generator = make_generator_model()
        discriminator = make_discriminator_model()

        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta1)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta1)

        return generator, discriminator, generator_optimizer, discriminator_optimizer

    @staticmethod
    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    @staticmethod
    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    @staticmethod
    def generate_and_save_images(filename, model, epoch, test_input):
        # 注意 training` 设定为 False
        # 因此，所有层都在推理模式下运行（batchnorm）。
        predictions = model(test_input, training=False)

        nrow = 8
        ncol = 8

        fig = plt.figure(figsize=(ncol+1, nrow+1)) 
        gs = gridspec.GridSpec(nrow, ncol,
                wspace=0.0, hspace=0.0, 
                top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
                left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 

        for i in range(predictions.shape[0]):
            ax= plt.subplot(gs[i//8, i%8])
            ax.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        #     plt.subplot(8, 8, i+1)
        #     plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')    
        # plt.subplots_adjust(wspace=0, hspace=0)

        plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0)
        # plt.show()

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([DCGAN_Config['num_examples_to_generate'], 1, 1, DCGAN_Config['nz']])
        # noise = tf.random.normal([self.batch_size, self.nz])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        return gen_loss, disc_loss

    def train(self):
        G_losses=[]
        D_losses=[]
        start_time = time.time()
        for epoch in range(self.num_epochs):     
            i = 0

            for image_batch in self.real_data_loader:
                i += 1
                image_batch = tf.image.resize(image_batch, [64, 64])
                image_batch = (image_batch - 127.5) / 127.5 # normalized to [-1, 1]
                errD, errG = self.train_step(image_batch)
                filename = os.path.join(RESULT_DIR, 'minst_dcgan_tf_epoch_{}_batch_{}.png'.format(epoch, i))
                # self.generate_and_save_images(filename, self.generator, epoch + 1, seed)

                #Output training stats
                if (i%50) == 0:
                    print('epoch:[{}/{}] batch:[{}]\t Loss_D: {:.4f}\t Loss_G: {:.4f}'.format(
                        epoch, self.num_epochs, i,
                        errD, errG))

                if epoch == self.num_epochs-1 :
                    #save losses for plotting later
                    G_losses.append(errG)
                    D_losses.append(errD)

            if (epoch%5) ==0:
                filename = os.path.join(RESULT_DIR, 'minst_dcgan_tf_epoch_{}.png'.format(epoch))
                self.generate_and_save_images(filename, self.generator, epoch + 1, seed)

            # 每 15 个 epoch 保存一次模型
            if (epoch + 1) % 10 == 0 or epoch == self.num_epochs-1:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)

        total_time = time.time() - start_time
        with open(RESULT_FILE, 'a') as f:
            f.write(f'\n\ntraining results:')
            f.write('\n total training time: \t {}'.format(total_time))
            f.write('\n final average D_loss and G_loss: {:.4f} / {:.4f}'.format(
                np.mean(tf.stack(D_losses)),
                np.mean(tf.stack(G_losses))))

        index = 0
        while os.path.exists(os.path.join(RESULT_DIR, 'minst_dcgan_tf_epoch_{}_{}.png'.format(self.num_epochs, index))):
            index += 1
        imgname = os.path.join(RESULT_DIR, 'minst_dcgan_tf_epoch_{}_{}.png'.format(self.num_epochs, index))
        # 最后一个 epoch 结束后生成图片
        self.generate_and_save_images(imgname, self.generator, epoch, seed)











