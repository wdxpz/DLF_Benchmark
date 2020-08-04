import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DCGAN_Config = {
    'workers': 1,
    'ngpu': 1,
    'batch_size': 128,
    'lr': 2e-4,
    'beta1': 0.5,
    'num_epochs': 20,
    'image_size': 64,
    'nz': 100,
    'nc': 1, #image channels
    'ngf': 64, #feature map size in generator
    'ndf': 64, #feature map size in discriminator
    'num_examples_to_generate': 64
}
