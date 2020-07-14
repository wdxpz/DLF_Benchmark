import os
from DCGAN.dcgan import DCGAN_TF

os.environ['NVIDIA_VISIBLE_DEVICES'] = '0'

num_repeat = 3

for i in range(num_repeat):
    dcgan = DCGAN_TF()
    dcgan.train()