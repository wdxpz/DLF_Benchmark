import os
os.environ['NVIDIA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['NVIDIA_VISIBLE_DEVICES'] = '1'

from DCGAN.dcgan import DCGAN_TF



num_repeat = 3

for i in range(num_repeat):
    dcgan = DCGAN_TF()
    dcgan.train()