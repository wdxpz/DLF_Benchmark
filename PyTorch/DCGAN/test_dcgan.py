from dcgan import DCGAN

num_repeat = 3

for i in range(num_repeat):
    dcgan = DCGAN()
    dcgan.train()
