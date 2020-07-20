from imageclassification.config import ImageClassifyConfig
from imageclassification.vgg16classifier import init, train, test

device, vgg16, CIFAR_train_loader, CIFAR_test_loader = init(ImageClassifyConfig)
train(device, vgg16, CIFAR_train_loader, ImageClassifyConfig)
test(device, vgg16, CIFAR_test_loader, ImageClassifyConfig)


