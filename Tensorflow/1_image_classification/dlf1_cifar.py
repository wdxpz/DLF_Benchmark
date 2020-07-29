import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import time

# hyper-parameters
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 128
EPOCHS = 30
ALPHA = 1e-4

# load dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
print('Training set size: %d'%(len(train_images)))
print('Test set size: %d'%(len(test_images)))
print('Image size: '+str(train_images[0].shape))

# Normalization
train_images = train_images/255.
test_images = test_images/255.
# One-hot encode
train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

# print(train_images.shape) -> (50000, 32, 32, 3)
# print(train_labels.shape) -> (50000, 10)

# load model VGG16
vgg16 = keras.applications.VGG16(
    include_top=False, 
    weights=None,
    input_shape=(32, 32, 3),
)
vgg16.summary()

model = keras.models.Sequential()
model.add(vgg16)
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(4096, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(4096, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(lr=ALPHA),
    metrics=['acc'])

start_train = time.time()
history = model.fit(
    train_images, train_labels,
    batch_size=TRAIN_BATCH_SIZE, epochs=EPOCHS,
    shuffle=True
)
print('Training accomplished. Total time: %.3f' % (time.time()-start_train))

start_eval = time.time()
test_loss, test_acc = model.evaluate(
    test_images, test_labels,
    batch_size=TEST_BATCH_SIZE
)
print('On test set: loss: %.5f, acc: %.5f' % (test_loss, test_acc))
print('Test accomplished. Total time: %.3f' % (time.time()-start_eval))

model.save('models/dlf1.h5')

# plt draw pics
history_dict = history.history
loss_vals = history_dict['loss']
acc_vals = history_dict['acc']
eps = range(1, EPOCHS+1)
plt.figure(figsize=(20, 6))

plt.subplot(1, 2, 1)
plt.plot(eps, loss_vals)
plt.title('Training loss')
plt.xlabel('epochs')
plt.ylabel('loss')

plt.subplot(1, 2, 2)
plt.plot(eps, acc_vals)
plt.title('Training accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')

plt.savefig('pics/dlf1.png')