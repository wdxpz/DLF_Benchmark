import tensorflow as tf
import keras
from keras import datasets
from keras import layers
from keras import models
from keras import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import time

# hyper-params
BATCH_SIZE = 64
EPOCHS = 20
VOCAB_SIZE = 32650  # 单词表长度
MAXLEN = 2000       # 评论的长度，超过截断
EMBED_DIM = 64

# load dataset
print('Loading data...')
(train_data, train_label),(test_data, test_label) = datasets.imdb.load_data(num_words=VOCAB_SIZE)
print(len(train_data), 'train sequences')
print(len(test_data), 'test sequences')

# pad input tuples
train_x = preprocessing.sequence.pad_sequences(train_data, maxlen=MAXLEN)
test_x = preprocessing.sequence.pad_sequences(test_data, maxlen=MAXLEN)
print('Training set input shape: ', train_x.shape)
print('Test set input shape: ', test_x.shape)

# build model
model = models.Sequential()
# Embedding layer
model.add(layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAXLEN))
# LSTM layer x 2
model.add(layers.Bidirectional(layers.LSTM(EMBED_DIM, return_sequences=True)))
# model.add(layers.Dropout(dropout_rate)) 不引入随机失活
model.add(layers.Bidirectional(layers.LSTM(EMBED_DIM, return_sequences=False)))
# FC layer  
model.add(layers.Dense(EMBED_DIM, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=['acc']
)

model.summary()

start_train = time.clock()
history = model.fit(
    train_x, train_label,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2
)
print('Total training time: %.3f' % (time.clock() - start_train))

start_test = time.clock()
test_loss, test_acc = model.evaluate(
    test_x, test_label,
    batch_size=BATCH_SIZE
)
print('On test set: loss - %.5f, acc - %.5f' % (test_loss, test_acc))
print('Total test time: %.3f' % (time.clock() - start_test))

'''
model.save('models/dlf2.h5')

# plt draw pics
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['acc']
val_acc = history_dict['val_acc']
eps = range(1, EPOCHS+1)
plt.figure(figsize=(20, 6))

plt.subplot(1, 2, 1)
plt.plot(eps, loss, 'bo', label='training loss')
plt.plot(eps, val_loss, 'b', label='validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(eps, acc, 'bo', label='training acc')
plt.plot(eps, val_acc, 'b', label='validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.savefig('pics/dlf2.png')
'''