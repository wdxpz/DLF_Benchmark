import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import matplotlib.pyplot as plt
import time

# hyper-params
BUFFER_SIZE = 10000
EPOCHS = 20
BATCH_SIZE = 64
EMBED_DIM = 64
DROPOUT = 0
ALPHA = 1e-4
VOCAB_SIZE = 32650

# load dataset
dataset, info = tfds.load('imdb_reviews/subwords32k', with_info=True, as_supervised=True)

train_dataset = dataset['train']
test_dataset = dataset['test']

train_size = len(list(train_dataset))
test_size = len(list(test_dataset))
print('Train set size:', train_size)    # 25000 examples
print('Test set size:', test_size)      # 25000 examples

# 每个batch分别pad，最后的长度不同
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset)).repeat()
test_dataset = test_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_dataset)).repeat()

# tokenizer = info.features['text'].encoder
# VOCAB_SIZE = tokenizer.vocab_size -> 32650

# build model
# 用于嵌入层和全连接层的初始化
embed_init = tf.keras.initializers.RandomUniform(-0.5, 0.5)
fc_init = tf.keras.initializers.RandomUniform(-0.5, 0.5)
bias_init = tf.keras.initializers.Zeros()

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM, embeddings_initializer=embed_init),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            EMBED_DIM, dropout=DROPOUT, return_sequences=True
            )),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            EMBED_DIM, dropout=DROPOUT, return_sequences=False
            )),
    tf.keras.layers.Dense(
        EMBED_DIM, activation='relu',
        kernel_initializer=fc_init, bias_initializer=bias_init),
    tf.keras.layers.Dense(
        1, activation='sigmoid',
        kernel_initializer=fc_init, bias_initializer=bias_init)
])

model.compile(
    loss = 'binary_crossentropy',
    optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA),
    metrics = ['acc']
)

model.summary()

# 取整计算每轮训练步数，避免报错：BaseCollectiveExecutor::StartAbort Out of range: End of sequence
steps = train_size // BATCH_SIZE

start_train = time.time()
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps
    # validation_data = test_dataset,
)
print('Total training time: %.3f mins' % ((time.time() - start_train)/60))

start_test = time.time()
test_loss, test_acc = model.evaluate(test_dataset, steps=steps)
print('On test set: loss - %.5f, acc - %.5f' % (test_loss, test_acc))
print('Total test time: %.3f secs' % (time.time() - start_test))

model.save('models/dlf2.h5')

'''
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