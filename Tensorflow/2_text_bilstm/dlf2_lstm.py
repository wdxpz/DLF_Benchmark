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

train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
test_dataset = test_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_dataset))

# tokenizer = info.features['text'].encoder
# VOCAB_SIZE = tokenizer.vocab_size -> 32650

# build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            EMBED_DIM, dropout=DROPOUT, return_sequences=True
            )),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            EMBED_DIM, dropout=DROPOUT, return_sequences=False
            )),
    tf.keras.layers.Dense(EMBED_DIM, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss = 'binary_crossentropy',
    optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA),
    metrics = ['acc']
)

model.summary()

start_train = time.time()
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data = test_dataset
)
print('Total training time: %.3f mins' % ((time.time() - start_train)/60))

start_test = time.time()
test_loss, test_acc = model.evaluate(test_dataset)
print('On test set: loss - %.5f, acc - %.5f' % (test_loss, test_acc))
print('Total test time: %.3f secs' % (time.time() - start_test))

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