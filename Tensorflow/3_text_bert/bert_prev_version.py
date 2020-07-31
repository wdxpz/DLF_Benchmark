# pip install transformers

import tensorflow as tf
from transformers import *

import time
import numpy as np
import pandas as pd
import re

# hyper-params
BATCH_SIZE = 32
EPOCHS = 3
MAXLEN = 128
ALPHA = 1e-5
EPSILON = 1e-5

# load data
train_data = pd.read_excel('datasets/IMDB-Movie-Reviews-Large-Dataset-50k/train.xlsx')
test_data = pd.read_excel('datasets/IMDB-Movie-Reviews-Large-Dataset-50k/test.xlsx')
print('Training set:', train_data.shape)
print('Test set:', test_data.shape)

# initial processing
def process_text(sentence):
  # remove html tags
  TAG_RE = re.compile(r'<[^>]+>')
  sentence = TAG_RE.sub('', sentence)
  # remove punctuations and numbers
  sentence = re.sub('[^a-zA-Z]', ' ', sentence)
  # remove multiple spaces
  sentence = re.sub(r'\s+', ' ', sentence)
  return sentence

train_reviews = []
train_sens = list(train_data['Reviews'])
for sen in train_sens:
    train_reviews.append(process_text(str(sen)))

test_reviews = []
test_sens = list(test_data['Reviews'])
for sen in test_sens:
  test_reviews.append(process_text(str(sen)))

# load labels and convert text to number 0/1
train_labels = train_data['Sentiment']
train_labels = np.array(list(map(lambda x: 1 if x=="pos" else 0, train_labels)))
train_y = np.expand_dims(train_labels, axis=1)

test_labels = test_data['Sentiment']
test_labels = np.array(list(map(lambda x: 1 if x=="pos" else 0, test_labels)))
test_y = np.expand_dims(test_labels, axis=1)

# encode text
# get tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
def tokenize_text(text):
  return tokenizer.encode(
    text, add_special_tokens=True, 
    max_length=MAXLEN, pad_to_max_length=True, truncation=True)

train_revs_encode = [tokenize_text(rev) for rev in train_reviews]
test_revs_encode = [tokenize_text(rev) for rev in test_reviews]

train_x = np.asarray(train_revs_encode, dtype=np.int32)
test_x = np.asarray(test_revs_encode, dtype=np.int32)

print('ENCODE SUCCESSFULLY')

'''
train_x = tf.keras.preprocessing.sequence.pad_sequences(
  train_revs_encode, maxlen=MAXLEN, dtype='int32',
  padding='post', truncating = 'post', value=0
)
test_x = tf.keras.preprocessing.sequence.pad_sequences(
  test_revs_encode, maxlen=MAXLEN, dtype='int32',
  padding='post', truncating = 'post', value=0
)
'''

# build model -> load directly
model = TFBertForSequenceClassification.from_pretrained('bert-base-cased')

optimizer = tf.keras.optimizers.Adam(
    learning_rate=ALPHA, 
    epsilon=EPSILON, 
    clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(
    optimizer=optimizer, 
    loss=loss, 
    metrics=[metric])

model.summary()

# training
start = time.time()
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS)
print('Total train time: %.3f' % (time.time()-start))

# testing
start_test = time.time()
test_loss, test_acc = model.evaluate(
    test_x, test_y,
    batch_size=BATCH_SIZE
)
print('On test set: loss - %.5f, acc - %.5f' % (test_loss, test_acc))
print('Total test time: %.3f' % (time.time() - start_test))