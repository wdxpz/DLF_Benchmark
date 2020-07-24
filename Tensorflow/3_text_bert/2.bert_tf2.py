# 预安装 bert-for-tf2
# 使用下载好的数据，链接为 https://github.com/laxmimerit/IMDB-Movie-Reviews-Large-Dataset-50k.git
# 数据格式为.xlsx，IMDB 的评论数据，train-25k, test-25k

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa
# import keras

import numpy as np
import pandas as pd
import os
import json
import time

from sentencepiece import re
import bert

# hyper-params
BATCH_SIZE = 32
EPOCHS = 3
MAXLEN = 128
ALPHA = 1e-5
DECAY = 1e-6

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
  # remove single character
  # sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
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
test_labels = test_data['Sentiment']
test_labels = np.array(list(map(lambda x: 1 if x=="pos" else 0, test_labels)))


# pretrained bert encoder
hub_url_bert = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
bert_layer = hub.KerasLayer(hub_url_bert, trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
# set up tokenizer
tokenizer = bert.bert_tokenization.FullTokenizer(
    vocabulary_file, to_lower_case
)

def tokenize_text(text):
  return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

# encode reviews
train_revs_encode = [tokenize_text(rev) for rev in train_reviews]
test_revs_encode = [tokenize_text(rev) for rev in test_reviews]

print('ENCODE SUCCESSFULLY')


# pad sequences for input
train_x = tf.keras.preprocessing.sequence.pad_sequences(
  train_revs_encode, maxlen=MAXLEN, dtype='int32',
  padding='post', truncating = 'post', value=0
)
train_y = np.expand_dims(train_labels, axis=1)

test_x = tf.keras.preprocessing.sequence.pad_sequences(
  test_revs_encode, maxlen=MAXLEN, dtype='int32',
  padding='post', truncating = 'post', value=0
)
test_y = np.expand_dims(test_labels, axis=1)


# build model
model_name = "uncased_L-12_H-768_A-12"
model_dir = bert.fetch_google_bert_model(model_name, ".models")
bert_params = bert.params_from_pretrained_ckpt(model_dir)
l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")

print('PARAMETERS LOADED.')

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(MAXLEN, )))
model.add(l_bert)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.build(input_shape=(None, MAXLEN))

model.compile(
    loss='binary_crossentropy',
    optimizer=tfa.optimizers.AdamW(weight_decay=DECAY, learning_rate=ALPHA),
    metrics=['acc']
)

model.summary()

start = time.clock()
model.fit(
      train_x, train_y,
      batch_size=BATCH_SIZE,
      epochs=EPOCHS)
print('Total train time: %.3f' % (time.clock()-start))

model.save('models/dlf3.h5')

start_test = time.clock()
test_loss, test_acc = model.evaluate(
    test_x, test_y,
    batch_size=BATCH_SIZE
)
print('On test set: loss - %.5f, acc - %.5f' % (test_loss, test_acc))
print('Total test time: %.3f' % (time.clock() - start_test))
