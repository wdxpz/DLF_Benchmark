# pip install transformers

import tensorflow as tf
import tensorflow_datasets as tfds
import transformers
from transformers import *

import time
import numpy as np
import pandas as pd
import re

# hyper-params
BUFFER_SIZE = 10000
BATCH_SIZE = 32
EPOCHS = 3
MAXLEN = 128
ALPHA = 1e-5
EPSILON = 1e-5

# load data
(train_data, test_data), info = tfds.load(
  'imdb_reviews/plain_text', split=(tfds.Split.TRAIN, tfds.Split.TEST), 
  as_supervised=True, with_info=True)

train_size = len(list(train_data))
test_size = len(list(test_data))
print('Train set size:', train_size)
print('Test set size:', test_size)

# encode text
# get tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def convert_example_to_feature(review):
  return tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=MAXLEN,
      pad_to_max_length=True, truncation=True,
      return_attention_mask=True
  )

# map to the expected input to TFBertForSequenceClassification
def map_example_to_dict(input_ids, attention_mask, token_type_ids, label):
  return {
      "input_ids": input_ids,
      "attention_mask": attention_mask,
      "token_type_ids": token_type_ids,
  }, label

def encode_examples(ds):
  input_ids_list = []
  attention_mask_list = []
  token_type_ids_list = []
  label_list = []
  
  for review, label in tfds.as_numpy(ds):
    bert_input = convert_example_to_feature(review.decode())
    input_ids_list.append(bert_input['input_ids'])
    attention_mask_list.append(bert_input['attention_mask'])
    token_type_ids_list.append(bert_input['token_type_ids'])
    label_list.append([label])
  
  return tf.data.Dataset.from_tensor_slices(
    (input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)

train_encoded = encode_examples(train_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_encoded = encode_examples(test_data).batch(BATCH_SIZE)

print('ENCODE SUCCESSFULLY')

# build model -> load directly
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

total_steps = int(train_size/BATCH_SIZE) * EPOCHS
optimizer, _ = transformers.create_optimizer(
  init_lr = ALPHA,
  num_train_steps=total_steps,
  num_warmup_steps=0,
  adam_epsilon=EPSILON
)
'''
optimizer = transformers.AdamWeightDecay(
    learning_rate=ALPHA, 
    epsilon=EPSILON, 
    clipnorm=1.0)
'''

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
    train_encoded,
    epochs=EPOCHS)
print('Total train time: %.3f mins.' % ((time.time()-start)/60))

# testing
start_test = time.time()
test_loss, test_acc = model.evaluate(test_encoded)
print('On test set: loss - %.5f, acc - %.5f' % (test_loss, test_acc))
print('Total test time: %.3f secs.' % (time.time() - start_test))