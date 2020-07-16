'''
tf官方教程，使用数据集 GLUE MRPC
需要安装
pip install tf-nightly
pip install tf-models-nightly
存在问题：基于IMDB数据集，数据处理存在问题
'''

import tensorflow as tf
import keras

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json

import tensorflow_hub as hub
import tensorflow_datasets as tfds

from official.modeling import tf_utils
from official import nlp
from official.nlp import bert

# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks

# hyper-params
BATCH_SIZE = 32
EPOCHS = 3
MAXLEN = 128
EMBED_DIM = 16
EPSILON = 1e-5

# directory contains: config, vocab, pretrained checkpoint
gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12"
tf.io.gfile.listdir(gs_folder_bert)
# pretrained bert encoder
hub_url_bert = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"


# load data
review, info = tfds.load(
    'imdb_reviews/plain_text', with_info=True)
# review.keys: ['test', 'train', 'unsupervised']

# set up tokenizer
tokenizer = bert.tokenization.FullTokenizer(
    vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
     do_lower_case=True)
print('Vocab size: ', len(tokenizer.vocab))

def encode_sentence(s, tokenizer):
   tokens = list(tokenizer.tokenize(s))
   tokens = tokens[:MAXLEN]
   tokens.append('[SEP]')
   return tokenizer.convert_tokens_to_ids(tokens)

def bert_encode(rev_dict, tokenizer):
  # num_examples = len(rev_dict["text"])
  sentence = tf.ragged.constant([
      encode_sentence(s, tokenizer)
      for s in np.array(rev_dict["text"])])

  cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence.shape[0]
  input_word_ids = tf.concat([cls, sentence], axis=-1)

  input_mask = tf.ones_like(input_word_ids).to_tensor()

  type_cls = tf.zeros_like(cls)
  type_s = tf.zeros_like(sentence)
  input_type_ids = tf.concat(
      [type_cls, type_s], axis=-1).to_tensor()

  inputs = {
      'input_word_ids': input_word_ids.to_tensor(),
      'input_mask': input_mask,
      'input_type_ids': input_type_ids}

  return inputs

rev_train = bert_encode(review['train'], tokenizer)
rev_train_labels = review['train']['label']

rev_test = bert_encode(review['test'], tokenizer)
rev_test_labels  = review['test']['label']


# build model
bert_config_file = os.path.join(gs_folder_bert, "bert_config.json")
config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())
bert_config = bert.configs.BertConfig.from_dict(config_dict)

bert_classifier, bert_encoder = bert.bert_models.classifier_model(
    bert_config, num_labels=2)

checkpoint = tf.train.Checkpoint(model=bert_encoder)
checkpoint.restore(
    os.path.join(gs_folder_bert, 'bert_model.ckpt')).assert_consumed()

train_size = len(rev_train_labels)
steps_per_epoch = int(train_size/BATCH_SIZE)
steps_warmup = int(EPOCHS * train_size * 0.1 / BATCH_SIZE)
optimizer = nlp.optimization.create_optimizer(
    1e-5, num_train_steps=steps_per_epoch*EPOCHS, num_warmup_steps=steps_warmup
)

# train the model
metrics = [keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

bert_classifier.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics)

start = time.clock()
bert_classifier.fit(
      rev_train, rev_train_labels,
      batch_size=BATCH_SIZE,
      epochs=EPOCHS,
      validation_split=0.2)
print('Total train time: %.3f' % (time.clock()-start))


'''
start_test = time.clock()
test_loss, test_acc = model.evaluate(
    test_x, test_label,
    batch_size=BATCH_SIZE
)
print('On test set: loss - %.5f, acc - %.5f' % (test_loss, test_acc))
print('Total test time: %.3f' % (time.clock() - start_test))


model.save('models/dlf3.h5')
'''