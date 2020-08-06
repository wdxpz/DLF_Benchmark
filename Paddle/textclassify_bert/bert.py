# coding=utf-8
import os
import time
import glob
import paddle
import paddlepalm as palm
import numpy as np
import json
import tarfile



def build_sample_file(imdb_dir, des_file, mode='train'):
    all_files = {}
    pos_dir = os.path.join(imdb_dir, 'aclImdb/{}/pos/*.txt'.format(mode))
    pos_text_files = glob.glob(pos_dir)
    neg_dir = os.path.join(imdb_dir, 'aclImdb/{}/neg/*.txt'.format(mode))
    neg_text_files = glob.glob(neg_dir)
    all_files = {
        '1': pos_text_files,
        '0': neg_text_files
        }
    with open(des_file, 'wt') as dst_f:
        dst_f.write('label\ttext_a\n')
        for label, texts in all_files.items():
            for text_file in texts:
                with open(text_file, "rt") as src_f:
                    content = src_f.read().replace('\t', ' ')
                    content = '{}\t{}\n'.format(label, content)
                    if content[:-1].find('\n') > 0:
                        raise Exception('sample text has more than 1 lines!')
                    if len(content.split('\t')) > 2:
                        print(len(content.split('\t')))
                        raise Exception('sample text has [tab]!')
                    dst_f.write(content)

    print('write total {} smaple in {} file'.format(len(pos_text_files)+len(neg_text_files), mode))


# configs
max_seqlen = 128
batch_size = 64
num_epochs = 3
lr = 1e-5
weight_decay = 1e-6
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, 'model/bert_model/pretrain/BERT-en-uncased-base')
checkpoint_dir = os.path.join(base_dir, 'model/bert_model/checkpoints')
data_dir = os.path.join(base_dir, 'data')
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
data_dir = os.path.join(data_dir, 'imdb')
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
vocab_path = os.path.join(model_dir, 'vocab.txt')


train_file = os.path.join(data_dir, 'train.tsv')
test_file = os.path.join(data_dir, 'test.tsv')
config = json.load(open(os.path.join(model_dir, 'bert_config.json')))
input_dim = config['hidden_size']
num_classes = 2
dropout_prob = 0.1
random_seed = 1
task_name = 'bert_imdb'
save_path = os.path.join(base_dir, 'result/bert')
pred_output = os.path.join(base_dir, 'result/bert/predict/')
pred_file = os.path.join(pred_output, 'predictions.json')
result_file = os.path.join(base_dir, 'result/results_bert.txt')
save_type = 'ckpt'
print_steps = 20
pre_params = os.path.join(model_dir, 'params')
save_steps = ( 25000 // batch_size ) * num_epochs 

trainer = palm.Trainer(task_name)

if not os.path.exists(train_file) or not os.path.exists(test_file):

    print("downloading and extracting IMDB data ....")
    URL = 'https://dataset.bj.bcebos.com/imdb%2FaclImdb_v1.tar.gz'
    MD5 = '7c2ac02c03563afcf9b574c7e56c153a'
    filename = paddle.dataset.common.download(URL, 'imdb', MD5)
    imdb_dir = os.path.dirname(filename)
    if not os.path.exists(os.path.join(imdb_dir, 'aclImdb')):
        tar = tarfile.open(filename)
        tar.extractall(imdb_dir) # specify which folder to extract to
        tar.close()

    print('build train file')
    build_sample_file(imdb_dir, train_file, 'train')
    print('buld test file')
    build_sample_file(imdb_dir, test_file, 'test')


def res_evaluate(predits_file=pred_file, test_file=test_file):
    labels = []
    with open(test_file, "r") as file:
        first_flag = True
        for line in file:
            line = line.split("\t")
            label = line[0]
            if label=='label':
                continue
            labels.append(str(label))
    file.close()

    preds = []
    with open(predits_file, "r") as file:
        for line in file.readlines():
            line = json.loads(line)
            pred = line['label']
            preds.append(str(pred))
    file.close()
    assert len(labels) == len(preds), "prediction result doesn't match to labels"
    print('data num: {}'.format(len(labels)))
    
    preds = np.array(preds)
    labels = np.array(labels)
    tp = np.sum((labels == '1') & (preds == '1'))
    fp = np.sum((labels == '0') & (preds == '0'))
    
    return (tp+fp)/(len(labels)*1.0)


def train():
    # -----------------------  for training ----------------------- 

    # step 1-1: create readers for training
    cls_reader = palm.reader.ClassifyReader(vocab_path, max_seqlen, seed=random_seed)
    # step 1-2: load the training data
    cls_reader.load_data(train_file, batch_size, num_epochs=num_epochs)

    # step 2: create a backbone of the model to extract text features
    ernie = palm.backbone.ERNIE.from_config(config)

    # step 3: register the backbone in reader
    cls_reader.register_with(ernie)

    # step 4: create the task output head
    cls_head = palm.head.Classify(num_classes, input_dim, dropout_prob)

    # step 5-1: create a task trainer
    
    # step 5-2: build forward graph with backbone and task head
    loss_var = trainer.build_forward(ernie, cls_head)

    # step 6-1*: use warmup
    n_steps = cls_reader.num_examples * num_epochs // batch_size
    warmup_steps = int(0.1 * n_steps)
    sched = palm.lr_sched.TriangularSchedualer(warmup_steps, n_steps)
    # step 6-2: create a optimizer
    adam = palm.optimizer.Adam(loss_var, lr, sched)
    # step 6-3: build backward
    trainer.build_backward(optimizer=adam, weight_decay=weight_decay)
  
    # step 7: fit prepared reader and data
    trainer.fit_reader(cls_reader)
    
    # step 8-1*: load pretrained parameters
    trainer.load_pretrain(pre_params)
    # step 8-2*: set saver to save model
    # save_steps = n_steps 
    # save_steps = 2396
    trainer.set_saver(save_steps=save_steps, save_path=checkpoint_dir, save_type=save_type)
    # step 8-3: start training
    start_time = time.time()
    trainer.train(print_steps=print_steps)
    duration = time.time()-start_time
    with open(result_file, 'a') as f:
        f.write('\n\ntraining results:')
        f.write('\ntraining time \t{}'.format(duration))
    trainer._save_ckpt


def test():
    # -----------------------  for prediction ----------------------- 

    # step 1-1: create readers for prediction
    print('prepare to predict...')
    predict_cls_reader = palm.reader.ClassifyReader(vocab_path, max_seqlen, seed=random_seed, phase='predict')
    # step 1-2: load the training data
    predict_cls_reader.load_data(test_file, batch_size)
    
    # step 2: create a backbone of the model to extract text features
    pred_ernie = palm.backbone.ERNIE.from_config(config, phase='predict')

    # step 3: register the backbone in reader
    predict_cls_reader.register_with(pred_ernie)
    
    # step 4: create the task output head
    cls_pred_head = palm.head.Classify(num_classes, input_dim, phase='predict')
    
    # step 5: build forward graph with backbone and task head
    trainer.build_predict_forward(pred_ernie, cls_pred_head)
 
    # step 6: load checkpoint
    # model_path = './outputs/ckpt.step'+str(save_steps)
    model_path = os.path.join(checkpoint_dir, 'ckpt.step'+str(save_steps))
    trainer.load_ckpt(model_path)

    # step 7: fit prepared reader and data
    trainer.fit_reader(predict_cls_reader, phase='predict')

    # step 8: predict
    print('predicting..')
    start_time = time.time()
    trainer.predict(print_steps=print_steps, output_dir=pred_output)
    duration = time.time()-start_time
    
    accuray = res_evaluate()
    with open(result_file, 'a') as f:
        f.write('\n\testing results:')
        f.write('\ntesting time \t{}'.format(duration))
        f.write('\ntesting accuracy \t{:.3f}\n'.format(accuray))


