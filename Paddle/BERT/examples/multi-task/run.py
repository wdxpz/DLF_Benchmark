# coding=utf-8
import paddlepalm as palm
import json


if __name__ == '__main__':

    # configs
    max_seqlen = 128
    batch_size = 16
    num_epochs = 20
    print_steps = 5
    lr = 2e-5
    num_classes = 130
    weight_decay = 0.01
    num_classes_intent = 26
    dropout_prob = 0.1
    random_seed = 0
    label_map = './data/atis/atis_slot/label_map.json'
    vocab_path = './pretrain/ERNIE-v2-en-base/vocab.txt'

    train_slot = './data/atis/atis_slot/train.tsv'
    train_intent = './data/atis/atis_intent/train.tsv'

    config = json.load(open('./pretrain/ERNIE-v2-en-base/ernie_config.json'))
    input_dim = config['hidden_size']

    # -----------------------  for training ----------------------- 

    # step 1-1: create readers 
    seq_label_reader = palm.reader.SequenceLabelReader(vocab_path, max_seqlen, label_map, seed=random_seed)
    cls_reader = palm.reader.ClassifyReader(vocab_path, max_seqlen, seed=random_seed)

    # step 1-2: load train data
    seq_label_reader.load_data(train_slot, file_format='tsv', num_epochs=None, batch_size=batch_size)
    cls_reader.load_data(train_intent, batch_size=batch_size, num_epochs=None)

    # step 2: create a backbone of the model to extract text features
    ernie = palm.backbone.ERNIE.from_config(config)

    # step 3: register readers with ernie backbone
    seq_label_reader.register_with(ernie)
    cls_reader.register_with(ernie)

    # step 4: create task output heads
    seq_label_head = palm.head.SequenceLabel(num_classes, input_dim, dropout_prob)
    cls_head = palm.head.Classify(num_classes_intent, input_dim, dropout_prob)
   
    # step 5-1: create task trainers and multiHeadTrainer
    trainer_seq_label = palm.Trainer("slot", mix_ratio=1.0)
    trainer_cls = palm.Trainer("intent", mix_ratio=1.0)
    trainer = palm.MultiHeadTrainer([trainer_seq_label, trainer_cls])
    # # step 5-2: build forward graph with backbone and task head
    loss1 = trainer_cls.build_forward(ernie, cls_head)
    loss2 = trainer_seq_label.build_forward(ernie, seq_label_head)
    loss_var = trainer.build_forward()

    # step 6-1*: enable warmup for better fine-tuning
    n_steps = seq_label_reader.num_examples * 1.5 * num_epochs // batch_size
    warmup_steps = int(0.1 * n_steps)
    sched = palm.lr_sched.TriangularSchedualer(warmup_steps, n_steps)
    # step 6-2: build a optimizer
    adam = palm.optimizer.Adam(loss_var, lr, sched)
    # step 6-3: build backward graph
    trainer.build_backward(optimizer=adam, weight_decay=weight_decay)

    # step 7: fit readers to trainer
    trainer.fit_readers_with_mixratio([seq_label_reader, cls_reader], "slot", num_epochs)

    # step 8-1*: load pretrained model
    trainer.load_pretrain('./pretrain/ERNIE-v2-en-base')
    # step 8-2*: set saver to save models during training
    trainer.set_saver(save_path='./outputs/', save_steps=300)
    # step 8-3: start training
    trainer.train(print_steps=10)
