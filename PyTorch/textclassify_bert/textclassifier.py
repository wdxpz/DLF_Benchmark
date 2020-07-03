import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup

from textclassify_bert.config import TextClassifier_Config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_FILE = 'text_bert_checkpoint_{}.pth'
RESULT_DIR = os.path.join(BASE_DIR, 'result')
RESULT_FILE = os.path.join(RESULT_DIR, 'text_bert_result.txt')

RANDOM_SEED = 123

class TextClassifier(object):

    def __init__(self, config=TextClassifier_Config):
        super(TextClassifier, self).__init__()

        self.bert_model = config['bert_model']
        self.max_sent_len = config['max_sentence_length']
        self.works = config['workers']
        self.ngpu = config['ngpu']
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.dropout = config['dropout']
        self.lr = config['lr']
        self.eps = config['epsilon']
        self.embed_size = config['embed_size']
        self.print_every = config['print_every']

        #device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # network
        self.net, self.tokenizer, self.bert_config = self._init_network()
        print(self.net, self.tokenizer, self.bert_config)
        #data
        self.train_data, self.test_data = self._init_data()


    def _init_network(self):
        '''
        PYTORCH-TRANSFORMERS
        * official site: https://pytorch.org/hub/huggingface_pytorch-transformers/
        * codelab site: https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/huggingface_pytorch-transformers.ipynb#scrollTo=Jk5LO-j1kXh3
        * github site: https://github.com/huggingface/transformers
        * bert model zoo:https://huggingface.co/transformers/pretrained_models.html

        * implementation and sample of modelForSequenceClassification:
        from line 1102 of https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bert.py

        *example of using modelForSequenceClassification for single sentence classification
        https://mccormickml.com/2019/07/22/BERT-fine-tuning/
        ###
        model loading in codelab and official site (by torch.hub) is different from github
        ###
        '''
        torch.hub.set_dir(MODEL_DIR)

        config = torch.hub.load('huggingface/pytorch-transformers', 'config', self.bert_model)
        config.num_labels = 2
        config.output_attentions = False
        config.output_hidden_states = False

        net = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', self.bert_model, config=config)
        net.to(self.device)

        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', self.bert_model)

        if self.device.type == 'cuda' and self.ngpu > 1:
            net = nn.DataParallel(net, list(range(self.ngpu)))

        return net, tokenizer, config

    def _init_data(self):

        TEXT = torchtext.data.Field(lower=True, include_lengths=True)  # necessary for packed_padded_sequence
        LABEL = torchtext.data.LabelField(dtype=torch.float)
        train_data, test_data = torchtext.datasets.IMDB.splits(TEXT, LABEL, root=DATA_DIR)
        # train_data, valid_data = org_train_data.split(random_state=random.seed(RANDOM_SEED), split_ratio=0.8)

        print(f'Num Train: {len(train_data)}')
        # print(f'Num Valid: {len(valid_data)}')
        print(f'Num Test: {len(test_data)}')

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True,
                                                   num_workers=self.works, collate_fn=self._convert_batch)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=True,
                                                  num_workers=self.works, collate_fn=self._convert_batch)

        return train_loader, test_loader

    def _convert_batch(self, batch):
        labels = torch.tensor([1 if entry.label.lower()=='pos' else 0 for entry in batch])

        texts = [self.tokenizer.encode(' '.join(entry.text), add_special_tokens=True, max_length=self.max_sent_len,
                                       pad_to_max_length=True) for entry in batch]

        attention_masks = [[int(token_id > 0) for token_id in seq] for seq in texts]

        texts = torch.tensor(texts)
        attention_masks = torch.tensor(attention_masks)

        return texts, attention_masks, labels

    def train(self):
        # for bertsequenceclassification model provide loss calculation, see: https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bert.py line 1171
        # criterion = nn.BCELoss().to(self.device)
        # optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        optimizer = AdamW(self.net.parameters(),
                          lr=self.lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=self.eps  # args.adam_epsilon  - default is 1e-8.
                          )

        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(self.train_data) * self.num_epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)

        self.net.train()

        train_accu = 0
        train_loss = 0
        start_time = time.time()

        for epoch in range(1, self.num_epochs+1):
            tsampels = 0
            for i, batch in enumerate(self.train_data):
                loss, accu, nsamples = self._train_func(batch, optimizer, scheduler)
                train_loss += loss
                train_accu += accu
                tsampels += nsamples

                if (i+1) % self.print_every == 0 or (i+1) == len(self.train_data):
                    print('epoch:[{}/{}] step:[{}/{}]\ttrain_loss: {:.4f}\ttrain_accu: {:.4f}'.format(
                        epoch, self.num_epochs, i+1, len(self.train_data),
                        train_loss/tsampels, train_accu/tsampels))
                    train_loss = 0
                    train_accu = 0
                    tsampels = 0



        total_time = time.time() - start_time
        with open(RESULT_FILE, 'a') as f:
            f.write(f'\ntraining results: \n')
            f.write('\n total training time: \t {}'.format(total_time))
            # f.write('\n final average training lost: \t {:.3f}'.format(train_loss))
            # f.write('\n final average training accuracy: \t{:.3f}'.format(train_accu))

    def _train_func(self, batch, optimizer, scheduler):
        optimizer.zero_grad()

        texts, attention_masks, labels = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)

        outputs = self.net(texts,
                           token_type_ids=None,
                           attention_mask=None,
                           labels=labels)
        loss, logits = outputs[:2]

        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

        _, predicts = torch.max(logits, 1)
        accu = (predicts == labels).sum().item()
        nsamples = labels.size()[0]

        return loss.item()*nsamples, accu, nsamples

    def test(self):
        self.net.eval()

        correct = 0
        total = 0

        start_time = time.time()
        with torch.no_grad():
            for i, batch in enumerate(self.train_data):
                texts, attention_masks, labels = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(
                    self.device)

                outputs = self.net(texts,
                                   token_type_ids=None,
                                   attention_mask=None,
                                   labels=labels)
                logits = outputs[1]
                _, predicts = torch.max(logits, 1)
                correct += (predicts == labels).sum().item()
                total += labels.size()[0]
        total_time = time.time() - start_time

        s = '\ntesting results: \n accuracy on {} test reviews: {:.3f}, total time: {} s\n'.format(total,
                                                                                                  correct * 1.0 / total,
                                                                                                  total_time)
        print(s)
        with open(RESULT_FILE, 'a') as f:
            f.write(s)






